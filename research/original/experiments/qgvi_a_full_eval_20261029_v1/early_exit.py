"""Opt-in ordinary-localizer prefix; failed verification hands evidence to the agent.

No tools or model requests are executed here. The controller journals every
selected action and charges its ordinary budgets. Target-view acceptance always
requires actual rendered pixels and every frozen predicate; retrieval scores and
source-pose geometry are candidate-generation signals, never semantic proof.
"""
from protocol import action
from io_utils import require

VERSION = 'ordinary_locator_verified_early_exit_v1'
VERIFICATION_PROMPT = '''This is an optional early-stop check, not a search-planning call.
Judge only early_exit_candidate_id against EVERY unchanged required predicate.
Use that candidate's actual Gaussian render and, when a source photograph is
present, the source pixels to verify identity and all source-relative constraints.
For every satisfied predicate, cite the candidate render observation ID and the
source observation ID when present. A retrieval score, caption or PnP inlier count
cannot establish semantic correctness. Mark unclear identity, visibility or
relations unknown; do not weaken or drop a predicate. Set anchor_supported false
when the source anchor cannot be identified. Failed or unknown checks continue
the same agent with the observations already collected. This is a controller
self-assessment, not an independent evaluation label. Return the required action
schema; the host uses only your assessment and reason and decides whether to
finish or continue. You do not choose retrieval or movement actions here.'''


def enabled(state): return state.get('config', {}).get('early_exit_enabled') is True


def handoff(state, reason):
    policy = state.setdefault('early_exit', {'version': VERSION, 'attempted': True})
    policy.update(phase='done', outcome='fallback_to_agent', reason=reason)
    policy.pop('pending', None)
    return None


def issue(state, policy, name, **arguments):
    if state['tool_counts'].get(name, 0) >= state['config']['tool_limits'].get(name, 0):
        return handoff(state, 'ordinary_tool_budget_unavailable')
    policy['attempted'] = True
    policy['pending'] = {'tool': name, 'event_offset': len(state['events'])}
    selected = action(name, **arguments)
    if name in {'search_text', 'search_image'}: selected['next_stage'] = 'verification'
    return {'action': selected, 'judge_only': False, 'forced_action': None, 'early_exit_prefix': True}


def consume(state, policy):
    pending = policy.pop('pending', None)
    if pending is None: return True
    events = state['events'][pending['event_offset']:]
    event = next((e for e in events if e.get('action') == pending['tool']), None)
    if not event or event.get('status') != 'complete':
        handoff(state, 'ordinary_tool_failed_or_unrecorded'); return False
    if pending['tool'] in {'search_text', 'search_image'}:
        ids = event.get('candidates', [])
        ids = [i for i in ids if i in state['candidates']]
        if not ids:
            handoff(state, 'ordinary_retrieval_empty'); return False
        policy['anchor_id'] = ids[0]
        policy['phase'] = 'pose' if policy['route'] == 'image' else 'render'
    elif pending['tool'] == 'estimate_pose':
        candidate = state['answers'].get(event.get('answer_id'), {})
        if candidate.get('kind') != 'source_camera_pose' or not candidate.get('accepted'):
            handoff(state, 'ordinary_source_geometry_unconfirmed'); return False
        policy['anchor_id'] = candidate['id']; policy['phase'] = 'source_finish' if state['output_type'] == 'source_camera_pose' else 'render'
    elif pending['tool'] == 'propose_view':
        candidate = state['answers'].get(event.get('answer_id'), {})
        if candidate.get('kind') != 'target_view' or not candidate.get('render_observation_id'):
            handoff(state, 'ordinary_target_render_missing'); return False
        policy['candidate_id'] = candidate['id']; policy['phase'] = 'verify'
    return True


def choose(state, query):
    """Return an ordinary journal-loop choice, or None for adaptive continuation."""
    if not enabled(state): return None
    if 'early_exit' not in state:
        route = state['config'].get('early_exit_route', 'auto')
        require(route in {'auto', 'text', 'image'}, 'invalid_early_exit_route')
        if query['output_type'] == 'source_camera_pose': route = 'image'
        elif route == 'auto': route = 'text'
        state['early_exit'] = {'version': VERSION, 'attempted': False, 'phase': 'search', 'route': route,
                               'outcome': None, 'reason': None}
        if state['stage'] != 'discovery' or state['decisions'] != 0:
            return handoff(state, 'not_initial_query_state')
        if query['output_type'] == 'target_view' and not query.get('predicates'):
            return handoff(state, 'target_predicates_required')
        if route == 'image' and not query.get('source_image'):
            return handoff(state, 'image_route_requires_source_pixels')
    policy = state['early_exit']
    if policy['phase'] == 'done': return None
    if not consume(state, policy): return None
    phase = policy['phase']
    if phase == 'search':
        return issue(state, policy, 'search_image') if policy['route'] == 'image' else issue(state, policy, 'search_text', query=query['instruction'])
    if phase == 'pose': return issue(state, policy, 'estimate_pose', region=None)
    if phase == 'render': return issue(state, policy, 'propose_view', view_id=policy['anchor_id'], pose_c2w=None)
    if phase == 'source_finish':
        policy.update(phase='done', outcome='early_exit', reason='existing_source_pose_geometric_gate')
        return {'action': action('finish', classification='answer', answer_id=policy['anchor_id']),
                'judge_only': False, 'forced_action': None, 'early_exit_prefix': True}
    if phase == 'verify':
        policy['phase'] = 'verifying'
        return {'action': None, 'judge_only': True, 'forced_action': action('advance'),
                'early_exit_judge': True, 'early_exit_candidate_id': policy['candidate_id'],
                'verification_prompt': VERIFICATION_PROMPT}
    return handoff(state, 'incomplete_verification_checkpoint')


def check_assessment(state, query, assessment, candidate_id, shown):
    """Strict additional evidence contract; standard schema validation runs first."""
    require(query['output_type'] == 'target_view', 'early_exit_verification_rejected')
    require(isinstance(assessment, dict) and assessment.get('answer_id') == candidate_id,
            'early_exit_verification_rejected')
    candidate = state['answers'].get(candidate_id, {})
    rendered = candidate.get('render_observation_id')
    source = state.get('source_observation_id') if query.get('source_image') else None
    visible = set(shown)
    require(candidate.get('kind') == 'target_view' and rendered in visible
            and state['observations'].get(rendered, {}).get('origin') == 'gaussian_render'
            and state['observations'][rendered].get('answer_id') == candidate_id,
            'early_exit_verification_rejected')
    if source:
        require(source in visible and state['observations'].get(source, {}).get('origin') == 'source',
                'early_exit_verification_rejected')
    elif query.get('source_image'):
        raise ValueError('early_exit_verification_rejected')
    expected = {p['id'] for p in query['predicates']}
    predicates = assessment.get('predicate_results', [])
    require(bool(expected) and len(predicates) == len(expected)
            and {p.get('predicate_id') for p in predicates} == expected, 'early_exit_verification_rejected')
    for predicate in predicates:
        evidence = set(predicate.get('evidence_ids', []))
        require(evidence <= visible, 'early_exit_verification_rejected')
        if predicate.get('status') == 'satisfied':
            require(rendered in evidence and (source is None or source in evidence),
                    'early_exit_verification_rejected')
    return assessment.get('anchor_supported') is True and all(p.get('status') == 'satisfied' for p in predicates)


def complete_verification(state, candidate_id, supported):
    policy = state['early_exit']
    if supported and state['answers'].get(candidate_id, {}).get('accepted'):
        policy.update(phase='done', outcome='early_exit', reason='all_frozen_predicates_visually_supported')
        return action('finish', classification='answer', answer_id=candidate_id)
    handoff(state, 'ordinary_candidate_failed_or_uncertain')
    selected = action('advance'); selected['next_stage'] = state['stage']
    return selected


def metrics(state):
    policy = state.get('early_exit', {})
    return {'enabled': enabled(state), 'attempted': policy.get('attempted', False),
            'early_exit': policy.get('outcome') == 'early_exit',
            'fallback_to_agent': policy.get('outcome') == 'fallback_to_agent',
            'route': policy.get('route'), 'reason': policy.get('reason'),
            'acceptance_scope': 'controller_decision_not_independent_evaluation'}
