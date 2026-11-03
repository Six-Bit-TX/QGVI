"""Deterministic fixed/rule controllers sharing the agent's tools and limits.

Policies make no external calls. A judge-only call supplies perception, never an
action choice. Source: NetVLAD then full-image PnP; rule adds one central-ROI PnP
attempt after failed support. Target: BM25 and NetVLAD, with rank interleaving
(never comparison of their score magnitudes). Fixed renders three references,
left/right probes and a +15-degree yaw proposal, then judges once. Rule judges
each render, stops on host-accepted support, explores a supported high-coverage
anchor, otherwise tries the next reference. These are frozen development
heuristics, not claimed optimal policies or calibrated rejection thresholds.
"""
from __future__ import annotations

import copy
import math

from protocol import ALLOWED, STAGES, action

VERSION = 'qgvi_deterministic_baselines_v1'
CENTER_ROI = [0.1, 0.1, 0.9, 0.9]
RULE_EXPLORATION_COVERAGE = 0.5
DIRECT_YAW_DEGREES = 15.0


def choice(selected=None, *, judge_only=False, forced_action=None):
    return {'action': selected, 'judge_only': judge_only, 'forced_action': forced_action}


def finish(state):
    ident = state.get('best_answer_id')
    return choice(action('finish', classification='answer' if ident else 'insufficient_evidence', answer_id=ident))


def remaining(state):
    return max(0, state['config']['max_decisions']-state.get('decisions', 0))


def event_action(event):
    name = event.get('action')
    return name.get('action') if isinstance(name, dict) else name


def policy_state(method, state):
    policy = state.setdefault('policy', {'version': VERSION, 'method': method,
        'searches': {}, 'search_issued': [], 'source_attempts': [], 'render_attempts': 0,
        'judged_answers': [], 'render_records': [], 'explored': {}, 'candidate_attempted': [],
        'fixed_cursor': 0})
    if policy.get('version') != VERSION or policy.get('method') != method:
        raise ValueError('Baseline policy binding changed')
    pending = policy.pop('pending_search', None)
    if pending:
        events = state.get('events', [])[pending['event_offset']:]
        match = next((e for e in reversed(events) if event_action(e) == pending['tool']), None)
        ids = (match or {}).get('candidates', [])
        if not ids and isinstance((match or {}).get('result'), dict):
            ids = match['result'].get('candidates', [])
        ids = [v.get('view_id') if isinstance(v, dict) else v for v in ids]
        # Failing retrieval cannot silently inherit another route's candidates.
        if match and match.get('status') == 'error':
            ids = []
        elif match is None:
            ids = [k for k in state.get('candidates', {}) if k not in pending['before_candidates']]
            policy.setdefault('warnings', []).append('retrieval_event_missing_used_new_candidate_ids_only')
        policy['searches'][pending['tool']] = list(dict.fromkeys(i for i in ids if i in state.get('candidates', {})))
    pending = policy.pop('pending_render', None)
    if pending:
        events = state.get('events', [])[pending['event_offset']:]
        event = next((e for e in reversed(events) if event_action(e) == pending['tool']), None)
        ident = (event or {}).get('answer_id')
        if ident is None and event is None:
            fresh = [k for k, v in state.get('answers', {}).items()
                     if k not in pending['before_answers'] and v.get('kind') == 'target_view']
            ident = fresh[-1] if fresh else None
        if ident not in state.get('answers', {}) or state['answers'].get(ident, {}).get('kind') != 'target_view':
            ident = None
        record = {k: pending[k] for k in ('tool', 'anchor', 'operation')}
        record.update(answer_id=ident, status=(event or {}).get('status', 'not_reported'))
        policy['render_records'].append(record)
        policy['last_render'] = record
    return policy


def ranked_candidates(policy):
    text_ids = policy['searches'].get('search_text', [])
    image_ids = policy['searches'].get('search_image', [])
    order = []
    for rank in range(max(len(text_ids), len(image_ids))):
        for ids in (text_ids, image_ids):
            if rank < len(ids) and ids[rank] not in order:
                order.append(ids[rank])
    return order


def available(state, tool):
    counts = state.get('tool_counts', {})
    if counts.get(tool, 0) >= state['config']['tool_limits'].get(tool, 0):
        return False
    observations = len(state.get('observations', {}))
    extra = state['config']['retrieval_limit'] if tool in ('search_text', 'search_image') else 1
    if tool == 'estimate_pose':
        extra = 0
    if observations+extra > state['config']['max_observations']:
        return False
    if tool in ('navigate', 'propose_view'):
        used = max(counts.get('navigate', 0)+counts.get('propose_view', 0), state['policy']['render_attempts'])
        if used >= min(6, state['config'].get('max_renders', 6)):
            return False
    return True


def stage_for(state, name):
    if name in ALLOWED[state['stage']]:
        return None
    for stage in STAGES[STAGES.index(state['stage'])+1:]:
        if name in ALLOWED[stage]:
            selected = action('advance'); selected['next_stage'] = stage
            return choice(selected)
    return finish(state)


def search(state, query, policy, tool, *, after=None):
    transition = stage_for(state, tool)
    if transition:
        return transition
    if not available(state, tool):
        policy['search_issued'].append(tool); policy['searches'][tool] = []
        return None
    policy['search_issued'].append(tool)
    policy['pending_search'] = {'tool': tool, 'event_offset': len(state.get('events', [])),
                                'before_candidates': list(state.get('candidates', {}))}
    selected = action(tool, **({'query': query['instruction'][:2000]} if tool == 'search_text' else {}))
    selected['next_stage'] = after
    return choice(selected)


def direct_yaw(frame):
    """Rotate +15 degrees about the known camera's local +y; keep its center."""
    pose = copy.deepcopy(frame['c2w'])
    if len(pose) != 4 or any(len(row) != 4 for row in pose) or not all(
            type(v) in (int, float) and math.isfinite(v) for row in pose for v in row):
        raise ValueError('Invalid known camera for deterministic proposal')
    radians = math.radians(DIRECT_YAW_DEGREES)
    c, s = math.cos(radians), math.sin(radians)
    rotation = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
    for row in range(3):
        pose[row][:3] = [sum(frame['c2w'][row][k]*rotation[k][col] for k in range(3)) for col in range(3)]
    return pose


def render(state, policy, anchor, operation, *, after=None):
    tool = 'navigate' if operation in ('left', 'right') else 'propose_view'
    transition = stage_for(state, tool)
    if transition:
        return transition, False
    if not available(state, tool) or remaining(state) < 2:
        return None, False
    frame = state.get('candidates', {}).get(anchor, state.get('answers', {}).get(anchor, {})).get('frame')
    if frame is None:
        return None, False
    if operation == 'direct_yaw':
        selected = action(tool, view_id=anchor, pose_c2w=direct_yaw(frame))
    elif operation in ('left', 'right'):
        selected = action(tool, view_id=anchor, movement=operation)
    else:
        selected = action(tool, view_id=anchor)
    selected['next_stage'] = after
    policy['render_attempts'] += 1
    policy['pending_render'] = {'tool': tool, 'anchor': anchor, 'operation': operation,
        'event_offset': len(state.get('events', [])), 'before_answers': list(state.get('answers', {}))}
    return choice(selected), True


def judge(state, policy, *, final=False, answer_id=None):
    if answer_id and answer_id not in policy['judged_answers']:
        policy['judged_answers'].append(answer_id)
    forced = action('finish', classification='insufficient_evidence') if final else None
    # The host validates assessment, then resolves final answer from saved best.
    return choice(judge_only=True, forced_action=forced)


def source_policy(method, state, query, policy):
    if state.get('best_answer_id'):
        return finish(state)
    if 'search_image' not in policy['search_issued']:
        result = search(state, query, policy, 'search_image', after='verification')
        if result:
            return result
    attempts = policy['source_attempts']
    if not attempts or (method == 'rule' and len(attempts) == 1):
        transition = stage_for(state, 'estimate_pose')
        if transition:
            return transition
        if not available(state, 'estimate_pose') or remaining(state) <= 1:
            return finish(state)
        region = None if not attempts else list(CENTER_ROI)
        attempts.append(region)
        return choice(action('estimate_pose', region=region))
    return finish(state)


def fixed_target(state, policy):
    if 'fixed_plan' not in policy:
        ranked = ranked_candidates(policy)
        image_ids = policy['searches'].get('search_image', [])
        text_ids = policy['searches'].get('search_text', [])
        common = next((i for i in text_ids if i in image_ids), None)
        anchor = common or (image_ids[0] if image_ids else ranked[0] if ranked else None)
        policy['fixed_plan'] = [(i, 'reference') for i in ranked[:3]]
        if anchor:
            policy['fixed_plan'] += [(anchor, 'left'), (anchor, 'right'), (anchor, 'direct_yaw')]
    plan = policy['fixed_plan']
    while policy['fixed_cursor'] < len(plan):
        anchor, operation = plan[policy['fixed_cursor']]
        # Always preserve one decision to inspect/assess the rendered results.
        if remaining(state) < 2:
            break
        result, issued = render(state, policy, anchor, operation)
        if issued:
            policy['fixed_cursor'] += 1
        elif result is None:
            policy['fixed_cursor'] += 1
        if result:
            return result
    if any(record['answer_id'] for record in policy['render_records']) and not policy.get('fixed_judged'):
        policy['fixed_judged'] = True
        return judge(state, policy, final=True)
    return finish(state)


def rule_target(state, policy):
    if state.get('best_answer_id'):
        return finish(state)
    last = policy.get('last_render') or {}
    ident = last.get('answer_id')
    if ident and ident not in policy['judged_answers']:
        return judge(state, policy, final=remaining(state) <= 1, answer_id=ident)
    if remaining(state) < 2:
        return finish(state)
    answer = state.get('answers', {}).get(ident, {})
    assessment = (answer.get('assessments') or [None])[-1]
    coverage = answer.get('coverage')
    meaningful_support = assessment and assessment.get('anchor_supported') and any(
        p.get('status') == 'satisfied' for p in assessment.get('predicate_results', []))
    can_explore = meaningful_support and type(coverage) in (int, float) and math.isfinite(coverage) and coverage >= RULE_EXPLORATION_COVERAGE
    if can_explore:
        anchor = last['anchor']
        explored = policy['explored'].setdefault(anchor, [])
        for operation in ('left', 'right', 'direct_yaw'):
            if operation in explored:
                continue
            result, issued = render(state, policy, anchor, operation)
            if issued or result is None:
                explored.append(operation)
            if result:
                return result
    for candidate in ranked_candidates(policy):
        if candidate in policy['candidate_attempted']:
            continue
        result, issued = render(state, policy, candidate, 'reference', after='refinement')
        if issued or result is None:
            policy['candidate_attempted'].append(candidate)
        if result:
            return result
    return finish(state)


def choose(method, state, query):
    if method not in ('fixed', 'rule'):
        raise ValueError('Unknown deterministic baseline')
    if query['output_type'] not in ('source_camera_pose', 'target_view'):
        raise ValueError('Unsupported query output type')
    policy = policy_state(method, state)
    if remaining(state) <= 0 or state['stage'] == 'finalization':
        return finish(state)
    if query['output_type'] == 'source_camera_pose':
        return source_policy(method, state, query, policy)
    for tool in ('search_text', 'search_image'):
        if tool not in policy['search_issued']:
            if remaining(state) < 3:
                return finish(state)
            result = search(state, query, policy, tool, after='verification' if tool == 'search_image' else None)
            if result:
                return result
    return fixed_target(state, policy) if method == 'fixed' else rule_target(state, policy)
