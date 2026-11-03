"""Host-owned evidence, stage budgets, acceptance and deadline fallback."""
import copy
import time
from io_utils import require
from protocol import STAGES,ALLOWED,OBJECTIVES

def initial(query,config,started=None):
    return {'schema_version':1,'query_id':query['id'],'output_type':query['output_type'],
        'started_unix':time.time() if started is None else started,'stage':'discovery',
        'stage_spent':{s:0. for s in STAGES},'retrieval_returns':0,'decisions':0,
        'tool_counts':{},'observations':{},'candidates':{},'geometry':{},'answers':{},
        'seen_observations':[],'unresolved_requirements':[p['id'] for p in query['predicates']],
        'best_answer_id':None,'events':[],'config':copy.deepcopy(config)}

def remaining(state,clock=None):
    current=time.time() if clock is None else clock
    total=state['config']['total_seconds']-(current-state['started_unix'])
    stage=state['config']['stage_seconds'][state['stage']]-state['stage_spent'][state['stage']]
    return max(0.,total),max(0.,stage)

def control(state,clock=None):
    total,stage=remaining(state,clock)
    return {'stage':state['stage'],'time_remaining_seconds':round(total,2),
        'stage_remaining_seconds':round(stage,2),'objective':OBJECTIVES[state['stage']],
        'allowed_tools':ALLOWED[state['stage']],
        'unresolved_requirements':state['unresolved_requirements'],
        'best_answer_id':state['best_answer_id'],
        'decisions_remaining':max(0,state['config']['max_decisions']-state['decisions']),
        'tool_calls_used':state['tool_counts'],'tool_call_limits':state['config']['tool_limits'],
        'retrieval_returns_remaining':1-state['retrieval_returns']}

def transition(state,target):
    if target is None or target==state['stage']:return
    current=state['stage']
    if target=='discovery':
        require(current in ['verification','refinement'] and state['retrieval_returns']==0,'retrieval_return_limit')
        state['retrieval_returns']+=1
    else:require(STAGES.index(target)>STAGES.index(current),'invalid_stage_transition')
    state['stage']=target

def validate_assessment(state,query,assessment):
    if assessment is None:return
    require(query['output_type']=='target_view','semantic_assessment_not_source_pose')
    ident=assessment['answer_id'];candidate=state['answers'].get(ident)
    require(candidate is not None and candidate['kind']=='target_view','unknown_target_answer')
    rendered=candidate['render_observation_id']
    require(rendered in state['seen_observations'],'target_render_not_inspected')
    predicates=assessment['predicate_results'];expected={p['id'] for p in query['predicates']}
    require(len(predicates)==len(expected) and {p['predicate_id'] for p in predicates}==expected,'predicate_set_changed')
    for p in predicates:
        require(all(e in state['seen_observations'] for e in p['evidence_ids']),'unseen_semantic_evidence')
        if p['status']=='satisfied':require(rendered in p['evidence_ids'],'predicate_without_target_render')
    require(all(g in state['geometry'] for g in assessment['geometric_evidence_ids']),'unknown_geometry_evidence')
    supported=assessment['anchor_supported'] and all(p['status']=='satisfied' for p in predicates)
    candidate['assessments']=candidate.get('assessments',[])+[copy.deepcopy(assessment)]
    # A later failed refinement does not erase an earlier supported answer.
    if supported:
        candidate['accepted']=True;candidate['acceptance_basis']='controller_visual_predicate_assessment_not_independent_ground_truth'
        if state['best_answer_id'] is None or assessment['prefer_over_saved_answer']:state['best_answer_id']=ident
    else:
        candidate['accepted']=False
        if state['best_answer_id']==ident:
            alternatives=[a['id'] for a in state['answers'].values() if a['kind']==state['output_type'] and a.get('accepted')]
            state['best_answer_id']=alternatives[-1] if alternatives else None
    state['unresolved_requirements']=[p['predicate_id'] for p in predicates if p['status']!='satisfied']

def final_answer(state,classification='answer',answer_id=None,reason='explicit_finish'):
    ident=answer_id or state['best_answer_id']
    if classification=='answer':
        require(ident in state['answers'] and state['answers'][ident].get('accepted'),'answer_not_supported')
        answer=state['answers'][ident]
        require(answer['kind']==state['output_type'],'answer_type_mismatch')
        if answer['kind']=='source_camera_pose':require(answer.get('pose_origin')=='source_image_matching_and_PnP','source_pose_origin_invalid')
        return {'classification':'answer','answer_id':ident,'output_type':answer['kind'],
            'pose_c2w':answer['pose_c2w'],'coordinate_units':'native_model_units',
            'frame':answer.get('frame'),'render_observation_id':answer.get('render_observation_id'),
            'evidence_ids':answer.get('evidence_ids',[]),'acceptance_basis':answer.get('acceptance_basis'),
            'termination_reason':reason,'evaluation_correctness':None}
    require(answer_id is None,'abstention_has_answer')
    if classification=='out_of_map':
        certificates=[g for g in state['geometry'].values() if g.get('calibrated_out_of_map') is True and g.get('calibration_binding') and g.get('query_id')==state['query_id']]
        require(bool(certificates),'no_calibrated_out_of_map_certificate')
    else:require(classification=='insufficient_evidence','unknown_classification')
    return {'classification':classification,'answer_id':None,'output_type':state['output_type'],
            'pose_c2w':None,'termination_reason':reason,'evaluation_correctness':None}

def fallback(state,reason):
    return final_answer(state,'answer' if state['best_answer_id'] else 'insufficient_evidence',reason=reason)
