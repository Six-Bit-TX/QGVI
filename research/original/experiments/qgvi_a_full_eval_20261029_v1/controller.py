"""Shared staged runner with durable accepted answers and an external deadline.

The scheduler is implementation infrastructure, not a research contribution.
Completed decisions replay from journals; interrupted reservations are never
silently retried. No source ground truth or semantic labels enter inference.
"""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import time

from io_utils import HERE,ROOT,read,write,sha,digest,require,now
from protocol import PROMPT,SCHEMA,VERSION,validate,action
from state import initial,control,remaining,transition,validate_assessment,final_answer,fallback
from tool_host import ToolHost,public_state
from providers import make_provider,ProviderError,MODEL_OUTPUT_ERRORS,validate_effort,DEFAULT_TRANSPORTS,require_speed_config
import early_exit

class DeadlineExpired(BaseException):pass

# Invalid model actions are measured outcomes. Unexpected execution failures and
# changed evidence stop qualification immediately, preserving an accepted answer.
RECOVERABLE_ACTION_ERRORS={'ValidationError','unused_argument','invalid_search_query','missing_view_anchor',
    'missing_movement','missing_classification','missing_next_stage','invalid_region','retrieval_return_limit',
    'invalid_stage_transition','semantic_assessment_not_source_pose','unknown_target_answer',
    'target_render_not_inspected','predicate_set_changed','unseen_semantic_evidence',
    'predicate_without_target_render','unknown_geometry_evidence','answer_not_supported','answer_type_mismatch',
    'source_pose_origin_invalid','abstention_has_answer','no_calibrated_out_of_map_certificate',
    'unknown_classification','unknown_camera_anchor','tool_not_allowed_in_stage','tool_call_limit',
    'render_budget_exceeded','unknown_inspection_target','camera_has_no_render','empty_crop','unusable_render',
    'observation_budget_exceeded','invalid_unresolved_predicates','camera_out_of_bounds','invalid_camera','invalid_se3',
    'early_exit_verification_rejected','image_query_required','render_domain_unknown'}
NAVIGATION_INFRASTRUCTURE_ERRORS={'server_unavailable','server_http_error','server_error','worker_failure',
    'worker_error','cache_binding_or_pixels_changed','cache_metadata_changed','cache_camera_changed',
    'invalid_renderer_dimensions','invalid_renderer_values','invalid_renderer_time','wrong_scene',
    'missing_camera_bounds','unsupported_scene','unsupported_camera_domain','invalid_intrinsics',
    'invalid_dimensions','invalid_domain','incomplete_camera','invalid_step'}

@contextmanager
def deadline(seconds):
    if seconds<=0:raise DeadlineExpired()
    old=signal.getsignal(signal.SIGALRM)
    def expired(*_):raise DeadlineExpired()
    signal.signal(signal.SIGALRM,expired);previous=signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if previous[0]:signal.setitimer(signal.ITIMER_REAL,*previous)

def default_config(profile=False):
    return {'schema_version':VERSION,'profile_only':profile,'total_seconds':2400,
        'stage_seconds':{'discovery':600,'verification':600,'refinement':900,'finalization':300},
        'max_decisions':12,'max_renders':6,'max_observations':48,'max_images_per_decision':7,
        'retrieval_limit':6,'minimum_unique_inliers':40,'request_timeout_seconds':600,
        'tool_limits':{'search_text':2,'search_image':2,'inspect':4,'estimate_pose':2,'navigate':4,'propose_view':6},
        'rejection_calibration':'development_gate_only_no_out_of_map_certificate',
        'reasoning_effort':{'gpt':'low'},
        'provider_transport':dict(DEFAULT_TRANSPORTS),
        'speed':'normal','service_tier':{'gpt':'default'},'fast_mode':False,
        'early_exit_enabled':False,'early_exit_route':'auto'}

def code_binding():
    files=['controller.py','protocol.py','state.py','tool_host.py','providers.py','speed_provider.py','retrieval_backend.py','navigation_backend.py','baselines.py','early_exit.py','io_utils.py']
    return {n:sha(HERE/n) for n in files if (HERE/n).is_file()}

def safe_error(exc):
    known={'unused_argument','invalid_search_query','missing_view_anchor','missing_movement','missing_classification',
        'missing_next_stage','invalid_region','retrieval_return_limit','invalid_stage_transition','semantic_assessment_not_source_pose',
        'unknown_target_answer','target_render_not_inspected','predicate_set_changed','unseen_semantic_evidence',
        'predicate_without_target_render','unknown_geometry_evidence','answer_not_supported','answer_type_mismatch',
        'source_pose_origin_invalid','abstention_has_answer','no_calibrated_out_of_map_certificate','unknown_classification',
        'source_pixels_changed','unknown_camera_anchor','tool_not_allowed_in_stage','tool_call_limit','render_budget_exceeded',
        'unknown_inspection_target','camera_has_no_render','empty_crop','unusable_render','render_intrinsics_changed',
        'observation_budget_exceeded','observation_pixels_changed','invalid_unresolved_predicates'}
    if isinstance(exc,ProviderError):return exc.kind
    if type(exc) is ValueError and str(exc) in known|RECOVERABLE_ACTION_ERRORS:return str(exc)
    kind=getattr(exc,'kind',None)
    if isinstance(kind,str) and kind in RECOVERABLE_ACTION_ERRORS|NAVIGATION_INFRASTRUCTURE_ERRORS:return kind
    return type(exc).__name__

def load_journal(folder,binding,query,config):
    original=folder/'initial_state.json'
    if original.exists():
        saved=read(original);require(saved['binding']==binding,'Initial state binding changed');state=saved['state']
    else:
        state=initial(query,config);write(original,{'binding':binding,'state':state})
    last_hash=digest(state);step=1
    while (folder/'steps'/f'{step:03d}.json').exists():
        record=read(folder/'steps'/f'{step:03d}.json')
        require(record['binding']==binding and record['previous_state_sha256']==last_hash,'Journal chain changed')
        require(record['state_sha256']==digest(record['state']),'Saved state changed')
        state=record['state'];last_hash=record['state_sha256'];step+=1
    # A recorded assessment is a durable checkpoint even if the subsequent
    # tool or finalization was interrupted. It must extend this exact state.
    checkpoint=folder/'steps'/f'{step:03d}.checkpoint.json'
    if checkpoint.exists():
        record=read(checkpoint)
        require(record['binding']==binding and record['previous_state_sha256']==last_hash and record['state_sha256']==digest(record['state']),'Assessment checkpoint changed')
        state=record['state']
    interrupted=(folder/'steps'/f'{step:03d}.reserved.json').exists()
    return state,step,interrupted

def run_one(query,method,model,folder,config,*,provider=None,host=None):
    # The effort belongs to the request configuration, not the prompt. Keeping
    # prompts identical makes effort the only controller variable within model.
    tier=require_speed_config(config)
    effort=validate_effort(model,config.get('reasoning_effort',{}).get(model,'high'))
    transport=config.get('provider_transport',{}).get(model,DEFAULT_TRANSPORTS[model])
    if provider is not None:
        if hasattr(provider,'reasoning_effort') and provider.reasoning_effort!=effort:
            raise ValueError('Injected provider reasoning effort does not match frozen configuration')
        if hasattr(provider,'transport') and provider.transport!=transport:
            raise ValueError('Injected provider transport does not match frozen configuration')
        if hasattr(provider,'service_tier') and provider.service_tier!=tier:
            raise ValueError('Injected provider service tier does not match frozen speed configuration')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    binding=digest({'query':query,'config':config,'method':method,'model':model,'code':code_binding()})
    result_file=folder/'result.json'
    if result_file.exists():
        result=read(result_file);require(result['binding']==binding,'Completed result binding changed');replay(folder);return result
    resumed_from_existing_journal=(folder/'initial_state.json').exists()
    start=time.perf_counter()
    state,step,interrupted=load_journal(folder,binding,query,config)
    model_seconds=0.;tool_seconds=0.;model_calls=0;termination=None;output=None
    if interrupted:output=fallback(state,'interrupted_reserved_step_not_retried');termination='interrupted'
    if host is None and output is None:host=ToolHost(query,folder)
    if provider is None and output is None:provider=make_provider(model,speed=config['speed'],reasoning_effort=effort,transport=transport)
    if not state['observations'] and output is None:host.initialize(state)
    # Initial pixels belong in the first journal parent, not in an unbound
    # mutable setup file. Persist initialization before the first reservation.
    if step==1 and not interrupted:
        write(folder/'initial_state.json',{'binding':binding,'state':state})
    while output is None and state['decisions']<config['max_decisions']:
        left,stage_left=remaining(state)
        if min(left,stage_left)<=0:output=fallback(state,'deadline');termination='timeout';break
        before=copy.deepcopy(state);previous_hash=digest(state);step_started=time.perf_counter();stage=state['stage']
        stem=folder/'steps'/f'{step:03d}';model_record=None;observation=None;selected=None;error=None;choice=None
        before_action=None;request_elapsed=0.;request_attempted=False;tool_attempted=False;tool_elapsed=0.
        write(stem.with_suffix('.reserved.json'),{'binding':binding,'previous_state_sha256':previous_hash,'reserved_at':now()})
        try:
            with deadline(min(left,stage_left,config['request_timeout_seconds'])):
                if method=='agent':choice=early_exit.choose(state,query) or {'action':None,'judge_only':False,'forced_action':None}
                else:
                    from baselines import choose
                    choice=choose(method,state,query)
                selected=choice.get('action')
                if selected is None or choice.get('judge_only'):
                    image,shown=host.contact_sheet(state,step)
                    payload=public_state(state,query,control(state));payload['visible_observation_ids']=shown
                    prompt=PROMPT
                    if choice.get('judge_only') and not choice.get('early_exit_judge'):
                        prompt+='\nThis is a perception-only baseline call. Assess the best rendered target against all predicates. Do not select retrieval or movement actions. The host applies its fixed/rule-based action; your assessment is used without changing that policy.'
                        payload['fixed_controller_action']=choice.get('forced_action')
                    if choice.get('early_exit_judge'):
                        prompt+='\n'+choice['verification_prompt']
                        payload['early_exit_candidate_id']=choice['early_exit_candidate_id']
                        payload['early_exit_source_observation_id']=state.get('source_observation_id')
                    write(stem.with_suffix('.input.json'),{'binding':binding,'system':prompt,'user':payload,'image_sha256':sha(image)})
                    provider.timeout=min(600,config['request_timeout_seconds'],max(1,min(*remaining(state))))
                    model_calls+=1;request_started=time.perf_counter();request_attempted=True
                    try:model_record=provider.request(image,prompt,payload,SCHEMA)
                    finally:
                        request_elapsed=time.perf_counter()-request_started;model_seconds+=request_elapsed
                    write(stem.with_suffix('.model.json'),model_record)
                    if model_record.get('reasoning_effort',effort)!=effort:
                        raise ProviderError('reported_reasoning_effort_mismatch')
                    if model_record.get('transport',transport)!=transport:
                        raise ProviderError('reported_transport_mismatch')
                    if model_record.get('requested_service_tier')!=tier or model_record.get('fast_mode_enabled') is not config['fast_mode']:
                        raise ProviderError('missing_speed_request_metadata')
                    selected=model_record['response']
                    state['seen_observations']=list(dict.fromkeys(state['seen_observations']+shown))
                    if choice.get('judge_only'):
                        fixed=copy.deepcopy(choice.get('forced_action')) or action('advance')
                        if fixed['action']=='advance':fixed['next_stage']=state['stage']
                        fixed['assessment']=selected['assessment'];fixed['reason']=selected['reason'];selected=fixed
                validate(selected)
                valid_predicates={p['id'] for p in query['predicates']}
                require(set(selected['unresolved_requirements'])<=valid_predicates,'invalid_unresolved_predicates')
                early_supported=None
                if choice.get('early_exit_judge'):
                    early_supported=early_exit.check_assessment(state,query,selected['assessment'],choice['early_exit_candidate_id'],shown)
                assessment_registered=selected['assessment'] is not None
                validate_assessment(state,query,selected['assessment'])
                if choice.get('early_exit_judge'):
                    selected=early_exit.complete_verification(state,choice['early_exit_candidate_id'],early_supported)
                if assessment_registered:
                    write(stem.with_suffix('.checkpoint.json'),{'binding':binding,'previous_state_sha256':previous_hash,'state':state,'state_sha256':digest(state)})
                before_action=copy.deepcopy(state)
                if choice.get('judge_only') and selected['action']=='finish':
                    selected['arguments']['classification']='answer' if state['best_answer_id'] else 'insufficient_evidence'
                    selected['arguments']['answer_id']=state['best_answer_id']
                name=selected['action']
                if name=='finish':
                    output=final_answer(state,selected['arguments']['classification'],selected['arguments']['answer_id']);termination='finished'
                elif name=='advance':transition(state,selected['next_stage'])
                else:
                    tool_started=time.perf_counter();tool_attempted=True
                    try:observation=host.execute(state,selected)
                    finally:
                        tool_elapsed=time.perf_counter()-tool_started;tool_seconds+=tool_elapsed
                    transition(state,selected['next_stage'])
                state['events'].append({'action':name,'status':'complete','answer_id':observation.get('answer_id') if observation else None,
                    'observation_ids':observation.get('new_observation_ids',[]) if observation else [],
                    'candidates':[c['view_id'] for c in observation.get('candidates',[])] if observation else [],
                    'coverage':observation.get('coverage') if observation else None})
        except DeadlineExpired:
            state=before_action or before;output=fallback(state,'deadline');termination='timeout';error='external_deadline'
        except ProviderError as exc:
            state=before;error=safe_error(exc)
            termination='model_output_error' if exc.kind in MODEL_OUTPUT_ERRORS else 'provider_error'
            failure_classification=termination
            if choice and choice.get('early_exit_judge') and exc.kind in MODEL_OUTPUT_ERRORS:
                early_exit.handoff(state,'ordinary_verifier_invalid_model_output')
                termination=None;output=None
            else:output=fallback(state,termination)
            write(stem.with_suffix('.failure.json'),{'binding':binding,'category':exc.kind,
                'classification':failure_classification,'safe_details':exc.safe_details,'model_wall_seconds':request_elapsed})
        except Exception as exc:
            state=before_action or before;error=safe_error(exc)
            state['events'].append({'action':selected.get('action') if isinstance(selected,dict) else 'invalid_response','status':'error','category':error})
            if choice and (choice.get('early_exit_judge') or choice.get('early_exit_prefix')) and error in RECOVERABLE_ACTION_ERRORS:
                early_exit.handoff(state,'ordinary_verifier_rejected_evidence' if choice.get('early_exit_judge') else 'ordinary_prefix_action_failed')
            if error not in RECOVERABLE_ACTION_ERRORS:
                output=fallback(state,'infrastructure_error');termination='infrastructure_error'
        if error and tool_attempted and selected['action'] in config['tool_limits']:
            # The handlers above restore the pre-action snapshot, undoing the
            # host's attempted-tool increment. Charge that failed attempt once.
            name=selected['action'];state['tool_counts'][name]=state['tool_counts'].get(name,0)+1
        state['decisions']+=1;state['stage_spent'][stage]+=time.perf_counter()-step_started
        record={'binding':binding,'previous_state_sha256':previous_hash,'state':state,'state_sha256':digest(state),
            'action':selected,'tool_result':observation,'error_category':error,'elapsed_seconds':time.perf_counter()-step_started,
            'model_wall_seconds':request_elapsed,'model_attempted':request_attempted,'tool_wall_seconds':tool_elapsed,'updated_at':now()}
        write(stem.with_suffix('.json'),record);write(folder/'state.json',state);step+=1
        write(folder/'progress.json',{'status':'running' if output is None else 'finishing','stage':state['stage'],
            'query_id':query['id'],'method':method,'model':model,'reasoning_effort':effort,'transport':transport,
            'decisions':state['decisions'],'best_answer_id':state['best_answer_id'],'updated_at':now()})
    if output is None:output=fallback(state,'decision_budget');termination='budget_exhausted'
    # Compute totals from durable records, so resumed timing is not reset.
    records=[read(p) for p in sorted((folder/'steps').glob('[0-9][0-9][0-9].json'))]
    calls=[read(p) for p in sorted((folder/'steps').glob('*.model.json'))]
    result={'schema_version':VERSION,'binding':binding,'query_id':query['id'],'scene':query['scene'],'method':method,
        'model':model,'requested_model':'gpt-6.1-sol' if model=='gpt' else 'deepseek-flash',
        'reasoning_effort':effort,'requested_reasoning_effort':effort,'transport':transport,
        'requested_speed':config['speed'],'requested_service_tier':tier,
        'served_service_tier':None,'fast_mode_enabled':config['fast_mode'],
        'recorded_request_efforts':sorted({r['reasoning_effort'] for r in calls if r.get('reasoning_effort')}),
        'effort_measurement_scope':'explicit request parameter; not an inference of hidden reasoning or cross-provider equivalent compute',
        'reported_models':sorted({r['reported_model'] for r in calls if r.get('reported_model')}),
        'status':'complete','termination':termination,'answer':output,'state':state,
        'timing':{'elapsed_wall_seconds':time.time()-state['started_unix'],'active_recorded_seconds':sum(r['elapsed_seconds'] for r in records),
            'resumed_from_existing_journal':resumed_from_existing_journal,
            'current_attempt_wall_seconds':time.perf_counter()-start,
            'model_seconds':sum(r.get('model_wall_seconds',0) for r in records),'this_attempt_model_seconds':model_seconds,
            'tool_seconds':sum(r.get('tool_wall_seconds',0) for r in records),
            'timing_scope':'development cache-reusing controller; resident setup and historical query feature preparation separate'},
        'model_calls':sum(bool(r.get('model_attempted')) for r in records),'successful_model_responses':len(calls),'reserved_steps':len(list((folder/'steps').glob('*.reserved.json'))),
        'tool_counts':state['tool_counts'],'early_exit':early_exit.metrics(state),'errors':[r['error_category'] for r in records if r['error_category']],
        'completed_at':now(),'annotation_status':'independent_model_assessment_pending_not_ground_truth'}
    result['result_sha256']=digest(result);write(result_file,result);return result

def replay(folder):
    folder=Path(folder);result=read(folder/'result.json');initial_saved=read(folder/'initial_state.json')
    require(result.get('result_sha256')==digest({k:v for k,v in result.items() if k!='result_sha256'}),'Completed result changed')
    previous=digest(initial_saved['state']);count=0
    for path in sorted((folder/'steps').glob('[0-9][0-9][0-9].json')):
        row=read(path);require(row['binding']==result['binding'] and row['previous_state_sha256']==previous,'Replay binding changed')
        require(row['state_sha256']==digest(row['state']),'Replay state changed');previous=row['state_sha256'];count+=1
    if digest(result['state'])!=previous:
        checkpoint=folder/'steps'/f'{count+1:03d}.checkpoint.json'
        require(checkpoint.is_file(),'Final state differs from journal')
        row=read(checkpoint)
        require(row['binding']==result['binding'] and row['previous_state_sha256']==previous and row['state_sha256']==digest(row['state']),'Replay checkpoint changed')
        require(digest(result['state'])==row['state_sha256'],'Final state differs from checkpoint')
    answer=result['answer'];expected=final_answer(result['state'],answer['classification'],answer['answer_id'],reason=answer['termination_reason'])
    require(answer==expected,'Final answer differs from supported state')
    return {'status':'passed','decisions_replayed':count,'model_requests':0,'tool_requests':0}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','replay']);p.add_argument('--task',type=Path);p.add_argument('--config',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--method',choices=['agent','fixed','rule'],default='agent');p.add_argument('--model',choices=['gpt','deepseek'],default='gpt')
    p.add_argument('--early-exit',action='store_true',help='Try ordinary localization and verify the complete target before adaptive search (agent method only).')
    p.add_argument('--early-exit-route',choices=['auto','text','image'],default=None,help='Candidate route for the optional early exit; setting a route alone does not enable it.')
    a=p.parse_args(argv)
    if a.command=='replay':
        if a.early_exit or a.early_exit_route is not None:p.error('Early-exit flags apply only to run; replay uses its frozen journal.')
        print(json.dumps(replay(a.out)))
    else:
        if a.task is None or a.config is None:p.error('run requires --task and --config')
        config=read(a.config)
        if a.early_exit:config['early_exit_enabled']=True
        if a.early_exit_route is not None:config['early_exit_route']=a.early_exit_route
        if a.method!='agent' and config.get('early_exit_enabled'):p.error('Early exit applies only to --method agent')
        r=run_one(read(a.task),a.method,a.model,a.out,config);print(json.dumps({'status':r['status'],'termination':r['termination'],'classification':r['answer']['classification']}))

if __name__=='__main__':main()
