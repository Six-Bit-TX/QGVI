"""Seven versioned tools, bound to query pixels and a shared native camera frame."""
from pathlib import Path
import time
import math
from PIL import Image,ImageOps,ImageDraw,ImageFont
from io_utils import HERE,read,write,sha,digest,require
from protocol import VERSION,ALLOWED

class ToolHost:
    def __init__(self,query,run,retrieval=None,navigation=None):
        self.query=query;self.run=Path(run);self.root=self.run/'assets'/query['id'];self.root.mkdir(parents=True,exist_ok=True)
        if retrieval is None:
            from retrieval_backend import RetrievalBackend
            retrieval=RetrievalBackend(query['scene'],HERE)
        self.retrieval=retrieval
        if hasattr(retrieval,'bind_source'):
            retrieval.bind_source(query.get('host_bundle_id') if query.get('source_image') else None)
        if navigation is None:
            from navigation_backend import NavigationClient
            navigation=NavigationClient(query['scene'])
        self.navigation=navigation

    def fixed_frame(self,frame):
        values={k:self.query.get(k) if self.query.get(k) is not None else frame.get(k)
                for k in ['K','width','height']}
        require(all(value is not None for value in values.values()),'invalid_camera')
        return {**{k:frame[k] for k in ['c2w','domain'] if k in frame},**values}

    def require_image_query(self):
        require(bool(self.query.get('source_image')) and bool(self.query.get('host_bundle_id')),
                'image_query_required')

    def add_observation(self,state,path,origin,*,answer_id=None,reference_id=None,region=None):
        path=Path(path).resolve();require(path.is_file(),'missing_observation')
        ident='obs_'+digest({'pixels':sha(path),'origin':origin,'query':self.query['id'],'answer':answer_id,'region':region})[:18]
        state['observations'][ident]={'id':ident,'image_path':str(path),'image_sha256':sha(path),
            'origin':origin,'answer_id':answer_id,'reference_id':reference_id,'region':region}
        return ident

    def initialize(self,state):
        if not self.query.get('source_image'):
            require(self.query['output_type']=='target_view','image_query_required')
            state['source_observation_id']=None
            return
        require(sha(self.query['source_image'])==self.query['source_image_sha256'],'source_pixels_changed')
        ident=self.add_observation(state,self.query['source_image'],'source')
        state['source_observation_id']=ident

    def anchor(self,state,ident):
        if ident in state['answers']:
            frame=state['answers'][ident]['frame']
            require(bool(frame.get('domain')),'render_domain_unknown')
            return self.fixed_frame(frame)
        require(ident in state['candidates'],'unknown_camera_anchor')
        return self.fixed_frame(state['candidates'][ident]['frame'])

    def execute(self,state,action):
        name=action['action'];args=action['arguments'];started=time.perf_counter()
        require(name in ALLOWED[state['stage']],'tool_not_allowed_in_stage')
        if name in ['search_image','estimate_pose'] or (name=='inspect' and args['view_id']=='SOURCE'):
            self.require_image_query()
        counts=state['tool_counts'];require(counts.get(name,0)<state['config']['tool_limits'].get(name,0),'tool_call_limit')
        if name in ['navigate','propose_view']:
            require(sum(counts.get(k,0) for k in ['navigate','propose_view'])<state['config']['max_renders'],'render_budget_exceeded')
        counts[name]=counts.get(name,0)+1
        result={'schema_version':VERSION,'tool':name,'new_observation_ids':[]}
        if name in ['search_text','search_image']:
            if name=='search_text':raw=self.retrieval.search_text(args['query'],limit=state['config']['retrieval_limit'],
                bundle_id=self.query.get('host_bundle_id') if self.query.get('source_image') else None)
            else:raw=self.retrieval.search_image(self.query['host_bundle_id'],limit=state['config']['retrieval_limit'])
            candidates=[]
            for candidate in raw['candidates']:
                c=dict(candidate);ident=c['view_id'];state['candidates'][ident]=c
                obs=self.add_observation(state,c['image_path'],'retrieved_reference',reference_id=ident)
                result['new_observation_ids'].append(obs)
                candidates.append({k:c[k] for k in ['view_id','score','caption','frame'] if k in c}|{'observation_id':obs})
            result.update(candidates=candidates,backend=raw.get('backend'),score_meaning=raw.get('score_meaning'),backend_timing=raw.get('timing'))
        elif name=='inspect':
            ident=args['view_id'];obs=state['observations'].get(ident)
            if ident=='SOURCE':obs=state['observations'][state['source_observation_id']]
            if obs is not None:source=obs['image_path'];answer_id=obs.get('answer_id')
            elif ident in state['answers']:
                answer=state['answers'][ident];require(answer.get('render_observation_id'),'camera_has_no_render')
                obs=state['observations'][answer['render_observation_id']];source=obs['image_path'];answer_id=ident
            else:
                require(ident in state['candidates'],'unknown_inspection_target');source=state['candidates'][ident]['image_path'];answer_id=None
            r=args['region'] or [0,0,1,1]
            with Image.open(source) as im:
                rgb=ImageOps.exif_transpose(im).convert('RGB');box=[round(r[0]*rgb.width),round(r[1]*rgb.height),round(r[2]*rgb.width),round(r[3]*rgb.height)]
                require(box[0]<box[2] and box[1]<box[3],'empty_crop')
                path=self.root/(digest({'source':sha(source),'region':r})+'.png');rgb.crop(tuple(box)).save(path)
            oid=self.add_observation(state,path,'inspection_crop',answer_id=answer_id,region=r)
            result.update(new_observation_ids=[oid],observation_id=oid,region=r,original_view_id=ident)
        elif name=='estimate_pose':
            raw=self.retrieval.estimate_pose(self.query['host_bundle_id'],region=args['region'])
            gid='geom_'+digest({'query':self.query['id'],'region':args['region'],'pose':raw.get('pose_c2w')})[:18]
            public={k:raw[k] for k in ['status','pose_c2w','evidence','coordinate_units','roi','roi_semantics','compute_seconds','pose_origin'] if k in raw}
            public.update(id=gid,query_id=self.query['id'],calibrated_out_of_map=False)
            state['geometry'][gid]=public
            n=raw.get('evidence',{}).get('unique_inlier_queries',0)
            supported=raw.get('pose_c2w') is not None and n>=state['config']['minimum_unique_inliers']
            aid='source_'+digest({'query':self.query['id'],'geometry':gid})[:16]
            if raw.get('pose_c2w') is not None:
                native={'c2w':raw['pose_c2w']}
                if self.query.get('domain'):native['domain']=self.query['domain']
                frame=self.fixed_frame(native)
                state['answers'][aid]={'id':aid,'kind':'source_camera_pose','pose_c2w':raw['pose_c2w'],'frame':frame,
                    'accepted':supported,'pose_origin':'source_image_matching_and_PnP','evidence_ids':[gid],
                    'acceptance_basis':'development_distinct_query_inlier_gate_not_calibrated_for_adaptive_policy'}
                if supported and self.query['output_type']=='source_camera_pose':state['best_answer_id']=aid
            result.update(geometry=public,answer_id=aid if raw.get('pose_c2w') else None,accepted=supported)
        elif name in ['navigate','propose_view']:
            frame=self.anchor(state,args['view_id'])
            if name=='propose_view':
                if args['pose_c2w'] is not None:
                    from navigation_backend import validate_pose
                    try:validate_pose(args['pose_c2w'])
                    except (ValueError,TypeError):raise ValueError('invalid_se3') from None
                    frame={**frame,'c2w':args['pose_c2w']}
                raw=self.navigation.propose_view(frame)
            else:raw=self.navigation.navigate(frame,args['movement'])
            require(raw.get('valid',True) and raw.get('status') not in ['insufficient_coverage','unusable'],'unusable_render')
            f=self.fixed_frame(raw['frame']);require(f=={k:raw['frame'][k] for k in f},'render_intrinsics_changed')
            aid='target_'+digest({'query':self.query['id'],'frame':f})[:16]
            oid=self.add_observation(state,raw['image_path'],'gaussian_render',answer_id=aid)
            previous=state['answers'].get(aid,{})
            state['answers'][aid]={**previous,'id':aid,'kind':'target_view','pose_c2w':f['c2w'],'frame':f,
                'render_observation_id':oid,'accepted':previous.get('accepted',False),'evidence_ids':[oid],
                'pose_origin':'known_virtual_camera_not_source_estimate','coverage':raw.get('coverage')}
            result.update(answer_id=aid,frame=f,new_observation_ids=[oid],render_observation_id=oid,
                coverage=raw.get('coverage'),cache_hit=raw.get('cache_hit'),compute_seconds=raw.get('compute_seconds'),
                origin='continuous_virtual_camera',collision_checked=False)
        else:raise ValueError('not_an_external_tool')
        require(len(state['observations'])<=state['config']['max_observations'],'observation_budget_exceeded')
        result['wall_seconds']=time.perf_counter()-started
        return result

    def contact_sheet(self,state,step):
        # Source plus the latest observations. The complete observation catalog
        # remains in state; only these IDs count as actually seen this call.
        source=state.get('source_observation_id');others=[k for k in state['observations'] if k!=source]
        limit=state['config']['max_images_per_decision']
        ids=([source]+(others[-(limit-1):] if limit>1 else [])) if source else others[-limit:]
        cell=512;header=44;columns=max(1,min(3,len(ids)));rows=max(1,math.ceil(len(ids)/columns))
        canvas=Image.new('RGB',(cell*columns,(cell+header)*rows),(242,244,246));draw=ImageDraw.Draw(canvas)
        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        except OSError:font=ImageFont.load_default()
        if not ids:
            # UI placeholder only: it has no observation ID and provides no
            # scene evidence to the acceptance gate.
            draw.text((20,24),'No source photograph',fill=(70,75,80),font=font)
        for i,ident in enumerate(ids):
            obs=state['observations'][ident];require(sha(obs['image_path'])==obs['image_sha256'],'observation_pixels_changed')
            x=(i%columns)*cell;y=(i//columns)*(cell+header)
            draw.text((x+6,y+2),ident,fill=(15,20,25),font=font)
            draw.text((x+6,y+22),obs['origin'],fill=(15,20,25),font=font)
            with Image.open(obs['image_path']) as im:
                rgb=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(cell-8,cell-8),Image.Resampling.LANCZOS)
                canvas.paste(rgb,(x+(cell-rgb.width)//2,y+header+(cell-rgb.height)//2))
        path=self.root/f'decision_{step:02d}.jpg';canvas.save(path,'JPEG',quality=92)
        return path,ids

def public_state(state,query,control):
    refs=list(state['candidates'].values())[-8:]
    answers=list(state['answers'].values())[-6:]
    best=state['answers'].get(state['best_answer_id'])
    if best and best['id'] not in [a['id'] for a in answers]:answers=[best]+answers
    return {'schema_version':VERSION,'instruction':query['instruction'],'output_type':query['output_type'],
        'required_predicates':query['predicates'],'source_intrinsics':query.get('K'),
        'source_image_present':bool(query.get('source_image')),'control':control,
        'reference_candidates':[{k:(c[k][:400] if k=='caption' and isinstance(c[k],str) else c[k]) for k in ['view_id','score','caption','frame'] if k in c} for c in refs],
        'observations':[{k:o[k] for k in ['id','origin','answer_id','reference_id','region'] if k in o} for o in list(state['observations'].values())[-12:]],
        'geometric_results':list(state['geometry'].values())[-3:],
        'camera_answers':[{k:a[k] for k in ['id','kind','frame','accepted','render_observation_id','coverage','evidence_ids'] if k in a} for a in answers],
        'recent_events':[{k:e[k] for k in ['action','status','category','answer_id','observation_ids'] if k in e} for e in state['events'][-4:]]}
