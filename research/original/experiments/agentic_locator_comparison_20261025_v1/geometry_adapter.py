#!/usr/bin/env python3
"""Shared, genuine SP/SG+PnP tool. Never substitutes a database camera pose.

Preparation privately uses source camera metadata ONLY to exclude self/duplicate
references. Inference consumes query pixels/K and the frozen reference map.
ROI means filtering full-image correspondences, not rerunning cropped features.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PILOT=ROOT/'experiments/agentic_locator_pilot_20261025_v1/pilot'
DEFAULT_OUT=HERE/'geometry'
RGB=ROOT/'experiments/cvpr2027_rgb'
UPSTREAM=RGB/'third_party/LoGS'
CONFIDENCE=RGB/'logs_selective_v2/confidence.py'
LOCKS=(ROOT/'experiments/cityloc_qgvi_v2/resources/gpu0.lock',Path('/home/xiongti1/slam/AlloMap/artifacts/cse_scaling_campaign_gpu.lock'))
SEED=20261025
FEATURE_CONFIG={'output':'feats-superpoint-n4096-r1024','model':{'name':'superpoint','nms_radius':3,'max_keypoints':4096},'preprocessing':{'globs':['*.png'],'grayscale':True,'resize_max':1024}}
MATCHER_CONFIG={'output':'matches-superglue','model':{'name':'superglue','weights':'outdoor','sinkhorn_iterations':5}}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)
def frozen(p,x):
    if Path(p).exists():
        if read(p)!=x:raise ValueError(f'Frozen binding changed: {p}')
    else:write(p,x)
def confidence_module():
    spec=importlib.util.spec_from_file_location('pilot_geometry_confidence',CONFIDENCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def exclusion_names(source,records, *, pose_atol=1e-6):
    """Private preparation only; do not expose source pose or exclusion reasons."""
    target=source['training_template_id'];found=[r for r in records if r['id']==target]
    if len(found)!=1:raise ValueError('Query training identity not unique in reference map')
    pose=np.asarray(source['c2w'],float)
    if not np.allclose(pose,found[0]['c2w'],rtol=0,atol=pose_atol):raise ValueError('Source/map coordinate frames differ')
    if len({r['name'] for r in records})!=len(records):raise ValueError('Duplicate reference names')
    excluded={r['name'] for r in records if r['id']==target or (found[0].get('sha256') is not None and r.get('sha256')==found[0]['sha256']) or np.allclose(np.asarray(r['c2w']),pose,rtol=0,atol=pose_atol)}
    return sorted(excluded)


def prepare(pilot=PILOT,out=DEFAULT_OUT,bundle_ids=None):
    pilot=Path(pilot);out=Path(out);bundles=read(pilot/'bundles.json')['bundles'];selected=[b for b in bundles if not b.get('reserve') and (bundle_ids is None or b['bundle_id'] in bundle_ids)]
    if not selected:raise ValueError('No selected bundles')
    queries=[];scenes={}
    for b in selected:
        scene=b['scene'];source=b['source']
        if scene not in ('block_a','sf'):raise ValueError('Pilot scene unsupported')
        protocol=RGB/('logs/block_a_full_native_v1/protocol.json' if scene=='block_a' else 'other_baselines_nine_completion_20261004_v1/sf/logs/protocol.json')
        p=read(protocol);sfmroot=Path(p['sfm_source']);train=read(sfmroot/'protocol.json');summary=read(sfmroot/'summary.json')
        if summary['status']!='complete' or train['feature_config']!=FEATURE_CONFIG:raise ValueError('Unqualified reference feature recipe')
        refs=sfmroot/'feats-superpoint-n4096-r1024.h5';glob=protocol.parent/'initializer/reference_netvlad.h5'
        if not refs.is_file() or not glob.is_file():raise FileNotFoundError('Reference feature cache missing')
        excluded=exclusion_names(source,train['records'])
        if sha(source['image_path'])!=source['image_sha256']:raise ValueError('Selected query pixels changed')
        q={k:source[k] for k in ('image_path','image_sha256','K','width','height')};q.update(id=b['bundle_id'],scene=scene,name=b['bundle_id']+'.png',excluded_reference_names=excluded)
        queries.append(q)
        if scene not in scenes:
            # Full reference feature bytes are verified once before first inference.
            scenes[scene]={'protocol_path':str(protocol),'protocol_sha256':sha(protocol),'training_protocol':str(sfmroot/'protocol.json'),'training_protocol_sha256':sha(sfmroot/'protocol.json'),'reference_features':str(refs),'reference_features_sha256':summary['features_sha256'],'reference_global':str(glob),'reference_global_sha256':sha(glob),'sfm_dir':str(sfmroot/'sfm'),'sfm_files':summary['artifacts_sha256'],'reference_names':[r['name'] for r in train['records']],'torch_cache':p['torch_cache'],'seed':p['seed'],'meters_per_model_unit':p['meters_per_model_unit']}
    plan={'schema_version':1,'recipe':'SP4096-r1024-fp16_SG-outdoor-it5_NetVLAD10_PnP12px_seed20261025_no_fallback','source_sha256':sha(__file__),'confidence_sha256':sha(CONFIDENCE),'queries':queries,'scenes':scenes,'feature_config':FEATURE_CONFIG,'matcher_config':MATCHER_CONFIG,'top_k':10,'roi_convention':'normalized keep-box x0,y0,x1,y1; filter full-image SP/SG correspondences; unchangedK','disclosure':'Original training-source simulator RGB. Self IDs, exact-image duplicates and same-camera references excluded from retrieval. Existing SfM may include source observations; not reconstruction-held-out.'}
    frozen(out/'plan.json',plan);return plan


def filter_region(corr,query,region):
    if region is None:return corr
    b=np.asarray(region,float)
    if b.shape!=(4,) or not np.isfinite(b).all() or not (0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):raise ValueError('region must be normalized[x0,y0,x1,y1] with positive area')
    # Saved keypoints use COLMAP+0.5; use corresponding full-image pixel centers.
    xy=np.asarray(corr['keypoints_colmap']);lo=b[:2]*[query['width'],query['height']];hi=b[2:]*[query['width'],query['height']]
    keep=np.all((xy>=lo)&(xy<=hi),axis=1);res=dict(corr)
    for k in ('keypoints_colmap','query_indices','point3d_ids','points3d'):res[k]=np.asarray(corr[k])[keep]
    res['reference_support']=[s for s,k in zip(corr['reference_support'],keep) if k]
    res['raw_match_observation_count']=sum(len(s) for s in res['reference_support'])
    return res


def solve(query,corr,region=None):
    import pycolmap
    start=time.perf_counter();corr=filter_region(corr,query,region);pose=None;ret=None
    if len(np.unique(corr['query_indices']))>=4 and len(np.unique(corr['point3d_ids']))>=4:
        k=np.asarray(query['K']);camera=pycolmap.Camera(model='PINHOLE',width=query['width'],height=query['height'],params=[k[0,0],k[1,1],k[0,2],k[1,2]])
        ret=pycolmap.estimate_and_refine_absolute_pose(np.asarray(corr['keypoints_colmap']),np.asarray(corr['points3d']),camera,estimation_options={'ransac':{'max_error':12.,'random_seed':SEED}},refinement_options={})
        if ret is not None:
            w2c=np.eye(4);w2c[:3]=ret['cam_from_world'].matrix();pose=np.linalg.inv(w2c).tolist()
    pred={'query_id':query['id'],'raw_c2w':pose,'status':'pnp' if pose is not None else 'pnp_failed'}
    evidence=confidence_module().extract_confidence(pred,query,corr)['features']
    return {'status':'estimated' if pose is not None else 'pnp_failed','pose_c2w':pose,'coordinate_units':'model','evidence':evidence,'compute_seconds':time.perf_counter()-start,'pose_origin':'SuperPoint+SuperGlue+triangulatedSfM+absolutePnP','fallback_used':False,'roi':region,'roi_semantics':'filter full-image correspondences; K unchanged','acceptance_calibrated_for_this_pilot':False}


def save_corr(path,corr):
    arrays={k:np.asarray(corr[k]) for k in ('keypoints_colmap','query_indices','point3d_ids','points3d')}
    meta={k:corr[k] for k in ('reference_support','reference_names','raw_match_observation_count')};path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays,metadata=np.array(json.dumps(meta)))
    path.with_suffix('.tmp').replace(path)
def load_corr(path):
    with np.load(path,allow_pickle=False) as f:return {**json.loads(str(f['metadata'])),**{k:f[k].copy() for k in ('keypoints_colmap','query_indices','point3d_ids','points3d')}}


@contextmanager
def gpu_lock(paths=LOCKS):
    streams=[]
    try:
        for p in paths:
            p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');streams.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        for f in reversed(streams):fcntl.flock(f,fcntl.LOCK_UN);f.close()


def validate_plan(out):
    p=read(Path(out)/'plan.json')
    if p['source_sha256']!=sha(__file__) or p['confidence_sha256']!=sha(CONFIDENCE):raise ValueError('Qualified geometry code changed')
    for q in p['queries']:
        if sha(q['image_path'])!=q['image_sha256']:raise ValueError('Query pixel binding changed')
    return p


def run(out=DEFAULT_OUT):
    out=Path(out);p=validate_plan(out)
    if (out/'COMPLETED.json').is_file():
        done=read(out/'COMPLETED.json')
        if done['plan_sha256']!=sha(out/'plan.json'):raise ValueError('Completion plan mismatch')
        for path,h in done['outputs_sha256'].items():
            if sha(path)!=h:raise ValueError('Completed cache changed')
        return done
    # Qualification hashes are checked before GPU admission; no map is rebuilt.
    for a in p['scenes'].values():
        checks={a['protocol_path']:a['protocol_sha256'],a['training_protocol']:a['training_protocol_sha256'],a['reference_features']:a['reference_features_sha256'],a['reference_global']:a['reference_global_sha256'],**a['sfm_files']}
        for path,h in checks.items():
            if sha(path)!=h:raise ValueError(f'Reference binding changed:{path}')
    with gpu_lock():
        import torch
        import h5py
        import pycolmap
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU inference')
        sys.path.insert(0,str(UPSTREAM));from hloc import extract_features,match_features
        torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
        torch.hub.download_url_to_file=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('Unprepared model download prohibited'))
        started=time.perf_counter();outputs=[];completed=0;conf=confidence_module();all_timings={}
        for scene,a in p['scenes'].items():
            scene_queries=[q for q in p['queries'] if q['scene']==scene];work=out/scene;inputs=work/'inputs';inputs.mkdir(parents=True,exist_ok=True);torch.hub.set_dir(a['torch_cache']);names=[]
            for q in scene_queries:
                dest=inputs/q['name']
                if dest.exists() and sha(dest)!=q['image_sha256']:raise ValueError('Input staging changed')
                if not dest.exists():shutil.copyfile(q['image_path'],dest)
                names.append(q['name'])
            def stage(name,fn):
                write(out/'status.json',{'status':'running','stage':name,'scene':scene,'completed':completed,'total':len(p['queries']),'updated_unix':time.time()});torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();all_timings[scene+'/'+name]=time.perf_counter()-t;return v
            global_path=stage('query_netvlad',lambda:extract_features.main(extract_features.confs['netvlad'],inputs,work,as_half=True,image_list=names,feature_path=work/'query_netvlad.h5'))
            features=stage('query_superpoint',lambda:extract_features.main(FEATURE_CONFIG,inputs,work,as_half=True,image_list=names,feature_path=work/'query_superpoint.h5'))
            retrieved={};ref_names=sorted(a['reference_names'])
            with h5py.File(a['reference_global'],'r') as rf,h5py.File(global_path,'r') as qf:
                matrix=np.stack([np.asarray(rf[n]['global_descriptor'],np.float32).reshape(-1) for n in ref_names])
                for q in scene_queries:
                    desc=np.asarray(qf[q['name']]['global_descriptor'],np.float32).reshape(-1);scores=matrix@desc;excluded=set(q['excluded_reference_names']);ranks=sorted(range(len(ref_names)),key=lambda i:(-float(scores[i]),ref_names[i]));retrieved[q['name']]=[ref_names[i] for i in ranks if ref_names[i] not in excluded][:p['top_k']]
                    if set(retrieved[q['name']])&excluded:raise AssertionError('Self reference survived')
            pairs=work/'pairs.txt';pairs.write_text(''.join(f'{q} {r}\n' for q,refs in retrieved.items() for r in refs))
            matches=stage('query_superglue',lambda:match_features.main(MATCHER_CONFIG,pairs,features,matches=work/'matches.h5',features_ref=Path(a['reference_features'])))
            reconstruction=stage('load_sparse_map',lambda:pycolmap.Reconstruction(a['sfm_dir']));images,points3d=reconstruction.images,reconstruction.points3D;proxy=SimpleNamespace(images=images,points3D=points3d);name_to_id={im.name:i for i,im in images.items()}
            for q in scene_queries:
                corr=conf.assemble_correspondences(proxy,q['name'],retrieved[q['name']],features,matches,name_to_id=name_to_id);cp=out/'correspondences'/(q['id']+'.npz');save_corr(cp,corr);result=solve(q,corr);result.update(hypothesis_id=digest({'query':q['image_sha256'],'plan':sha(out/'plan.json'),'roi':None})[:24],query_image_sha256=q['image_sha256'],plan_sha256=sha(out/'plan.json'),correspondences_sha256=sha(cp),reference_count=len(retrieved[q['name']]),self_reference_excluded=True)
                rp=out/'results'/(q['id']+'.json');write(rp,result);outputs.extend([cp,rp]);completed+=1;write(out/'status.json',{'status':'running','stage':'pose','scene':scene,'completed':completed,'total':len(p['queries']),'updated_unix':time.time()});print(json.dumps({'event':'geometry_pose','completed':completed,'total':len(p['queries']),'scene':scene,'pnp_success':result['status']=='estimated','unique_inliers':result['evidence']['unique_inlier_queries']}),flush=True)
            outputs.extend([global_path,features,matches,pairs]);del proxy,images,points3d,reconstruction;torch.cuda.empty_cache()
        done={'status':'complete','query_count':completed,'plan_sha256':sha(out/'plan.json'),'outputs_sha256':{str(f):sha(f) for f in outputs},'total_seconds':time.perf_counter()-started,'stage_seconds':all_timings,'reference_database_rebuilt':False,'source_gt_used_for_inference':False,'source_gt_used_privately_for_self_exclusion':True,'model_versions':{'torch':torch.__version__,'pycolmap':pycolmap.__version__}}
        write(out/'COMPLETED.json',done);write(out/'status.json',{'status':'complete','stage':'complete','completed':completed,'total':len(p['queries']),'updated_unix':time.time()});return done


def public_result(record, *, elapsed,cache_hit):
    keys=('hypothesis_id','status','pose_c2w','coordinate_units','evidence','compute_seconds','pose_origin','fallback_used','roi','roi_semantics','acceptance_calibrated_for_this_pilot','reference_count','self_reference_excluded')
    return {**{k:record[k] for k in keys if k in record},'elapsed_seconds':elapsed,'cache_hit':cache_hit,'shared_precomputed_matching':True}


def estimate(bundle_id,region=None,out=DEFAULT_OUT):
    start=time.perf_counter();out=Path(out);p=validate_plan(out);qs=[q for q in p['queries'] if q['id']==bundle_id]
    if len(qs)!=1:raise ValueError('Unknown host-mapped query')
    q=qs[0];bp=out/'results'/(bundle_id+'.json');cp=out/'correspondences'/(bundle_id+'.npz');done=read(out/'COMPLETED.json')
    if done['plan_sha256']!=sha(out/'plan.json') or any(done['outputs_sha256'].get(str(f))!=sha(f) for f in (bp,cp)):raise ValueError('Completed geometry receipt changed')
    base=read(bp)
    if base['plan_sha256']!=sha(out/'plan.json') or base['correspondences_sha256']!=sha(cp):raise ValueError('Geometry cache binding changed')
    if region is None:return public_result(base,elapsed=time.perf_counter()-start,cache_hit=True)
    corr=load_corr(cp);filter_region(corr,q,region)  # Validate even if an ROI cache exists.
    key=digest({'plan':base['plan_sha256'],'query':q['image_sha256'],'region':list(map(float,region))});rp=out/'roi_results'/(key+'.json');cached=rp.exists()
    if cached:result=read(rp)
    else:
        result=solve(q,corr,list(map(float,region)));result.update(hypothesis_id=key[:24],reference_count=base['reference_count'],self_reference_excluded=True);write(rp,result)
    return public_result(result,elapsed=time.perf_counter()-start,cache_hit=cached)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run','estimate'));p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--pilot',type=Path,default=PILOT);p.add_argument('--bundle-id');p.add_argument('--region',type=json.loads);a=p.parse_args()
    result=prepare(a.pilot,a.out) if a.command=='prepare' else run(a.out) if a.command=='run' else estimate(a.bundle_id,a.region,a.out)
    print(json.dumps(result if a.command=='estimate' else {'status':result.get('status','prepared'),'query_count':result.get('query_count',len(result.get('queries',[])))},allow_nan=False))
if __name__=='__main__':main()
