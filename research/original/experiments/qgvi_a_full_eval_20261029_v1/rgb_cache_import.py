"""Qualified read-only RGB feature reuse; never import a historical pose/decision.

SuperPoint has a producer checksum. Matches have a separately dated checksum
and must replay the producer's correspondence count. NetVLAD is extracted fresh
because its historical output lacks a producer checksum. A changed top-10 list
causes fresh SuperGlue, rather than silently adopting old correspondences.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT_RECEIPT = ROOT / 'experiments/cvpr2027_rgb/logs_selective_v2/QUEUE_RECEIPT.json'

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def require(value, message):
    if not value: raise ValueError(message)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def public_signature(q):
    return (q['image_sha256'],int(q['width']),int(q['height']),json.dumps(q['K'],sort_keys=True))

def qualify_query(query, producer):
    require(query['image_sha256']==producer['image_sha256'], 'RGB source pixels differ from feature producer')
    require(query['width']==producer['width'] and query['height']==producer['height'], 'RGB dimensions differ from feature producer')
    require(np.array_equal(np.asarray(query['K']),np.asarray(producer['K'])), 'RGB intrinsics differ from feature producer')
    require(sha(query['image_path'])==query['image_sha256'], 'RGB source pixels changed')

def validate_local(group, query):
    n=len(group['keypoints'])
    require(group['image_size'][()].tolist()==[query['width'],query['height']], 'Cached feature image size differs')
    require(group['keypoints'].shape==(n,2) and group['descriptors'].shape==(256,n) and group['scores'].shape==(n,), 'Cached SP shapes differ')
    require(all(group[k].dtype==np.float16 for k in ('keypoints','descriptors','scores')), 'Cached SP precision differs')
    require(all(np.isfinite(group[k][()]).all() for k in ('keypoints','descriptors','scores')), 'Nonfinite cached SP')

def copy_feature(source, old_name, destination, new_name):
    with h5py.File(source,'r') as src,h5py.File(destination,'x') as dst:
        src.copy(old_name,dst,name=new_name)
    return Path(destination)

def copy_matches(source, old_name, new_name, references, destination):
    # Released HLOC pair conventions; do not import GPU modules in CPU tests.
    with h5py.File(source,'r') as src,h5py.File(destination,'x') as dst:
        for ref in references:
            found=None
            for reverse in (False,True):
                a,b=(ref,old_name) if reverse else (old_name,ref)
                for sep in ('/','_'):
                    name=sep.join((a.replace('/','-'),b.replace('/','-')))
                    if name in src: found=(name,reverse);break
                if found:break
            require(found is not None,'Frozen SG pair missing')
            name,reverse=found
            a,b=(ref,new_name) if reverse else (new_name,ref)
            target='/'.join((a.replace('/','-'),b.replace('/','-')))
            parent,leaf=target.rsplit('/',1)
            src.copy(name,dst.require_group(parent),name=leaf)
    return Path(destination)

def audit(out, *, audit_receipt=AUDIT_RECEIPT):
    """Freeze qualified reuse by pixels/K and reference identity; CPU only."""
    out=Path(out);plan=read(out/'plan.json');started=time.perf_counter()
    require(plan.get('cache_import_source_sha256')==sha(__file__),'RGB importer code differs from frozen plan')
    receipt=read(audit_receipt);queue_path=Path(receipt['plan']['path'])
    require(sha(queue_path)==receipt['plan']['sha256'],'Historical cache freeze plan changed')
    frozen=read(queue_path);bindings={str(audit_receipt):sha(audit_receipt),str(queue_path):sha(queue_path)}
    rows={};scene_reports={}
    for scene,asset in plan['scenes'].items():
        queries=[q for q in plan['queries'] if q['scene']==scene]
        protocol_path=Path(asset['protocol_path']);protocol=read(protocol_path);directory=protocol_path.parent/'initializer'
        eligible=[]
        try:
            require(frozen['protocol_sha256'].get(str(protocol_path))==sha(protocol_path),'Protocol lacks matching historical binding')
            require(protocol['feature_config']==plan['feature_config'] and protocol['matcher_config']==plan['matcher_config'],'Feature/matcher recipe differs')
            require(Path(protocol['sfm_source'])/'sfm'==Path(asset['sfm_dir']),'Map coordinate source differs')
            producer_queries=[r['query'] for r in protocol['rows']]
            lookup={}
            for p in producer_queries:lookup.setdefault(public_signature(p),[]).append(p)
            paths=[directory/n for n in ('summary.json','predictions.json','query_superpoint.h5','matches.h5','retrieval_top10.txt')]
            for path in paths:
                expected=frozen['test_cache_sha256'].get(str(path))
                require(expected is not None and sha(path)==expected,'Frozen historical cache changed: '+str(path))
                bindings[str(path)]=expected
            summary=read(directory/'summary.json')
            require(summary['status']=='complete' and summary['protocol_sha256']==sha(protocol_path),'Producer completion binding differs')
            require(summary['no_query_ground_truth_access'] is True,'Producer not public-only')
            require(summary['query_features_sha256']==sha(directory/'query_superpoint.h5'),'SP producer checksum differs')
            require(summary['predictions_sha256']==sha(directory/'predictions.json'),'Producer retrieval/count receipt changed')
            # Only names/counts are projected. Historical pose, acceptance and GT
            # are deliberately never retained or consumed by inference.
            evidence={p['query_id']:{'references':p['retrieved_training_names'],'raw_match_count':p['num_matches']} for p in read(directory/'predictions.json')}
            pairs={}
            for line in (directory/'retrieval_top10.txt').read_text().splitlines():
                a,b=line.split();pairs.setdefault(a,[]).append(b)
            with h5py.File(directory/'query_superpoint.h5','r') as features:
                for query in queries:
                    try:
                        candidates=lookup.get(public_signature(query),[])
                        require(bool(candidates),'No exact pixel/K producer query')
                        old=sorted(candidates,key=lambda r:r['id'])[0];qualify_query(query,old)
                        name=old['id']+'.png';validate_local(features[name],query)
                        e=evidence[old['id']]
                        require(e['references']==pairs.get(name,[]),'Producer retrieval and pair file differ')
                        require(set(e['references'])<=set(asset['reference_names']),'Reference membership differs')
                        require(len(e['references'])==plan['top_k'] and len(set(e['references']))==plan['top_k'],'Historical top-k differs')
                        rows[query['id']]={'eligible':True,'query_id':query['id'],'scene':scene,'source_name':name,
                            'query_superpoint':str(directory/'query_superpoint.h5'),'matches':str(directory/'matches.h5'),
                            'reference_names':e['references'],'raw_match_observation_count':e['raw_match_count'],
                            'query_image_sha256':query['image_sha256'],'producer_query':{k:old[k] for k in ('image_sha256','K','width','height')}}
                        eligible.append(query['id'])
                    except (ValueError,KeyError,OSError) as exc:
                        rows[query['id']]={'eligible':False,'reason':str(exc),'scene':scene}
            scene_reports[scene]={'qualified_sources':len(eligible),'total':len(queries),'historical_batch_timings':summary.get('timings',{})}
        except (ValueError,KeyError,OSError) as exc:
            for query in queries:rows[query['id']]={'eligible':False,'reason':str(exc),'scene':scene}
            scene_reports[scene]={'qualified_sources':0,'total':len(queries),'reason':str(exc)}
    result={'status':'audited','plan_sha256':sha(out/'plan.json'),'importer_sha256':sha(__file__),
        'bindings_sha256':bindings,'queries':rows,'scenes':scene_reports,
        'qualified_source_count':sum(r['eligible'] for r in rows.values()),'query_count':len(rows),
        'audit_seconds':time.perf_counter()-started,'historical_pose_or_decision_reused':False,
        'fresh_stages':['NetVLAD','reference ranking','correspondence assembly','PnP seed20261025'],
        'conditional_reuse':'SP bytes producer-bound; SG bytes independently frozen, reuse only exact fresh top10 and replayed correspondence count',
        'fallback':'Any unqualified source uses complete fresh feature/match preparation; changed ranking uses fresh SG',
        'timing_limitation':'Historical reused-source per-query extraction/matching latency unavailable; current preparation time is not fresh end-to-end latency.'}
    target=out/'RGB_CACHE_AUDIT.json'
    if target.exists():
        prior=read(target)
        require({k:v for k,v in prior.items() if k!='audit_seconds'}=={k:v for k,v in result.items() if k!='audit_seconds'},'Frozen RGB cache audit differs')
        return prior
    write(target,result);return result

def load_verified(out):
    path=Path(out)/'RGB_CACHE_AUDIT.json'
    if not path.exists():return None
    data=read(path)
    require(data['plan_sha256']==sha(Path(out)/'plan.json') and data['importer_sha256']==sha(__file__),'RGB audit binding changed')
    for path,checksum in data['bindings_sha256'].items():require(sha(path)==checksum,'Qualified RGB cache bytes changed')
    return data
