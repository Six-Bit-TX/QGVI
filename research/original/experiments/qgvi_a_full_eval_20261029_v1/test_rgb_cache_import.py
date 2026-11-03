"""CPU-only checks of provenance, aliases, and fresh-PnP/cache boundaries."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rgb_import_test',HERE/'rgb_cache_import.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('rgb_runtime_test',HERE/'source_geometry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

def fixture(tmp_path):
    out=tmp_path/'geometry';out.mkdir();old=tmp_path/'old';old.mkdir()
    image=tmp_path/'source.png';image.write_bytes(b'public pixels')
    q={'id':'opaque','name':'opaque.png','scene':'sf','image_path':str(image),'image_sha256':m.sha(image),'K':[[10,0,8],[0,10,6],[0,0,1]],'width':16,'height':12}
    producer={**q,'id':'old_query'}
    with h5py.File(old/'query_superpoint.h5','w') as f:
        d=f.create_group('old_query.png');d['keypoints']=np.array([[1,2],[3,4]],np.float16)
        d['descriptors']=np.zeros((256,2),np.float16);d['scores']=np.ones(2,np.float16);d['image_size']=[16,12]
        d['keypoints'].attrs['uncertainty']=1.
    with h5py.File(old/'matches.h5','w') as f:
        d=f.create_group('old_query.png/ref.png');d['matches0']=np.array([0,1],np.int16);d['matching_scores0']=np.array([.8,.9],np.float16)
    protocol={'rows':[{'query':producer}],'sfm_source':str(tmp_path/'sfm'),'feature_config':{'sp':1},'matcher_config':{'sg':1}}
    protocol_path=tmp_path/'protocol.json';m.write(protocol_path,protocol)
    predictions=[{'query_id':'old_query','retrieved_training_names':['ref.png'],'num_matches':2,'accepted':True,'c2w':'forbidden'}]
    m.write(old/'predictions.json',predictions);(old/'retrieval_top10.txt').write_text('old_query.png ref.png\n')
    m.write(old/'summary.json',{'status':'complete','protocol_sha256':m.sha(protocol_path),'no_query_ground_truth_access':True,'query_features_sha256':m.sha(old/'query_superpoint.h5'),'predictions_sha256':m.sha(old/'predictions.json')})
    # Audit expects the historical initializer beside protocol.
    old.rename(tmp_path/'initializer');old=tmp_path/'initializer'
    queue=tmp_path/'QUEUE.json';m.write(queue,{'protocol_sha256':{str(protocol_path):m.sha(protocol_path)},'test_cache_sha256':{str(p):m.sha(p) for p in old.iterdir()}})
    receipt=tmp_path/'RECEIPT.json';m.write(receipt,{'plan':{'path':str(queue),'sha256':m.sha(queue)}})
    plan={'cache_import_source_sha256':m.sha(HERE/'rgb_cache_import.py'),'feature_config':protocol['feature_config'],'matcher_config':protocol['matcher_config'],'top_k':1,'queries':[q],'scenes':{'sf':{'protocol_path':str(protocol_path),'sfm_dir':str(tmp_path/'sfm/sfm'),'reference_names':['ref.png']}}}
    m.write(out/'plan.json',plan)
    return out,old,receipt,q

def test_audit_projects_only_public_feature_evidence_and_replays(tmp_path):
    out,old,receipt,q=fixture(tmp_path);a=m.audit(out,audit_receipt=receipt)
    assert a['qualified_source_count']==1 and a['historical_pose_or_decision_reused'] is False
    row=a['queries']['opaque'];assert row['source_name']=='old_query.png'
    assert 'accepted' not in json.dumps(row) and 'forbidden' not in json.dumps(row) and 'c2w' not in json.dumps(row)
    assert m.audit(out,audit_receipt=receipt)==a
    assert m.load_verified(out)==a

@pytest.mark.parametrize('change',['pixels','K','membership','checksum','recipe'])
def test_failed_provenance_is_explicit_fresh_fallback(tmp_path,change):
    out,old,receipt,q=fixture(tmp_path);plan=m.read(out/'plan.json')
    if change=='pixels':Path(q['image_path']).write_bytes(b'changed')
    elif change=='K':plan['queries'][0]['K'][0][0]=99
    elif change=='membership':plan['scenes']['sf']['reference_names']=['other.png']
    elif change=='checksum':(old/'matches.h5').write_bytes(b'changed')
    elif change=='recipe':plan['matcher_config']={'wrong':True}
    m.write(out/'plan.json',plan);a=m.audit(out,audit_receipt=receipt)
    assert a['qualified_source_count']==0 and a['queries']['opaque']['eligible'] is False
    assert a['queries']['opaque']['reason']

def test_import_rejects_changed_cache_after_audit(tmp_path):
    out,old,receipt,q=fixture(tmp_path);m.audit(out,audit_receipt=receipt)
    (old/'matches.h5').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='cache bytes changed'):m.load_verified(out)

def test_group_alias_preserves_feature_order_precision_and_attributes(tmp_path):
    out,old,receipt,q=fixture(tmp_path)
    target=tmp_path/'copied.h5';m.copy_feature(old/'query_superpoint.h5','old_query.png',target,'opaque.png')
    with h5py.File(target) as f:
        assert list(f)==['opaque.png'];assert f['opaque.png/keypoints'].dtype==np.float16
        assert f['opaque.png/keypoints'].attrs['uncertainty']==1.
    target=tmp_path/'copied_matches.h5';m.copy_matches(old/'matches.h5','old_query.png','opaque.png',['ref.png'],target)
    with h5py.File(target) as f:assert f['opaque.png/ref.png/matches0'][()].tolist()==[0,1]

def test_fresh_top10_change_forces_fresh_matching(tmp_path):
    runtime=g.Runtime.__new__(g.Runtime);runtime.reuse_rows={'opaque':{'reference_names':['old.png']}}
    runtime.geometry=SimpleNamespace(MATCHER_CONFIG={'recipe':'exact'})
    calls=[];runtime.match_features=SimpleNamespace(main=lambda *a,**kw:calls.append((a,kw)) or 'fresh')
    pairs=tmp_path/'pairs.txt';pairs.write_text('opaque.png new.png\n')
    assert runtime.match(pairs,tmp_path/'features.h5',tmp_path,{'reference_features':'refs.h5'})=='fresh'
    assert len(calls)==1

def test_cached_match_count_mismatch_blocks_new_pose(tmp_path):
    runtime=g.Runtime.__new__(g.Runtime);runtime.match_query={'matches.h5':{'raw_match_observation_count':42}}
    runtime.confidence=SimpleNamespace(assemble_correspondences=lambda *a,**kw:{'raw_match_observation_count':41})
    with pytest.raises(ValueError,match='correspondence count differs'):
        runtime.assemble({'name':'opaque.png'},['ref.png'],'features.h5','matches.h5',{'proxy':object(),'name_to_id':{}})
