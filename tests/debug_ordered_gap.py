import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from low_clarity_splitter import split_low_clarity_instances
def trace(frame,event,arg):
    if frame.f_code.co_name=='_ordered_partition' and event=='return':
        v=frame.f_locals
        print('RETURN',frame.f_lineno,'none',arg is None)
        if 'curves' in v: print('mindiff',np.diff(v['curves'],axis=0).min())
        if 'result' in v:
            print('label',v.get('i'),'components',v.get('n'),'sizes',np.bincount(v['components'].ravel())[1:])
            path=ROOT/'validation_outputs/gap_test04/debug_partition.png'
            Image.fromarray(v['result']).save(path)
        if 'core' in v and 'region' in v: print('retained', (v['core']&v['region']).sum()/v['core'].sum())
    return trace
for stem in ('0513-0-1','0513-10-1','0513-10-2'):
    root=ROOT/'validation_outputs/gap_test04'; data=np.load(root/(stem+'.npz'))
    original=np.asarray(Image.open(root/(stem+'_coarse.png')))
    print(stem)
    sys.settrace(trace)
    split_low_clarity_instances(original,data['prob'],data['image']/255,max_instances=10)
    sys.settrace(None)
