import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from low_clarity_splitter import _core_evidence, _grow_cores

def trace(frame,event,arg):
    if frame.f_code.co_name=='_core_evidence' and event=='return':
        print({k:v for k,v in frame.f_locals.items() if k in ('overlap','support','head_gap','head_scores')})
    return trace

for stem,label,threshold in [('0513-10-1',6,.85),('0513-10-2',8,.95)]:
    root=ROOT/'validation_outputs/gap_test04'
    data=np.load(root/(stem+'.npz')); p=data['prob']; image=data['image']/255.
    mask=np.asarray(Image.open(root/(stem+'_coarse.png')))==label
    yy,xx=np.nonzero(mask); height=np.ptp(yy)+1
    components,n=ndimage.label(ndimage.binary_erosion(mask & (p[1]>=threshold),iterations=2))
    cores=[components==i for i in range(1,n+1) if (components==i).sum()>.08*mask.sum()]
    print(stem,label,[(c.sum(),np.ptp(np.nonzero(c)[0])) for c in cores])
    sys.settrace(trace)
    result=_core_evidence(*cores,p,image,height)
    sys.settrace(None)
    if result is not None:
        partition=_grow_cores(mask,result[:2],p)
        print('unassigned',int((mask & (partition==0)).sum()),'areas',[(partition==i).sum() for i in (1,2)],
              'components',[ndimage.label(partition==i)[1] for i in (1,2)])
