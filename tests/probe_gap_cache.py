import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage, signal
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

root = ROOT / 'validation_outputs/gap_test04'
for stem, label in [('0513-0-1',8),('0513-10-1',5),('0513-10-1',6),('0513-10-2',8),('0513-10-2',9)]:
    data=np.load(root/(stem+'.npz'))
    p=data['prob']; image=data['image']
    mask=np.asarray(Image.open(root/(stem+'_coarse.png')))==label
    yy,xx=np.nonzero(mask); height=np.ptp(yy)+1
    print(stem,label, 'bbox',xx.min(),xx.max(), yy.min(),yy.max())
    for t in (.55,.7,.85,.9,.95):
        for e in (1,2,3,4):
            seeds,n=ndimage.label(ndimage.binary_erosion(mask & (p[1]>=t),iterations=e))
            pieces=[]
            for l in range(1,n+1):
                y,x=np.nonzero(seeds==l)
                if len(y)>mask.sum()*.06 and np.ptp(y)>height*.25:
                    pieces.append((len(y),int(y.min()),int(y.max()),round(float(x.mean()),1)))
            if len(pieces)>1: print(t,e,pieces)
    for frac in (.08,.15,.25,.4,.6,.8):
        y=int(yy.min()+frac*height); xs=np.flatnonzero(mask[y]);
        if len(xs):
            lo,hi=xs.min(),xs.max()
            v=ndimage.gaussian_filter1d(p[1,y],1.)
            peaks,_=signal.find_peaks(v[lo:hi+1],prominence=.15,distance=6)
            print('row',y,'span',lo,hi,'peaks',[(lo+x,round(float(v[lo+x]),2)) for x in peaks])
