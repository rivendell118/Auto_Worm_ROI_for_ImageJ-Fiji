import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from low_clarity_splitter import split_low_clarity_instances
from evaluate_worm_unet import colorize_labels

for folder in sys.argv[1:] or ['test04']:
    root=ROOT/'validation_outputs'/('gap_'+folder)
    for path in sorted(root.glob('*.npz')):
        data=np.load(path)
        original=np.asarray(Image.open(path.with_name(path.stem+'_coarse.png')))
        result,reports=split_low_clarity_instances(original,data['prob'],data['image'].astype(np.float32)/255,
                                                  max_instances=10)
        assert np.array_equal(result>0,original>0)
        if int(original.max())>=10:
            assert np.array_equal(result,original), path
        if folder=='test04':
            assert int(result.max())==10,path
        Image.fromarray(result).save(path.with_name(path.stem+'_split.png'))
        gray=np.repeat(data['image'][:,:,None],3,axis=2)
        panels=[gray,(gray*.6+colorize_labels(original)*.4).astype(np.uint8),
                (gray*.6+colorize_labels(result)*.4).astype(np.uint8)]
        canvas=Image.fromarray(np.concatenate(panels,axis=1))
        canvas.thumbnail((1536,650))
        canvas.save(path.with_name(path.stem+'_split_compare.jpg'))
        print(folder,path.stem,int(original.max()),int(result.max()),reports,flush=True)
