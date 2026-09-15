"""Compare actual high-mode FP32 + tip inference with untouched 0.3.1 source."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import batch_worm_roi as current
from train_worm_unet import WormUNet
from inspect_roi_dataset import image_array

spec=importlib.util.spec_from_file_location('unchanged_baseline',ROOT.parent/'0908 0.3.1 CUDA/src/batch_worm_roi.py')
baseline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
checkpoint=torch.load(ROOT/'自动圈虫/models/0.1.0/worm.pt',map_location=device)
tips=torch.load(ROOT/'自动圈虫/models/0.1.0/tip.pt',map_location=device)
model=WormUNet(base=checkpoint.get('base',16)).to(device)
model.load_state_dict(checkpoint['model_state']); model.eval()
tip=WormUNet(base=tips.get('base',12),in_channels=tips.get('in_channels',2),out_channels=tips.get('out_channels',2)).to(device)
tip.load_state_dict(tips['model_state']); tip.eval()
kwargs=dict(image_size=checkpoint.get('image_size',512),
            interior_threshold=checkpoint.get('postprocess_interior_threshold'),
            erosion_iterations=checkpoint.get('postprocess_erosion',0),
            min_area_fraction=checkpoint.get('postprocess_min_area',.005),
            min_height_fraction=checkpoint.get('postprocess_min_height',.10),
            max_instances=checkpoint.get('postprocess_max_instances',12),
            tip_model=tip,tip_patch_size=tips.get('patch_size',192),
            tip_probability_threshold=tips.get('probability_threshold',.40),
            tip_replace_fraction=tips.get('replace_fraction',.14),
            normalization_mode=checkpoint.get('normalization_mode','legacy'))
for path in (WORK/'test/test04/0513-10-1.tif', WORK/'test/test02/1229-0-1-40flo.tif'):
    raw=image_array(Image.open(path))
    old=baseline.predict_raw(model,raw,device,**kwargs)
    new=current.predict_raw(model,raw,device,**kwargs)
    assert all(np.array_equal(a,b) for a,b in zip(old,new)), path
    print(path.name,'HIGH_MODE_PIXEL_EXACT',int(new[0].max()),flush=True)
