# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

SRC = os.path.join(SPECPATH, "src")
datas, binaries, hiddenimports = [], [], []
for pkg in ("torch", "cv2", "scipy"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += [
    "batch_worm_roi", "train_worm_unet", "evaluate_worm_unet",
    "evaluate_tip_refiner", "inspect_roi_dataset",
    "prepare_segmentation_dataset", "worm_shape_refiner",
    "low_clarity_splitter", "manual_head_annotation",
    "worm_segment_selector",
]

a = Analysis(
    [os.path.join(SRC, "worm_roi_gui.py")],
    pathex=[SRC], binaries=binaries, datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[os.path.join(SRC, "rthook_torch_dll.py")],
    excludes=["matplotlib", "tkinter.test"], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, exclude_binaries=True, name="AutoWormGUI",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="AutoWormImageJ")
