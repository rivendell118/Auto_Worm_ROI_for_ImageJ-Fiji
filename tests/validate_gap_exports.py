import csv
import sys
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]
new=ROOT/'validation_outputs/test04_gap_final'
old=WORK/'test04/_auto_roi'
with (new/'batch_summary.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==8
unchanged=0
for row in rows:
    stem=Path(row['image']).stem
    with zipfile.ZipFile(new/(stem+'_RoiSet.zip')) as z:
        assert len(z.namelist())==11
        roi_bytes={n:z.read(n) for n in z.namelist()}
    with (new/(stem+'_measurements.csv')).open(encoding='utf-8-sig',newline='') as f:
        measured=list(csv.DictReader(f))
    assert len(measured)==11 and row['worm_count']=='10'
    if int(row['low_clarity_split_count']):
        assert row['qc_status']=='REVIEW_LOW_CLARITY_SPLIT'
    else:
        with zipfile.ZipFile(old/(stem+'_RoiSet.zip')) as z:
            assert roi_bytes=={n:z.read(n) for n in z.namelist()},stem
        with (old/(stem+'_measurements.csv')).open(encoding='utf-8-sig',newline='') as f:
            assert measured==list(csv.DictReader(f)),stem
        unchanged+=1
print('EXPORTS_OK: eight images, 11 ROIs/rows each; unchanged ROI bytes and measurements:',unchanged)
packed=ROOT/'validation_outputs/packaged_low_gap/_auto_roi'
if packed.is_dir():
    for row in rows:
        stem=Path(row['image']).stem
        for suffix in ('_measurements.csv','_split_qc.csv'):
            assert (new/(stem+suffix)).read_bytes()==(packed/(stem+suffix)).read_bytes(),(stem,suffix)
        with zipfile.ZipFile(new/(stem+'_RoiSet.zip')) as a, zipfile.ZipFile(packed/(stem+'_RoiSet.zip')) as b:
            assert {n:a.read(n) for n in a.namelist()}=={n:b.read(n) for n in b.namelist()},stem
    print('PACKAGED_LOW_MATCHES_SOURCE: eight images, ROI bytes, measurements, split reports')
for stem in ('0513-0-1','0513-10-1','0513-10-2'):
    panels=[]
    for root,title in ((old,'Before'),(new,'Low-mode head-gap split')):
        im=Image.open(root/(stem+'_QC.png')).convert('RGB')
        im.thumbnail((720,720))
        panel=Image.new('RGB',(720,750),'black')
        panel.paste(im,(0,30)); ImageDraw.Draw(panel).text((10,8),title,fill='white')
        panels.append(panel)
    canvas=Image.new('RGB',(1440,750))
    canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(720,0))
    canvas.save(new/(stem+'_before_after.jpg'))
