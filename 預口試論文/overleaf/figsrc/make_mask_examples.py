"""Offline inference on six held-out frames; never runs Isaac or trains weights."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'evaluation'))
sys.path.insert(0, str(HERE / 'ch03'))
from method_diagrams import FONT, plt
import house_seg as seg
import torch

def main():
    torch.set_num_threads(2)
    data = REPO / 'evaluation/house_seg_data/v2'
    ckpt = REPO / 'evaluation/house_seg_ckpt/run2/best.pth'
    meta = json.loads((ckpt.parent/'meta.json').read_text())
    rows = [json.loads(s) for s in (data/'index.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['run'] in meta['test_runs']]
    model = seg.load_model(str(ckpt), 'cpu')
    names = ['門把', '門扇', '圓柱', '抽屜把手', '筆狀代理物', '抽屜面板']
    colors = np.array([[230,80,70],[70,160,235],[250,180,45],
                       [120,90,215],[55,195,135],[225,100,180]], dtype=float)
    fig, axes = plt.subplots(3, 4, figsize=(10, 6.7))
    evidence=[]
    for i,c in enumerate(seg.CLASSES[1:]):
        eligible = sorted([r for r in rows if r['counts'].get(c,0)>=seg.MIN_AREA],
                          key=lambda r:(r['counts'][c],r['stem']))
        # Prefer complete objects in context rather than near-field clipped surfaces.
        # GT only selects the frame; overlay is exclusively model prediction.
        desired=[.006,.20,.018,.008,.006,.12][i]
        choices=[]
        for candidate in eligible:
            gt=np.asarray(Image.open(data/(candidate['stem']+'_label.png')))==i+1
            yy,xx=np.where(gt)
            clipped=min(xx.min(),yy.min(),gt.shape[1]-1-xx.max(),gt.shape[0]-1-yy.max())<12
            cost=8*int(clipped)+abs(np.log(max(gt.mean(),1e-8)/desired))
            choices.append((cost,candidate['stem'],candidate))
        for _,_,r in sorted(choices,key=lambda z:(z[0],z[1]))[:30]:
            rgb=np.array(Image.open(data/(r['stem']+'_rgb.png')).convert('RGB'),copy=True)
            lab=np.asarray(Image.open(data/(r['stem']+'_label.png')))
            pred=seg.predict(model,rgb,'cpu')
            mask=np.zeros(lab.shape,dtype=bool)
            for ins in pred.get(c,[]): mask |= ins['mask']
            gt=lab==i+1
            example_iou=float((gt&mask).sum())/max(int((gt|mask).sum()),1)
            if mask.sum()>0 and example_iou>.30:break
        else:raise ValueError(('no legible prediction among candidate examples',c))
        overlay=rgb.copy();overlay[mask]=(.48*rgb[mask]+.52*colors[i]).astype('uint8')
        j=(i//2,2*(i%2))
        axes[j].imshow(rgb);axes[j].set_title(names[i]+'：RGB',fontproperties=FONT,fontsize=11)
        axes[j[0],j[1]+1].imshow(overlay)
        axes[j[0],j[1]+1].set_title('模型預測遮罩',fontproperties=FONT,fontsize=11)
        g=lab==(i+1);union=int((g|mask).sum())
        evidence.append(dict(class_name=c,frame=r['stem'],run=r['run'],
            predicted_pixels=int(mask.sum()),example_iou=float((g&mask).sum())/max(union,1),
            rgb_sha256=hashlib.sha256((data/(r['stem']+'_rgb.png')).read_bytes()).hexdigest()))
        assert mask.sum()>0,(c,r['stem'])
        print(c,r['stem'],evidence[-1]['example_iou'],flush=True)
    for ax in axes.flat:ax.axis('off')
    fig.subplots_adjust(left=.005,right=.995,bottom=.015,top=.95,wspace=.025,hspace=.17)
    out=HERE/'ch02';out.mkdir(exist_ok=True)
    for ext in ('pdf','png'):fig.savefig(out/('mask_rcnn_class_examples.'+ext),dpi=220)
    plt.close(fig)
    (out/'mask_examples_evidence.json').write_text(json.dumps(dict(
        model=str(ckpt.relative_to(REPO)),weights_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        device='cpu',score_threshold=seg.SCORE_T,mask_threshold=seg.BIN_T,
        split='test (runs 85, 86, 87)',selection='GT area/boundary ranked; first nonempty prediction with example IoU > .30, for illustration only',
        purpose='qualitative examples; not a new test-set score',examples=evidence),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
