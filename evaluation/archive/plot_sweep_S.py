import csv, glob, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams['font.family']=['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus']=False

def num(x):
    x=str(x).strip()
    if x=='' or x.lower()=='nan': return None
    try: return float(x)
    except: return None
def succ(r): return str(r['success']).lower() in ('true','1')

Ss, arr, path, smo = [], [], [], []
for f in sorted(glob.glob('evaluation/results/sweep/S_*.csv'),
                key=lambda p:int(re.search(r'S_(\d+)',p).group(1))):
    S=int(re.search(r'S_(\d+)',f).group(1))
    rows=[r for r in csv.DictReader(open(f)) if succ(r)]
    def m(k):
        v=[num(r[k]) for r in rows]; v=[x for x in v if x is not None]
        return np.mean(v) if v else np.nan
    Ss.append(S); arr.append(m('arrival_time_s')); path.append(m('path_length_m')); smo.append(m('smooth_vy'))
Ss=np.array(Ss)
best=15

fig,axes=plt.subplots(1,3,figsize=(13,4.2))
fig.suptitle('輸入平滑成本 S 之 sweep(gmpc_scan,每點 5 趟)', fontsize=14, fontweight='bold')
panels=[('到達時間 (s)',arr,'%.0f'),('路徑長 (m)',path,'%.1f'),('橫向平滑度 σ(v_y)',smo,'%.3f')]
for ax,(title,y,fmt) in zip(axes,panels):
    y=np.array(y)
    ax.axvspan(13,21,color='#c8e6c9',alpha=0.6,zorder=0,label='甜蜜點')
    ax.plot(Ss,y,'-o',color='#2e7d32',lw=2,ms=6,zorder=3)
    bi=list(Ss).index(best)
    ax.plot(best,y[bi],'o',ms=13,mfc='none',mec='#c62828',mew=2.5,zorder=4)
    ax.annotate(f'S={best}\n(最佳)',(best,y[bi]),textcoords='offset points',
                xytext=(8,10),color='#c62828',fontweight='bold',fontsize=10)
    for xv,yv in zip(Ss,y):
        ax.annotate(fmt%yv,(xv,yv),textcoords='offset points',xytext=(0,-14),
                    ha='center',fontsize=7.5,color='#444')
    ax.set_title(title,fontsize=12,fontweight='bold')
    ax.set_xlabel('S(輸入增量權重)'); ax.set_xticks(Ss)
    ax.grid(alpha=0.3)
    pad=(np.nanmax(y)-np.nanmin(y))*0.25 or 1
    ax.set_ylim(np.nanmin(y)-pad, np.nanmax(y)+pad)
plt.tight_layout(rect=[0,0,1,0.94])
out='evaluation/results/figs/sweep_S.png'
plt.savefig(out,dpi=150,bbox_inches='tight')
print('saved ->',out)
print('S :',list(Ss)); print('arr:',[f'{v:.0f}' for v in arr])
print('path:',[f'{v:.1f}' for v in path]); print('smo:',[f'{v:.3f}' for v in smo])
