import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams['font.family']=['Noto Sans CJK JP']; plt.rcParams['axes.unicode_minus']=False
rng=np.random.default_rng(5)

wall_y=2.0; dyn=(4.0,0.95,0.38)
def scene(ax):
    ax.add_patch(Rectangle((0,wall_y),8,0.5,color='#607d8b'))
    ax.text(0.15,wall_y+0.17,'牆',color='white',fontsize=10,fontweight='bold')
    ax.add_patch(Circle((dyn[0],dyn[1]),dyn[2],color='#37474f'))
    ax.text(dyn[0],dyn[1],'動態',color='white',ha='center',va='center',fontsize=7.5)
    ax.add_patch(FancyArrowPatch((dyn[0],dyn[1]-0.5),(dyn[0],dyn[1]-0.9),
                 color='#37474f',arrowstyle='-|>',mutation_scale=12,lw=1.8))
    ax.set_xlim(0,8); ax.set_ylim(-0.1,2.7); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

fig,ax=plt.subplots(1,3,figsize=(15.5,4.4))
px=np.linspace(0.4,7.6,120)

# 1) 沒有 static-CBF -> 撞牆
scene(ax[0])
py=1.15+0.95*np.exp(-((px-4)/1.1)**2)
ax[0].plot(px,py,'-',color='#2e7d32',lw=2.3)
ax[0].plot(px[np.argmax(py)],py.max(),'x',color='#c62828',ms=15,mew=3)
ax[0].text(px[np.argmax(py)],py.max()+0.16,'撞牆!',color='#c62828',ha='center',fontsize=11,fontweight='bold')
ax[0].set_title('① 沒有 static-CBF\nCBF 對牆全盲 → 側移撞牆',fontsize=11.5,fontweight='bold')

# 2) scan-based -> 不撞但抖 (flicker/chatter)
scene(ax[1])
wx=np.linspace(2.6,5.4,5)+rng.uniform(-0.3,0.3,5); wy=wall_y+rng.uniform(0,0.15,5)
ax[1].plot(wx,wy,'x',color='#c62828',ms=8,mew=2,ls='')
for X,Y in zip(wx,wy): ax[1].add_patch(Circle((X,Y),0.30,fill=False,ec='#c62828',ls=':',lw=1.1,alpha=.7))
base=1.02+0.5*np.exp(-((px-4)/1.0)**2)
chat=base+ (np.abs(px-4)<1.4)*0.09*np.sin(px*11)   # jitter near wall
ax[1].plot(px,chat,'-',color='#2e7d32',lw=2.3)
ax[1].text(4,1.75,'牆點跳動→抖',color='#c62828',ha='center',fontsize=9,fontweight='bold')
ax[1].set_title('② scan-based static-CBF\n牆點每幀跳動 → 不撞但 chatter',fontsize=11.5,fontweight='bold')

# 3) map-based -> 平滑
scene(ax[2])
wx3=np.linspace(2.6,5.4,5); wy3=np.full(5,wall_y)
ax[2].plot(wx3,wy3,'x',color='#c62828',ms=8,mew=2,ls='')
for X,Y in zip(wx3,wy3): ax[2].add_patch(Circle((X,Y),0.30,fill=False,ec='#c62828',ls=':',lw=1.1,alpha=.7))
py3=1.02+0.5*np.exp(-((px-4)/1.0)**2)
ax[2].plot(px,py3,'-',color='#2e7d32',lw=2.3)
ax[2].text(6.5,1.35,'✓ 平滑\n不抖不撞',color='#2e7d32',ha='center',fontsize=10,fontweight='bold')
ax[2].set_title('③ map-based static-CBF（定案）\n地圖 EDT 最近牆點 → 平滑',fontsize=11.5,fontweight='bold')

fig.suptitle('靜態 CBF 的演進：沒有 → scan-based → map-based（示意）',fontsize=14,fontweight='bold')
fig.text(0.5,0.02,'綠=機器人軌跡   紅色叉 / 虛圈 = CBF 用的牆點與 keep-out',ha='center',fontsize=9,color='#555')
plt.tight_layout(rect=[0,0.03,1,0.93])
out='evaluation/results/figs/static_cbf_evolution.png'
plt.savefig(out,dpi=155,bbox_inches='tight'); print('saved ->',out)
