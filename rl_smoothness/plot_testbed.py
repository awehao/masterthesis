import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family':['Noto Sans CJK JP'],'axes.unicode_minus':False})
from env2d import Avoid2DEnv

def rollout(seed, **kw):
    env=Avoid2DEnv(seed=seed, **kw); env.reset(seed=seed)
    P=[env.p.copy()]; OB=[[ (o['x'],o['y']) for o in env.obs] ]
    done=False
    while not done:
        _,_,done,info=env.step(None)
        P.append(env.p.copy()); OB.append([(o['x'],o['y']) for o in env.obs])
    return np.array(P), np.array(OB), env.episode_metrics(), env.goal, info

SEED=7
fig,axes=plt.subplots(1,2,figsize=(12,6))
cfgs=[('開闊無噪 (無 wiggle)', dict(n_obs=4)),
      ('加感知噪 (重現 wiggle)', dict(n_obs=4, obs_noise_pos=0.05, obs_noise_vel=0.20))]
for ax,(title,kw) in zip(axes,cfgs):
    P,OB,m,goal,info=rollout(SEED,**kw)
    # obstacle paths (faint) + final positions
    for j in range(OB.shape[1]):
        ax.plot(OB[:,j,0],OB[:,j,1],color='#bbb',lw=0.8,alpha=0.6)
        ax.add_patch(plt.Circle(OB[-1,j],0.25,color='#888',alpha=0.5))
    # robot trajectory
    ax.plot(P[:,0],P[:,1],'-',color='#2e7d32',lw=2.2,label='機器人軌跡')
    ax.plot(*P[0],'o',color='blue',ms=10,label='起點')
    ax.plot(*goal,'*',color='red',ms=18,label='終點')
    ax.set_title(f"{title}\nheading={m['deg_per_m']:.1f}°/m  jerk_p95={m['jerk_p95']:.2f}",
                 fontsize=12,fontweight='bold')
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_aspect('equal')
    ax.grid(alpha=0.3); ax.legend(loc='lower right',fontsize=9)
fig.suptitle('RL 平滑度測試場:感知噪重現 wiggle(同場景對照)',fontsize=14,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
out='/home/howardchen/masterthesis/evaluation/results/figs/rl_testbed_wiggle.png'
plt.savefig(out,dpi=150,bbox_inches='tight'); print('saved ->',out)
