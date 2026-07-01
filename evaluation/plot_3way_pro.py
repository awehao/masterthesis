"""Publication-quality 3-way benchmark comparison (N=40).
Reads the final CSVs, aggregates (success over all 40; time/path/rmse over
successful; jerk/smooth over successful; clearance over valid), renders a clean
6-panel figure with std error bars and 'ours' highlighting."""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family':['Noto Sans CJK JP'],'axes.unicode_minus':False,
    'axes.edgecolor':'#555','axes.linewidth':0.8,'font.size':10})

FILES={'GMPC+CBF':'evaluation/results/sweep/_pre_sweep_gmpc_scan.csv',
       'MPPI':'evaluation/results/omnibot_dynamic_mppi.csv',
       'RPP':'evaluation/results/omnibot_dynamic_rpp.csv'}
NAMES=list(FILES); COL=['#00695c','#1f6fb2','#c1443c']
def num(x):
    x=str(x).strip()
    return None if x==''or x.lower()=='nan' else (float(x) if _f(x) else None)
def _f(x):
    try: float(x); return True
    except: return False
def sc(r): return str(r['success']).lower() in ('true','1')

A={}
for nm,f in FILES.items():
    R=list(csv.DictReader(open(f))); S=[r for r in R if sc(r)]
    V=[r for r in R if r.get('clearance_source','none')!='none']
    def stat(k,sub):
        v=[num(r[k]) for r in sub]; v=[x for x in v if x is not None]
        return (np.mean(v),np.std(v)) if v else (np.nan,0)
    A[nm]=dict(N=len(R),ns=len(S),
        succ=100*len(S)/len(R), succ_lbl=f"{len(S)}/{len(R)}",
        coll=100*sum(1 for r in R if str(r.get('collided','')).lower() in('true','1'))/len(R),
        arr=stat('arrival_time_s',S), path=stat('path_length_m',S),
        rmse=stat('tracking_rmse_m',S),
        svx=stat('smooth_vx',S)[0], svy=stat('smooth_vy',S)[0], swz=stat('smooth_wz',S)[0])

fig,ax=plt.subplots(2,3,figsize=(13.5,8)); ax=ax.flat
fig.suptitle('動態避障 三方 benchmark(N=40)',fontsize=16,fontweight='bold',y=0.98)
x=np.arange(3)
def bars(a,vals,errs=None,fmt='%.1f',lbls=None,title=''):
    b=a.bar(x,vals,yerr=errs,color=COL,width=0.62,edgecolor='#222',lw=0.7,
            capsize=4,error_kw=dict(elinewidth=1.1,ecolor='#333'))
    b[0].set_hatch('//')                      # mark ours
    a.set_title(title,fontsize=12,fontweight='bold',pad=8)
    a.set_xticks(x); a.set_xticklabels(NAMES,fontsize=10)
    a.grid(axis='y',alpha=0.25); a.set_axisbelow(True)
    for sp in ('top','right'): a.spines[sp].set_visible(False)
    for i,bb in enumerate(b):
        h=bb.get_height(); t=(lbls[i] if lbls else fmt%h)
        a.annotate(t,(bb.get_x()+bb.get_width()/2,h),ha='center',va='bottom',
                   fontsize=9.5,fontweight='bold',xytext=(0,3+(errs[i] if errs else 0)*0),
                   textcoords='offset points')
    top=max(vals)+ (max(errs) if errs else 0)
    a.set_ylim(0,top*1.22)

bars(ax[0],[A[n]['succ'] for n in NAMES],fmt='%.0f%%',
     lbls=[f"{A[n]['succ_lbl']}\n{A[n]['succ']:.0f}%" for n in NAMES],title='成功率 (%)')
bars(ax[1],[A[n]['coll'] for n in NAMES],fmt='%.0f%%',
     lbls=[f"{A[n]['coll']:.0f}%" for n in NAMES],title='碰撞趟比例 (%)')
bars(ax[2],[A[n]['arr'][0] for n in NAMES],[A[n]['arr'][1] for n in NAMES],
     '%.0f',title='到達時間 (s)')
bars(ax[3],[A[n]['path'][0] for n in NAMES],[A[n]['path'][1] for n in NAMES],
     '%.1f',title='路徑長 (m)')
# smoothness grouped (vx/vy/wz)
a=ax[4]; w=0.25; keys=['svx','svy','swz']; lab=['v_x','v_y','ω_z']
for j,k in enumerate(keys):
    vv=[A[n][k] for n in NAMES]
    bb=a.bar(x+(j-1)*w,vv,w,color=COL,edgecolor='#222',lw=0.5,alpha=[1,.65,.4][j])
    for r in bb: a.annotate('%.2f'%r.get_height(),(r.get_x()+r.get_width()/2,r.get_height()),
                 ha='center',va='bottom',fontsize=7,xytext=(0,1),textcoords='offset points')
a.set_title('控制平滑度 std(cmd)',fontsize=12,fontweight='bold',pad=8)
a.set_xticks(x); a.set_xticklabels(NAMES,fontsize=10); a.grid(axis='y',alpha=0.25); a.set_axisbelow(True)
for sp in ('top','right'): a.spines[sp].set_visible(False)
from matplotlib.patches import Patch
a.legend([Patch(facecolor='gray',alpha=al) for al in (1,.65,.4)],lab,fontsize=8,ncol=3,loc='upper center',title='軸')
a.set_ylim(0,max(A[n][k] for n in NAMES for k in keys)*1.3)
bars(ax[5],[A[n]['rmse'][0] for n in NAMES],[A[n]['rmse'][1] for n in NAMES],
     '%.3f',title='追蹤 RMSE (m)')

fig.text(0.5,0.005,'斜線填充 = 本方法 (ours)   ·   誤差棒 = ±1σ(成功趟)',
         ha='center',fontsize=9,color='#555')
plt.tight_layout(rect=[0,0.02,1,0.96])
out='evaluation/results/figs/compare_3way_pro.png'
plt.savefig(out,dpi=160,bbox_inches='tight'); print('saved ->',out)
for n in NAMES:
    print(n,A[n]['succ_lbl'],f"coll={A[n]['coll']:.0f}%",
          f"arr={A[n]['arr'][0]:.0f}±{A[n]['arr'][1]:.0f}",
          f"path={A[n]['path'][0]:.1f}±{A[n]['path'][1]:.1f}",
          f"rmse={A[n]['rmse'][0]:.3f}")
