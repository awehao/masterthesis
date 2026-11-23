import csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family':['Noto Sans CJK JP'],'axes.unicode_minus':False})
def num(x):
    x=str(x).strip()
    try: return float(x)
    except: return None
R=list(csv.DictReader(open('evaluation/results/sweep/_pre_sweep_gmpc_scan.csv')))
def col(k):
    v=[num(r[k]) for r in R]; return [x for x in v if x is not None]
mean=np.mean(col('solve_time_mean_ms'))
p95 =np.mean(col('solve_time_p95_ms'))
mx  =np.mean(col('solve_time_max_ms'))
budget=50.0

fig,ax=plt.subplots(figsize=(6.8,4.6))
labels=['平均','p95','最大']; vals=[mean,p95,mx]; cols=['#00695c','#2e8b8b','#7bb5b5']
b=ax.bar(labels,vals,color=cols,width=0.6,edgecolor='#222',lw=0.8,zorder=3)
ax.axhline(budget,color='#c62828',ls='--',lw=2,zorder=2)
ax.text(2.35,budget+1.5,'控制週期 50ms (20Hz)',color='#c62828',ha='right',fontsize=10,fontweight='bold')
for bb,v in zip(b,vals):
    ax.annotate(f'{v:.1f} ms',(bb.get_x()+bb.get_width()/2,v),ha='center',va='bottom',
                fontsize=11,fontweight='bold',xytext=(0,3),textcoords='offset points')
ax.set_ylim(0,budget*1.2); ax.set_ylabel('QP 求解時間 (ms)',fontsize=11)
ax.set_title('即時性:GMPC 求解時間遠低於控制週期 (N=40)',fontsize=12.5,fontweight='bold')
ax.grid(axis='y',alpha=0.25); ax.set_axisbelow(True)
for s in ('top','right'): ax.spines[s].set_visible(False)
# headroom annotation
ax.annotate('',xy=(2,budget),xytext=(2,mx),arrowprops=dict(arrowstyle='<->',color='#888'))
ax.text(1.7,(mx+budget)/2,f'餘裕\n{budget-mx:.0f}ms',color='#666',fontsize=8.5,ha='right',va='center')
plt.tight_layout()
out='evaluation/results/figs/solve_time_realtime.png'
plt.savefig(out,dpi=160,bbox_inches='tight'); print('saved ->',out,f'| {mean:.1f}/{p95:.1f}/{mx:.1f} ms')
