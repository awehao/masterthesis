import csv, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams['font.family']=['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus']=False

FILES = {
 'GMPC+CBF\n(ours)':'evaluation/results/sweep/_pre_sweep_gmpc_scan.csv',
 'MPPI':'evaluation/results/omnibot_dynamic_mppi.csv',
 'RPP':'evaluation/results/omnibot_dynamic_rpp.csv',
}
COLORS = ['#2e7d32', '#1565c0', '#c62828']   # green / blue / red

def num(x):
    if x is None: return None
    x=str(x).strip()
    if x=='' or x.lower()=='nan': return None
    try: return float(x)
    except: return None
def is_succ(r): return str(r['success']).strip().lower() in ('true','1')

agg={}
for name,f in FILES.items():
    rows=list(csv.DictReader(open(f)))
    N=len(rows)
    succ=[r for r in rows if is_succ(r)]
    def m(key, subset):
        vs=[num(r.get(key)) for r in subset]; vs=[v for v in vs if v is not None]
        return float(np.mean(vs)) if vs else float('nan')
    def sd(key, subset):
        vs=[num(r.get(key)) for r in subset]; vs=[v for v in vs if v is not None]
        return float(np.std(vs)) if len(vs)>1 else 0.0
    valid_clr=[r for r in rows if r.get('clearance_source','none')!='none']
    agg[name]=dict(
        succ_pct = 100.0*len(succ)/N,
        succ_lbl = f"{len(succ)}/{N}",
        path = m('path_length_m', succ), path_sd = sd('path_length_m', succ),
        arr  = m('arrival_time_s', succ), arr_sd = sd('arrival_time_s', succ),
        jerk = m('jerk_vx', succ),  # mean over successful runs (report method)
        clr  = m('min_clearance_m', valid_clr),
        rmse = m('tracking_rmse_m', succ),
        coll = sum(1 for r in rows if str(r.get('collided','')).lower() in ('true','1')),
        N=N,
    )
    a=agg[name]
    print(f"{name.splitlines()[0]:9s} succ={a['succ_lbl']}({a['succ_pct']:.0f}%) path={a['path']:.1f} "
          f"arr={a['arr']:.0f} jerk_vx={a['jerk']:.2f} minclr={a['clr']:+.3f} rmse={a['rmse']:.3f} coll={a['coll']}")

names=list(FILES.keys())
def vals(k): return [agg[n][k] for n in names]

fig,axes=plt.subplots(2,3,figsize=(13,7.5))
fig.suptitle('動態避障 三方比較 (N=40)  —  GMPC+CBF vs MPPI vs RPP', fontsize=15, fontweight='bold')

panels=[
 ('成功率 (%)',      'succ_pct', None,      '%.0f%%', False),
 ('路徑長 (m)  ↓越短越好','path','path_sd', '%.1f',  True),
 ('到達時間 (s)  ↓',  'arr','arr_sd',       '%.0f',  True),
 ('平滑度 jerk_vx (m/s²)  ↓','jerk',None,    '%.2f',  False),
 ('最小間距 (m)  ↑越大越安全','clr',None,    '%+.2f', False),
 ('追蹤 RMSE (m)',    'rmse',None,          '%.3f',  False),
]
for ax,(title,key,sdk,fmt,_) in zip(axes.flat,panels):
    v=vals(key); err=vals(sdk) if sdk else None
    bars=ax.bar(range(3), v, yerr=err, color=COLORS, capsize=4,
                edgecolor='black', linewidth=0.6, width=0.6)
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels([n.replace('\n',' ') for n in names], fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    if key=='clr': ax.axhline(0, color='red', lw=1, ls='--', alpha=0.7)
    for i,b in enumerate(bars):
        h=b.get_height()
        lbl=fmt % h
        if key=='succ_pct': lbl=f"{agg[names[i]]['succ_lbl']}\n{h:.0f}%"
        ax.annotate(lbl,(b.get_x()+b.get_width()/2, h),
                    ha='center', va='bottom' if h>=0 else 'top',
                    fontsize=9.5, fontweight='bold',
                    xytext=(0, 3 if h>=0 else -3), textcoords='offset points')
    pad=0.18*(max(v)-min(min(v),0) or 1)
    ax.set_ylim(min(min(v),0)-pad, max(v)+pad*1.6)

plt.tight_layout(rect=[0,0,1,0.96])
out='evaluation/results/figs/compare_3way_N40.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print("saved ->", out)
