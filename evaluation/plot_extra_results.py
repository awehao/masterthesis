import csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family':['Noto Sans CJK JP'],'axes.unicode_minus':False})
FILES={'GMPC+CBF':'evaluation/results/sweep/_pre_sweep_gmpc_scan.csv',
       'MPPI':'evaluation/results/omnibot_dynamic_mppi.csv',
       'RPP':'evaluation/results/omnibot_dynamic_rpp.csv'}
NAMES=list(FILES); COL=['#00695c','#1f6fb2','#c1443c']
def num(x):
    x=str(x).strip()
    try: return float(x)
    except: return None
def sc(r): return str(r['success']).lower() in ('true','1')
D={}
for nm,f in FILES.items():
    R=list(csv.DictReader(open(f))); N=len(R)
    D[nm]=dict(N=N,
        succ=100*sum(sc(r) for r in R)/N,
        coll=100*sum(1 for r in R if str(r.get('collided','')).lower() in('true','1'))/N,
        arr=[num(r['arrival_time_s']) for r in R if sc(r) and num(r['arrival_time_s']) is not None],
        clr=[num(r['min_clearance_m']) for r in R if r.get('clearance_source','none')!='none'
             and num(r['min_clearance_m']) is not None])

fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(15,4.6))

# ② 安全-成功 權衡散點
for i,nm in enumerate(NAMES):
    a1.scatter(D[nm]['coll'],D[nm]['succ'],s=420,color=COL[i],edgecolor='#222',lw=1.2,zorder=3)
    a1.annotate(nm,(D[nm]['coll'],D[nm]['succ']),xytext=(8,-4),textcoords='offset points',
                fontsize=11,fontweight='bold')
a1.scatter([],[]);
a1.text(2,102,'理想角落',color='#00695c',fontsize=9)
a1.set_xlabel('碰撞趟比例 (%)  →越右越危險',fontsize=10)
a1.set_ylabel('成功率 (%)',fontsize=10)
a1.set_title('② 安全–成功 權衡',fontsize=12.5,fontweight='bold')
a1.set_xlim(-5,95); a1.set_ylim(-5,112); a1.grid(alpha=0.25); a1.set_axisbelow(True)
for s in ('top','right'): a1.spines[s].set_visible(False)

# ④ 到達時間 box
data=[D[n]['arr'] for n in NAMES]
bp=a2.boxplot(data,patch_artist=True,widths=0.55,showmeans=True,
              meanprops=dict(marker='D',mfc='white',mec='black',ms=6))
for p,c in zip(bp['boxes'],COL): p.set_facecolor(c); p.set_alpha(.75); p.set_edgecolor('#222')
for med in bp['medians']: med.set_color('#222')
a2.set_xticklabels(NAMES,fontsize=10); a2.set_ylabel('到達時間 (s)',fontsize=10)
a2.set_title('④ 到達時間分布(成功趟)',fontsize=12.5,fontweight='bold')
a2.grid(axis='y',alpha=0.25); a2.set_axisbelow(True)
for s in ('top','right'): a2.spines[s].set_visible(False)

# ③ 最小間距 box (bonus)
data2=[D[n]['clr'] for n in NAMES]
bp2=a3.boxplot(data2,patch_artist=True,widths=0.55)
for p,c in zip(bp2['boxes'],COL): p.set_facecolor(c); p.set_alpha(.75); p.set_edgecolor('#222')
for med in bp2['medians']: med.set_color('#222')
a3.axhline(0,color='#c62828',ls='--',lw=1.6)
a3.text(3.4,0.01,'碰撞線',color='#c62828',ha='right',fontsize=9)
a3.set_xticklabels(NAMES,fontsize=10); a3.set_ylabel('最小間距 (m)',fontsize=10)
a3.set_title('③ 最小間距分布(<0=撞穿)',fontsize=12.5,fontweight='bold')
a3.grid(axis='y',alpha=0.25); a3.set_axisbelow(True)
for s in ('top','right'): a3.spines[s].set_visible(False)

plt.tight_layout()
out='evaluation/results/figs/extra_results.png'
plt.savefig(out,dpi=160,bbox_inches='tight'); print('saved ->',out)
for n in NAMES: print(n,f"succ={D[n]['succ']:.0f}% coll={D[n]['coll']:.0f}% arr_n={len(D[n]['arr'])} clr_n={len(D[n]['clr'])}")
