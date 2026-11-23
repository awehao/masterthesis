"""geom_v3 的間距與事件圖。三版度量並列，明示哪一版是評估用。"""
import csv, os, collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
import numpy as np
for f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',
          '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(f): fm.fontManager.addfont(f); break
plt.rcParams.update({'font.family':['Noto Sans CJK JP'],'axes.unicode_minus':False,
                     'figure.dpi':130})
OFFC, ONC = '#B45309', '#1D4ED8'
B='evaluation/results/geom_v3'
rows=list(csv.DictReader(open(f'{B}/confirm40/per_run.csv')))
for r in rows:
    for k in ('old_disc_min','full_sdf_min','exec_min'): r[k]=float(r[k])

# ---- 圖 A：三版度量逐趟比較 ----
fig,axes=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={'width_ratios':[2.2,1]})
ax=axes[0]
x=np.arange(len(rows))
ordr=sorted(range(len(rows)), key=lambda i: rows[i]['exec_min'])
for key,lab,c,mk in (('old_disc_min','舊圓盤（YAML 外接圓）','#9CA3AF','v'),
                     ('full_sdf_min','SDF 完整定義（含漏建的 c2）','#7A5C00','s'),
                     ('exec_min','執行幾何（評估用）','#15803D','o')):
    ax.plot(x,[rows[i][key] for i in ordr],mk,ms=4.5,color=c,label=lab,alpha=.85)
ax.axhline(0,color='#111',lw=1.2)
ax.set_xlabel('38 趟可評估資料（依執行幾何間距排序）'); ax.set_ylabel('最小間距（m）')
ax.set_title('三版度量逐趟比較 — 負值只出現在前兩版',fontsize=11)
ax.legend(fontsize=8.5,frameon=False); ax.grid(alpha=.25,axis='y')

ax=axes[1]
neg=[sum(1 for r in rows if r[k]<0) for k in ('old_disc_min','full_sdf_min','exec_min')]
bars=ax.bar(['舊圓盤','完整 SDF','執行幾何'],neg,
            color=['#9CA3AF','#7A5C00','#15803D'])
for b,v in zip(bars,neg):
    ax.text(b.get_x()+b.get_width()/2,v+0.3,str(v),ha='center',fontsize=11,weight='bold')
ax.set_ylabel('負間距趟數 / 38'); ax.set_ylim(0,15)
ax.set_title('負值趟數：13 → 7 → 0',fontsize=11); ax.grid(alpha=.25,axis='y')
fig.suptitle('間距度量改版：舊圓盤近似 → SDF 完整定義 → 重建的執行幾何',fontsize=12.5)
fig.text(.5,.015,'負值不是碰撞證據 —— 前兩版把「SDF 有、模擬器未生成」的第二塊 '
         'collision 算了進去（dyn_obs_4）。執行幾何才對應實際跑過的場景。',
         ha='center',fontsize=8,color='#4B5563')
fig.tight_layout(rect=[0,.045,1,.95]); fig.savefig(f'{B}/figA_metric_versions.png')
plt.close(fig)

# ---- 圖 B：主要指標 ----
R={(int(r['seed']),r['rep'],r['cond']):r for r in rows}
seeds=sorted({int(r['seed']) for r in rows})
fig,ax=plt.subplots(figsize=(9.5,5.2))
inc,exc=[],[]
for i,s in enumerate(seeds):
    ds=[]
    for rep in ('r1','r2'):
        o,n=R.get((s,rep,'off')),R.get((s,rep,'on'))
        ds.append(n['exec_min']-o['exec_min'] if o and n else None)
    v=[d for d in ds if d is not None]
    ok = len(v)==2 and all(abs(d)>=0.005 for d in v) and v[0]*v[1]>0
    for j,d in enumerate(ds):
        if d is None: continue
        ax.plot(i, d, 'o' if j==0 else 's', ms=8,
                color=(ONC if d>0 else OFFC) if ok else '#9CA3AF',
                mfc=(ONC if d>0 else OFFC) if ok else 'none', mew=1.5)
    (inc if ok else exc).append(s)
ax.axhline(0,color='#111',lw=1.2)
ax.axhspan(-0.005,0.005,color='#9CA3AF',alpha=.18)
ax.text(len(seeds)-.4,0.006,'±5 mm 方向不定帶',fontsize=7.5,color='#4B5563',ha='right')
ax.set_xticks(range(len(seeds)))
ax.set_xticklabels([f'seed {s}'+('' if s in inc else '\n(排除)') for s in seeds],fontsize=8.5)
ax.set_ylabel('最小間距 ON − OFF（m，執行幾何）')
ax.set_title('主要指標：每 seed 兩次配對的差值（○ 第 1 次  □ 第 2 次）',fontsize=12)
ax.grid(alpha=.25,axis='y')
fig.text(.5,.02,'計入 6 個 seed（ON 較大 2、ON 較小 4），雙尾符號檢定 p = 0.6875，α = 0.05 '
         '→ 未取得一致變化方向的顯著證據。灰色空心 = 依事前規則排除。',
         ha='center',fontsize=8.5,color='#4B5563')
fig.tight_layout(rect=[0,.06,1,1]); fig.savefig(f'{B}/figB_primary.png'); plt.close(fig)

# ---- 圖 C：遭遇事件（以執行幾何重新辨識）----
ev=list(csv.DictReader(open(f'{B}/confirm40/events.csv')))
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
PH=['pre','core','post']; LBL=['前','中','後']
for ax,(key,t) in zip(axes,[('vyb','|v_y,body|（m/s）'),('ang','車頭−行進夾角（°）')]):
    for cond,c in (('off',OFFC),('on',ONC)):
        E=[e for e in ev if e['cond']==cond]
        M=[]
        for e in E:
            y=[float(e[f'{p}_{key}']) for p in PH]
            if any(np.isnan(y)): continue
            ax.plot(range(3),y,'-',color=c,alpha=.18,lw=.9); M.append(y)
        M=np.array(M)
        ax.plot(range(3),np.median(M,axis=0),'-o',color=c,lw=2.6,ms=7,
                label=f'{cond.upper()} 中位（n={len(M)}）',zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(LBL,fontsize=11)
    ax.set_title(t,fontsize=10.5); ax.grid(alpha=.25,axis='y')
    ax.legend(fontsize=8,frameon=False)
fig.suptitle('遭遇事件（以執行幾何重新辨識，門檻未變：進入 1.00 m、最近需 ≤0.50 m）',
             fontsize=11.5)
fig.text(.5,.02,f'confirm 批共 {len(ev)} 個事件；細線 = 一個事件，粗線 = '
         '先取每事件各階段中位再取事件間中位。事件彼此相關，非獨立樣本。',
         ha='center',fontsize=8,color='#4B5563')
fig.tight_layout(rect=[0,.06,1,.94]); fig.savefig(f'{B}/figC_events.png'); plt.close(fig)
print('figA_metric_versions.png / figB_primary.png / figC_events.png 已輸出 ->', B)
