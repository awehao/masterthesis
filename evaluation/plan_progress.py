#!/usr/bin/env python3
"""論文實驗進度快照（11/27 Howard：新規劃每個實驗一行、已完成／總趟數＋百分比，可即時更新的界面）。

用法：plan_progress.py [--stats <dyn_summary json>] --out <snapshot.json>
  總趟數來自 預口試論文/方法導向實驗與消融規劃_20261127.zip 的 core_planned_runs.csv／optional_planned_runs.csv（逐趟清單）。
  完成數來自 evaluation/results/plan1127/registry.csv（run_key,status,...；status=done 才算）。新規劃尚未開跑 ⇒ 全部 0。
  另列新規劃之前已開跑的導航動態批次 dyn1126（不計入新規劃總數；能否沿用為 N1 資料須另行確認共同條件）。
輸出一份快照 JSON（看板的一列資料），由 Claude 寫進進度看板。"""
import argparse
import csv
import datetime as dt
import glob
import io
import json
import os
import re
import zipfile
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
ZIP = os.path.join(WS, '預口試論文', '方法導向實驗與消融規劃_20261127.zip')
REG = os.path.join(HERE, 'results', 'plan1127', 'registry.csv')
DYN = 'dyn1126'

MOD = OrderedDict([
    ('M0', ('核心', '模型資料蒐集', '三構型區域 × 4 段獨立激勵（訓練 3／驗證 3／測試 6）')),
    ('N1', ('核心', '導航外部基線', '3 種複雜布局（多房間／辦公室走廊／倉儲貨架）× 靜態／交會 × 10 種子 × 4 導航方法')),
    ('N2', ('核心', '導航消融', '開闊／狹窄交會 × 10 種子 × B–E 四配置（A 沿用 N1）')),
    ('W1', ('核心', '全身外部基線', '4 類操作 × 一般／受限 × 10 種子 × 3 策略')),
    ('W2', ('核心', '全身控制消融', '圓柱／抽屜 × 10 種子 × B–F 五配置（A 沿用 W1）')),
    ('E0', ('核心', '住宅完整任務', '2 布局 × 無／有行人 × 10 種子 × 3 策略')),
    ('S1', ('選配', '底盤選位對照', '4 類操作 × 10 種子；最近可行底盤位置')),
    ('K1', ('選配', '門扇位姿限制對照', '門扇 10 種子；移除硬位姿限制')),
    ('H1', ('選配', '完整時域曲線', 'W2 的 20 案例補另外兩個步數')),
])
METH = {
    'nominal_excitation': '名目激勵紀錄',
    'nav_full': '本研究導航（完整）', 'mppi_omni_shared_guard': 'MPPI Omni', 'dwb_omni_shared_guard': 'DWB Omni',
    'rpp_shared_guard': 'RPP', 'nav_no_motion_prediction': 'B 不用障礙速度', 'nav_no_cbf': 'C 移除 CBF',
    'nav_no_reference_adjustment': 'D 移除參考調整', 'nav_speed_only': 'E 僅速度調整',
    'wb_mpc_full': '本研究 WB-MPC', 'holistic_qp_adapted': 'Holistic QP（移植）',
    'stop_before_approach_wb_mpc': 'WB-MPC 停駐後接近', 'wb_fixed_bias': 'B 固定偏置', 'wb_no_tcp_integral': 'C 關 TCP 積分',
    'wb_fixed_bias_no_tcp_integral': 'D 兩補償皆關', 'wb_direct_rate_fixed_model_no_tcp_integral': 'E 直接積分模型',
    'wb_horizon_1': 'F 單步', 'wb_nearest_feasible_base': '最近可行底盤位置', 'wb_no_door_pose_tube': '移除門扇位姿限制',
    'wb_other_multistep_horizon_1': '另一步數（一）', 'wb_other_multistep_horizon_2': '另一步數（二）',
}
N2_METH = {'B': 'nav_no_motion_prediction', 'C': 'nav_no_cbf', 'D': 'nav_no_reference_adjustment', 'E': 'nav_speed_only'}
N1_METH = {'gmpc_scan': 'nav_full', 'mppi': 'mppi_omni_shared_guard', 'dwb': 'dwb_omni_shared_guard', 'rpp': 'rpp_shared_guard'}
DYN_METH = [('gmpc_scan', '本研究 GMPC'), ('mppi', 'MPPI'), ('dwb', 'DWB'), ('rpp', 'RPP')]
DYN_COND = [('walk', 'const', '步行・等速'), ('walk', 'stopgo', '步行・停走'), ('fast', 'const', '快速・等速'), ('fast', 'stopgo', '快速・停走')]


def status(done, total, running=False):
    if total and done >= total:
        return '完成'
    return '進行中' if (done > 0 or running) else '未開始'


def plan_rows():
    z = zipfile.ZipFile(ZIP)
    runs = []
    for nm in ('experiment_plan/core_planned_runs.csv', 'experiment_plan/optional_planned_runs.csv'):
        runs += list(csv.DictReader(io.TextIOWrapper(z.open(nm), encoding='utf-8')))
    done = set()
    n1_done = {}                                                       # N1 改用三種複雜布局（11/27 Howard），名稱與規劃清單不同 ⇒ 依方法計數
    n2_done = {}
    if os.path.exists(REG):
        rows = list(csv.DictReader(open(REG)))
        done = {r['run_key'] for r in rows if r.get('status') == 'done'}
        _r2 = os.path.join(HERE, 'results', 'plan1127', 'n1_r2_routes.txt')   # 11/28：這 15 條降難度重跑（n1r2）⇒ 只計 n1r2；其餘 45 條沿用 n1main
        r2 = set(open(_r2).read().split()) if os.path.exists(_r2) else set()
        def _n1ok(r):
            ly, sd = r['run_key'].split('/')[2], int(r['run_key'].split('/')[4][4:])
            return r.get('batch') == ('n1r2' if f'{ly}:{sd}' in r2 else 'n1main')
        for k in {r['run_key'] for r in rows if r.get('status') == 'done' and r['run_key'].startswith('N1/') and _n1ok(r)}:
            meth = N1_METH.get(k.split('/')[-1], k.split('/')[-1])
            n1_done[meth] = n1_done.get(meth, 0) + 1
        for k in {r['run_key'] for r in rows if r.get('status') == 'done' and r['run_key'].startswith('N2/')}:   # N2：配置字母 ⇒ 規劃清單的方法名
            meth = N2_METH.get(k.split('/')[-1], k.split('/')[-1])
            n2_done[meth] = n2_done.get(meth, 0) + 1
    exps, cells = [], []
    for i, (m, (tier, name, design)) in enumerate(MOD.items()):
        x = [r for r in runs if r['module'] == m]
        d = (min(sum(n1_done.values()), len(x)) if m == 'N1' else min(sum(n2_done.values()), len(x)) if m == 'N2'
             else sum(1 for r in x if r['run_key'] in done))
        exps.append(dict(id=m, order=i + 1, tier=tier, code=m, name=name, design=design, done=d, total=len(x), status=status(d, len(x))))
        for meth, n in Counter(r['method'] for r in x).items():
            dd = (min(n1_done.get(meth, 0), n) if m == 'N1' else min(n2_done.get(meth, 0), n) if m == 'N2'
                  else sum(1 for r in x if r['method'] == meth and r['run_key'] in done))
            cells.append(dict(id=f'{m}:{meth}', module=m, method=METH.get(meth, meth), done=dd, total=n))
    return exps, cells


def dyn_rows(stats_path):
    R = os.path.join(HERE, 'results', DYN)
    log = open(os.path.join(R, DYN + '.log')).read().splitlines() if os.path.exists(os.path.join(R, DYN + '.log')) else []
    have = {}
    for f in glob.glob(os.path.join(R, '*_r*_*.csv')):
        m = re.match(r'(\w+?)_(\w+?)_r(\d+)_(gmpc_scan|mppi|dwb|rpp)\.csv$', os.path.basename(f))
        if m and sum(1 for _ in open(f)) > 1:
            have[(m.group(1), m.group(2), int(m.group(3)), m.group(4))] = True
    conds = []
    for sp, pt, lab in DYN_COND:
        row = dict(id=f'{sp}_{pt}', label=lab, total=10)
        for k, _ in DYN_METH:
            row[k] = sum(1 for (a, b, _, c) in have if a == sp and b == pt and c == k)
        conds.append(row)
    running = any('batch ' + DYN in s for s in log) and not any('BATCH DONE' in s for s in log)
    fill_running = any('FILL start' in s for s in log) and not any('FILL DONE' in s for s in log[-200:])
    n_done = len(have)
    skips = sum(1 for s in log if 'skipping' in s)
    caps = sum(1 for s in log if 'cap hit' in s and 'FILL' not in s)
    recent = []
    for s in [s for s in log if ' try' in s][-8:][::-1]:
        m = re.match(r'(\d\d:\d\d:\d\d) (?:FILL )?(\w+) (\w+) r(\d+) (\w+) try(\d+) : (.*)$', s)
        if not m:
            continue
        res = m.group(7)
        short = '到達' if 'goal reached' in res else ('逾時（200 s）' if 'cap hit' in res else ('系統失敗、重來' if 'skipping' in res else res[:20]))
        cond = dict((f'{a}_{b}', c) for a, b, c in DYN_COND).get(f'{m.group(2)}_{m.group(3)}', m.group(2) + ' ' + m.group(3))
        recent.append(dict(id=f'{m.group(1)}-{m.group(5)}-{m.group(6)}', time=m.group(1), case=f'{cond}・路線 {m.group(4)}',
                           method=dict(DYN_METH).get(m.group(5), m.group(5)), attempt=int(m.group(6)), result=short))
    methods = []
    stats_note = ''
    if stats_path and os.path.exists(stats_path):
        S = json.load(open(stats_path))
        summ = S['summary']
        nruns = len(S.get('runs', []))
        stats_note = f'依前 {nruns} 趟有效結果'
        for k, nm in DYN_METH:
            v = summ.get(f'all_all_{k}')
            if not v:
                continue
            p = summ.get(f'paired_time_{k}_minus_gmpc') if k != 'gmpc_scan' else None
            methods.append(dict(id=k, name=nm, n=v['n'], success=v['success'], contact_runs=v['dyn_contact_runs'],
                                t_d05=v['t_d05_mean'], time_mean=v['time_ok_mean'], path_ratio=v['path_ratio_ok_median'],
                                dt_vs_gmpc=(p or {}).get('median')))
    stopped = any('stop file' in s for s in log[-5:])
    batch = dict(done=n_done, total=160, skips=skips, caps=caps, running=running or fill_running,
                 status=('已停止' if stopped else status(n_done, 160, running or fill_running)), last=(log[-1] if log else ''), stats_note=stats_note)
    return batch, conds, methods, recent


def n1pilot_row(bname_=None):
    """一個 N1／N2／開發批次的進度列（批次名預設取 current_batch.txt）。"""
    R = os.path.join(HERE, 'results', bname_ or os.environ.get('PILOT_BATCH', open(os.path.join(HERE, 'results', 'plan1127', 'current_batch.txt')).read().strip() if os.path.exists(os.path.join(HERE, 'results', 'plan1127', 'current_batch.txt')) else 'n1pilot3'))
    lp = os.path.join(R, os.path.basename(R) + '.log')
    log = open(lp).read().splitlines() if os.path.exists(lp) else []
    n = sum(1 for f in glob.glob(os.path.join(R, 'n1_*_k*_*.csv')) if sum(1 for _ in open(f)) > 1)
    run = bool(log) and not any('BATCH DONE' in s for s in log)
    recent = []
    for s in [s for s in log if ' try' in s][-8:][::-1]:
        m = re.match(r'(\d\d:\d\d:\d\d) (?:L\d )?(n1_\w+?) (static|crossing) k(\d+) (\w+) try(\d+) : (.*)$', s)
        if m:
            res = m.group(7)
            short = '到達' if 'goal reached' in res else ('逾時（200 s）' if 'cap hit' in res else ('系統失敗、重來' if 'skipping' in res else res[:20]))
            lay = dict(n1_bigarena='多房間', n1_office='辦公室', n1_warehouse='倉儲').get(m.group(2), m.group(2))
            recent.append(dict(id=f'p{m.group(1)}-{m.group(5)}-{m.group(6)}', time=m.group(1), case=f"N1 先導・{lay}・{'交會' if m.group(3) == 'crossing' else '靜態'}",
                               method=dict(DYN_METH).get(m.group(5), m.group(5)), attempt=int(m.group(6)), result=short))
    nc = next((int(re.match(r'\d+', x.split('cases=')[1]).group()) for x in log if 'cases=' in x), None)
    if nc is None:                                                     # run_n1_batch.sh 沒有 cases= 行：由開頭的 layouts／conds／seeds 算
        hd = next((x for x in log if ' batch ' in x and 'layouts=' in x), '')
        mm = re.search(r'layouts=(.*?) conds=(.*?) seeds=(\d+)-(\d+)', hd)
        nc = len(mm.group(1).split()) * len(mm.group(2).split()) * (int(mm.group(4)) - int(mm.group(3)) + 1) if mm else 6
        mc = re.search(r'cfgs=(.*?) NAV_VMAX', hd)
        if mc:                                                         # N2／開發批次：每案例跑的是配置數，不是 4 個方法
            nc = nc * len(mc.group(1).split()) / 4.0
    bname = os.path.basename(R)
    nm_ = {'n1hardpilot': 'N1 加難先導（平行，已停）', 'n1hard12': 'N1 加難先導（種子 101，12 趟）', 'n1main': 'N1 正式第一輪（種子 1–20，240 趟）', 'n1r2': 'N1 第二版：15 條降難度路線 × 4 方法重跑（60 趟，其餘 45 條沿用第一輪）',
           'gmpcdev': 'GMPC 改進開發比較 v1（偵測錯誤，已停）', 'gmpcdev2': 'GMPC 改進開發比較（種子 101–104：原版 H／卡住解鎖 U／＋規劃器遮行人 UF）', 'gmpcdev3': 'GMPC 改進開發比較第二輪（種子 101–104：卡住解鎖＋緊急避讓 UE／三項全開 UFE）', 'n1v2': 'N1 正式重跑：GMPC 改進版（種子 1–20 × 3 布局，60 趟）', 'gmpcdev4': 'GMPC 擴大開發驗證（種子 105–114：原版 H／三項全開 UFE）', 'n2main': 'N2 導航消融（多房間＋辦公室 × 種子 1–10 × B／C／D／E）'}.get(bname, f'批次 {bname}')
    return dict(id='n1pilot', name=nm_, design='三種複雜布局 × 交會 × 4 方法（穿越、對向會車、轉角未知障礙）；GMPC 住宅任務版、四方法 0.26 m/s',
                done=n, total=int(round(nc * 4)), skips=sum(1 for s in log if 'skipping' in s), caps=sum(1 for s in log if 'cap hit' in s),
                running=run, status=status(n, int(round(nc * 4)), run), last=(log[-1] if log else ''), stats_note=''), recent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    exps, cells = plan_rows()
    batch, conds, methods, recent = dyn_rows(a.stats)
    pilot, precent = n1pilot_row()
    n1m = []                                                          # 目前 N1 批次的中間結果（只讀 CSV，快）
    try:
        import n1_summary as NS
        cb = open(os.path.join(HERE, 'results', 'plan1127', 'current_batch.txt')).read().strip()
        rr = NS.collect('n1final' if cb == 'n1r2' else cb, csv_only=True)   # 11/28：N1 最終＝45 條 n1main＋15 條 n1r2
        SS = NS.summarize(rr) if rr else {}
        for mth in NS.METHODS:
            v = SS.get(f'all|{mth}')
            if v:
                pv = SS.get(f'paired|{mth}_vs_gmpc', {})
                n1m.append(dict(id=mth, name=NS.NAME[mth], n=v['n'], success=v['success'], clean=v['clean'],
                                success_pct=round(100.0 * v['success'] / v['n'], 1), clean_pct=round(100.0 * v['clean'] / v['n'], 1),
                                min_clr=v['min_clr_mean'], time_mean=v['time_mean'], path_ratio=v['path_ratio_median'],
                                dt_vs_gmpc=pv.get('time_median')))
    except Exception as e_:                                            # noqa: BLE001
        print('n1 summary skipped:', repr(e_)[:80])
    extra = []
    dv = os.path.join(HERE, 'results', 'plan1127', 'current_batch_dev.txt')
    if os.path.exists(dv):
        dvb = open(dv).read().strip()
        if os.path.isdir(os.path.join(HERE, 'results', dvb)):
            row2, rec2 = n1pilot_row(dvb)
            row2['id'] = dvb
            extra.append(row2); precent = rec2 + precent
    recent = precent[:10]
    now = dt.datetime.now()
    snap = dict(at=now.isoformat(timespec='seconds'), at_label=now.strftime('%m/%d %H:%M'),
                experiments=exps, cells=cells, batch=[pilot] + extra,                    # 11/27 Howard：面板刪除 dyn1126
                conditions=[], methods=n1m, recent=recent)
    json.dump(snap, open(a.out, 'w'), ensure_ascii=False, indent=1)
    core = [e for e in exps if e['tier'] == '核心']
    print(f"核心 {sum(e['done'] for e in core)}/{sum(e['total'] for e in core)}；含選配 {sum(e['done'] for e in exps)}/{sum(e['total'] for e in exps)}；{pilot['name']} {pilot['done']}/{pilot['total']}")


if __name__ == '__main__':
    main()
