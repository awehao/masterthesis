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
    if os.path.exists(REG):
        rows = list(csv.DictReader(open(REG)))
        done = {r['run_key'] for r in rows if r.get('status') == 'done'}
        for k in {r['run_key'] for r in rows if r.get('status') == 'done' and r['run_key'].startswith('N1/')}:
            meth = N1_METH.get(k.split('/')[-1], k.split('/')[-1])
            n1_done[meth] = n1_done.get(meth, 0) + 1
    exps, cells = [], []
    for i, (m, (tier, name, design)) in enumerate(MOD.items()):
        x = [r for r in runs if r['module'] == m]
        d = sum(1 for r in x if r['run_key'] in done) if m != 'N1' else min(sum(n1_done.values()), len(x))
        exps.append(dict(id=m, order=i + 1, tier=tier, code=m, name=name, design=design, done=d, total=len(x), status=status(d, len(x))))
        for meth, n in Counter(r['method'] for r in x).items():
            dd = sum(1 for r in x if r['method'] == meth and r['run_key'] in done) if m != 'N1' else min(n1_done.get(meth, 0), n)
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


def n1pilot_row():
    """N1 先導（三布局 × 靜態／交會 × 種子 101 × 4 方法 = 24 趟）。"""
    R = os.path.join(HERE, 'results', 'n1pilot')
    lp = os.path.join(R, 'n1pilot.log')
    log = open(lp).read().splitlines() if os.path.exists(lp) else []
    n = sum(1 for f in glob.glob(os.path.join(R, 'n1_*_k*_*.csv')) if sum(1 for _ in open(f)) > 1)
    run = bool(log) and not any('BATCH DONE' in s for s in log)
    recent = []
    for s in [s for s in log if ' try' in s][-8:][::-1]:
        m = re.match(r'(\d\d:\d\d:\d\d) (n1_\w+?) (static|crossing) k(\d+) (\w+) try(\d+) : (.*)$', s)
        if m:
            res = m.group(7)
            short = '到達' if 'goal reached' in res else ('逾時（200 s）' if 'cap hit' in res else ('系統失敗、重來' if 'skipping' in res else res[:20]))
            lay = dict(n1_bigarena='多房間', n1_office='辦公室', n1_warehouse='倉儲').get(m.group(2), m.group(2))
            recent.append(dict(id=f'p{m.group(1)}-{m.group(5)}-{m.group(6)}', time=m.group(1), case=f"N1 先導・{lay}・{'交會' if m.group(3) == 'crossing' else '靜態'}",
                               method=dict(DYN_METH).get(m.group(5), m.group(5)), attempt=int(m.group(6)), result=short))
    return dict(id='n1pilot', name='N1 先導（種子 101）', design='三種複雜布局 × 靜態／交會 × 4 方法；GMPC 住宅任務版、四方法 0.26 m/s',
                done=n, total=24, skips=sum(1 for s in log if 'skipping' in s), caps=sum(1 for s in log if 'cap hit' in s),
                running=run, status=status(n, 24, run), last=(log[-1] if log else ''), stats_note=''), recent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    exps, cells = plan_rows()
    batch, conds, methods, recent = dyn_rows(a.stats)
    pilot, precent = n1pilot_row()
    recent = (precent + recent)[:10]
    now = dt.datetime.now()
    snap = dict(at=now.isoformat(timespec='seconds'), at_label=now.strftime('%m/%d %H:%M'),
                experiments=exps, cells=cells, batch=[pilot, dict(id=DYN, name='dyn1126（11/26，已停）', design='行人太快、GMPC 設定不對，只當有瑕疵的先導', **batch)],
                conditions=conds, methods=methods, recent=recent)
    json.dump(snap, open(a.out, 'w'), ensure_ascii=False, indent=1)
    core = [e for e in exps if e['tier'] == '核心']
    print(f"核心 {sum(e['done'] for e in core)}/{sum(e['total'] for e in core)}；含選配 {sum(e['done'] for e in exps)}/{sum(e['total'] for e in exps)}；N1 先導 {pilot['done']}/24；dyn1126 {batch['done']}/160（{batch['status']}）")


if __name__ == '__main__':
    main()
