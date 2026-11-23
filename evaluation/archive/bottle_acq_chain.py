#!/usr/bin/env python3
"""目標取得事件鏈對照（Codex 20261102_092128；只讀既有紀錄，不重訓、不重跑）。
對兩趟（原區域 U3：bottle_online5_u3_1；R-B1：bottle_rb1_1）列 U 段開始到抓取計畫承接附近：
  瓶入畫（w1_log.expected：以真值與相機位姿投影的可見判定，評估用）→ 分割選中（候選列 valid）→ xy 有效 → 支撐有效（support_observed）／高度有效 →
  連續穩定候選（known_type k_ready）→ 快照檢查完成（check t_publish）→ 規劃承接（grasp plan t_acc）。
每個事件記任務時刻與當時區域 x（底盤世界 x − 區域偏移）、底盤區域 y、偏航、V_HIGH 段；另列 U 段到承接之間各旗標的影格比例、k_why 原因分布、
相機到瓶距離。代表影格：w1_frames（約 1 Hz 存檔的手腕 RGB）取最接近「首次入畫、首次分割有效、首次穩定候選」的影格存 PNG。
輸出 results/bottle/acq_chain.json、第七次進度報告素材/圖表/acq_chain_frames/。
    python3 evaluation/bottle_acq_chain.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = dict(orig_u3='bottle_online5_u3_1', rb1='bottle_rb1_1')
OUT = os.path.join(HERE, 'results', 'bottle', 'acq_chain.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'acq_chain_frames')
WHY = {0: 'ok', 1: '列數不足', 2: '逾期', 3: '跨幀不一致', 4: '看不到支撐或不連續'}


def one(tag, run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    ro = np.array((r.get('region') or {}).get('offset', [0.0, 0.0]), float)
    T0 = float(r['segtarget_link']['T0'])
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']

    def base(tt):
        k = int(np.argmin(np.abs(t - tt)))
        return dict(x_reg=round(float(c['base_x'][k] - ro[0]), 4), y_reg=round(float(c['base_y'][k] - ro[1]), 4), yaw_deg=round(float(np.degrees(c['base_yaw'][k])), 2))
    vh = r['vhigh_seq']['log']
    t_u = next(e[0] for e in vh if e[1] == 'U')

    def vh_at(tt):
        return next((p for (tq, p, _, _) in reversed(vh) if tq <= tt + 1e-9), 'TUCK')
    W = np.load(os.path.join(D, 'w1_log.npz'), allow_pickle=True)
    wc = {str(k): W['rows'][:, i] for i, k in enumerate(W['cols'])}
    L = [json.loads(l) for l in open(os.path.join(D, 'candidates.jsonl'))]
    rows = [x for x in L if x.get('kind') == 'row']
    cands = [x for x in L if x.get('kind') == 'cand']
    checks = [x['check'] for x in L if x.get('rec') == 'check']
    on = r['online']
    g = next(x for x in on['grasp_plans'] if x['state'] == 'accepted')
    t_acc = float(g['t_acc'])
    lo, hi = t_u, t_acc + 0.5
    # 事件
    tw = wc['task_t']
    exp = wc['expected'] > 0.5
    k_in = int(np.argmax(exp & (tw >= lo))) if (exp & (tw >= lo)).any() else None
    rt = np.array([x['row']['stamp'] - T0 for x in rows])
    fl = {f: np.array([x['row'].get(f) == 1.0 for x in rows]) for f in ('valid', 'xy_valid', 'support_observed', 'height_valid')}
    first = {f: (float(rt[np.argmax(fl[f] & (rt >= lo))]) if (fl[f] & (rt >= lo)).any() else None) for f in fl}
    ct = np.array([x['t_eval'] - T0 for x in cands])
    kr = np.array([x.get('k_ready') == 1.0 for x in cands])
    t_kready = float(ct[np.argmax(kr & (ct >= lo))]) if (kr & (ct >= lo)).any() else None
    ck = next((k for k in checks if int(k['seq']) == int(g['seq'])), None)
    ev = [('瓶入畫（真值投影，評估用）', None if k_in is None else float(tw[k_in])), ('分割選中（valid）', first['valid']), ('xy 有效', first['xy_valid']),
          ('支撐有效（support_observed）', first['support_observed']), ('高度有效（height_valid）', first['height_valid']),
          ('快照首幀（t_first）', None if ck is None else round(ck['t_first'] - T0, 3)), ('快照末幀（t_last）', None if ck is None else round(ck['t_last'] - T0, 3)),
          ('快照檢查完成（t_publish）＝首次合格目標／首次規劃請求', None if ck is None else round(ck['t_publish'] - T0, 3)), ('穩定候選就緒（k_ready）', t_kready),
          ('規劃承接', t_acc)]
    chain = [dict(event=n_, t=None if tt is None else round(tt, 3), vhigh=None if tt is None else vh_at(tt), **(base(tt) if tt is not None else {})) for n_, tt in ev]
    # 窗內比例與原因
    mw = (tw >= lo) & (tw <= hi)
    mr = (rt >= lo) & (rt <= hi)
    mc = (ct >= lo) & (ct <= hi)
    whys = {}
    for x in np.array(cands)[mc]:
        w = WHY.get(int(x.get('k_why', -1)), str(x.get('k_why')))
        whys[w] = whys.get(w, 0) + 1
    frac = dict(n_w1=int(mw.sum()), expected=round(float(exp[mw].mean()), 3) if mw.any() else None,
                n_rows=int(mr.sum()), **{f: round(float(fl[f][mr].mean()), 3) if mr.any() else None for f in fl}, n_cand=int(mc.sum()), k_why=whys)
    dist = dict(at_first_in=None if k_in is None else round(float(wc['dist'][k_in]), 3), at_accept=round(float(wc['dist'][int(np.argmin(np.abs(tw - t_acc)))]), 3))
    # 入畫前的分割情形：入畫判定為真但分割無效的時段
    gap = []
    if k_in is not None:
        for f in ('valid', 'xy_valid', 'support_observed'):
            if first[f] is not None:
                gap.append(dict(flag=f, delay_after_in_s=round(first[f] - float(tw[k_in]), 3)))
    margin = dict(x_acc=round(float(g['x_acc']), 4), x_latest=round(float(g['x_latest']), 4), plan_accept_margin_m=round(float(g['x_latest']) - float(g['x_acc']), 4),
                  note='計畫承接餘裕（x_latest 已含在途、煞停與 0.02 m 餘裕），不是碰撞安全距離')
    # 代表影格
    F = np.load(os.path.join(D, 'w1_frames.npz'), allow_pickle=True)
    fst = F['stamp'] - 0.0
    os.makedirs(FIG, exist_ok=True)
    reps = {}
    from PIL import Image
    for nm, tt in (('first_in', None if k_in is None else float(tw[k_in])), ('first_valid', first['valid']), ('first_support', first['support_observed']), ('k_ready', t_kready), ('accept', t_acc)):
        if tt is None:
            continue
        k = int(np.argmin(np.abs((fst - T0) - tt)))
        p = os.path.join(FIG, f'{tag}_{nm}_t{(fst[k] - T0):.2f}.png')
        Image.fromarray(np.asarray(F['rgb'][k], np.uint8)).save(p)
        reps[nm] = dict(event_t=round(tt, 3), frame_t=round(float(fst[k] - T0), 3), file=os.path.relpath(p, os.path.dirname(HERE)))
    return dict(run=run, region_offset=ro.tolist(), T0=T0, t_u=t_u, vhigh=vh, chain=chain, window=[round(lo, 3), round(hi, 3)], frac=frac, dist=dist, delays=gap, margin=margin, frames=reps,
                bottle_region=[round(float(c['bottle_base_x'][0] - ro[0]), 4), round(float(c['bottle_base_y'][0] - ro[1]), 4), round(float(c['bottle_base_z'][0]), 4)])


def main():
    out = {k: one(k, v) for k, v in RUNS.items()}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k, o in out.items():
        print('==', k, o['run'], '瓶(區域)', o['bottle_region'], 'U 開始', o['t_u'], '承接餘裕', o['margin']['plan_accept_margin_m'])
        for e in o['chain']:
            print('   ', e)
        print('   比例', o['frac'], '距離', o['dist'], '延遲', o['delays'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
