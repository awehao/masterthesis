#!/usr/bin/env python3
"""KF 時鐘候選追蹤器的回放核對（Codex 20261101_135333）：原追蹤器 vs 候選，同輸入。非物理、同程序（無節點傳輸延遲）。
0 合成：等速 0.45 m/s 移動物，軌跡誕生後漏 0／3 拍再關聯；第 3 次更新時的速度估計（原：高估；候選：接近真值）。
1 真正移動物（速度估計與發布延遲）：實際掃描紀錄 bottle_w1_dyn1sr_np1、dyn1sr_g1、dyn2sr_np1；重建掃描 bottle_w1_dyn2sr_g1（該趟無掃描紀錄）。
  對讀回真值：速度誤差 p50／p95／最大、> 0.5 m/s 筆數、> 0.8 m/s 的發布；首次發布時刻（對案例 t_start）。
2 未知靜態物：bottle_w1c_1（漏帶第二桌地圖，離線射線，bottle_tracker_replay）假動態掃描數與最高速；dyn1sr_g1 的未入圖方塊 OBST 假移動；名目 sr_nominal1 假動態。
通過條件：候選在 1 不增加 > 0.5 m/s 誤差筆數、消除 dyn2sr_g1 22.4 s 的 1.5 m/s 發布、首次發布不晚於原版 0.2 s 以上；在 2 不增加假動態。
    python3 evaluation/test_bottle_tracker_kfclock.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_tracker_kfclock as KC                     # noqa: E402
import bottle_tracker_replay as BTR                     # noqa: E402
import bottle_tracker_scanlog_replay as TR              # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402
import test_bottle_place as TP                          # noqa: E402
from ammr_wholebody_mpc.kalman_tracker import KalmanTracker2D   # noqa: E402

ORIG = NC.ScanObstacleTracker
CAND = KC.make(ORIG)
BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def with_cls(cls, fn, *a, **k):
    NC.ScanObstacleTracker = cls
    try:
        return fn(*a, **k)
    finally:
        NC.ScanObstacleTracker = ORIG


def case_synth():
    out = {}
    for miss in (0, 3):
        res = {}
        for nm, fix in (('orig', False), ('cand', True)):
            v, dt = 0.45, 0.1
            ts = [0.0] + [dt * (k + 1 + miss) for k in range(4)]
            kf = KalmanTracker2D(init_xy=(0.0, 0.0), sigma_pos=0.01, sigma_vel=0.40, sigma_meas=0.05, init_vel_var=1.0)
            if fix:
                kf._last_t_ns = 0
            for t in ts[1:3]:
                kf.step(t_ns=int(round(t * 1e9)), y_xy=(v * t, 0.0))
            res[nm] = float(np.hypot(*kf.velocity))
        out[miss] = res
        print(f'   合成：誕生後漏 {miss} 拍，第 3 次更新速度 原 {res["orig"]:.3f}／候選 {res["cand"]:.3f} m/s（真值 0.45）')
    chk('0 合成：候選第 3 次更新速度誤差 < 原版', all(abs(r['cand'] - 0.45) < abs(r['orig'] - 0.45) for r in out.values()), out)
    return out


def recon_rows(run, case):
    """dyn2sr_g1 無掃描紀錄：以紀錄位姿＋場景盒＋移動物讀回位置重建 10 Hz 掃描（22.6 s 曾以此重現 Shield 改寫）。"""
    CA = DC.CASES[case]
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    bx = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']
    ch = NC.NavChain(map_boxes=TR.map_boxes())
    rows, nxt = [], 0.0
    for i, t in enumerate(c['t']):
        if t + 1e-9 < nxt:
            continue
        nxt = t + 0.1
        q = (float(c['base_x'][i]), float(c['base_y'][i]), float(c['base_yaw'][i]))
        r = NC.ray_boxes(np.array(q[:2]), q[2], bx, G.LIDAR_Z)
        if np.isfinite(c['dyn_x'][i]):
            r = np.minimum(r, DC.ray_cyl(np.array(q[:2]), NC.scan_angles() + q[2], np.array([c['dyn_x'][i], c['dyn_y'][i]]), CA['radius'], NC.RANGE_MAX))
        ch.on_scan(float(t), r, q)
        truth = dict(x=float(c['dyn_x'][i]), y=float(c['dyn_y'][i]), vx=float(c['dyn_vx'][i]), vy=float(c['dyn_vy'][i])) if np.isfinite(c['dyn_x'][i]) else None
        rows.append(dict(t=float(t), dyn=[dict(d) for d in ch.dyn], truth=truth))
    return rows


def stats(rows, CA):
    s = TR.summarize(rows, CA['radius'])
    E = np.array([e[3] for e in s['speed_err']]) if s['speed_err'] else np.zeros(1)
    fast = [(round(r['t'], 2), round(float(np.hypot(d['vx'], d['vy'])), 3)) for r in rows for d in r['dyn'] if np.hypot(d['vx'], d['vy']) > 0.8]
    t_move = [e[0] for e in s['speed_err'] if e[0] >= CA['t_start'] - 1e-9]
    return dict(n_pub=len(s['speed_err']), err_p50=float(np.median(E)), err_p95=float(np.percentile(E, 95)), err_max=float(E.max()), n_gt_0p5=int((E > 0.5).sum()),
                fast=fast, first_pub_after_start=(min(t_move) - CA['t_start']) if t_move else None, n_false=s['n_false'], false_fast=s['false_fast'])


def case_movers():
    out = {}
    for run, case, rec in (('bottle_w1_dyn1sr_np1', 'w1_dyn_pass1', True), ('bottle_w1_dyn1sr_g1', 'w1_dyn_pass1', True), ('bottle_w1_dyn2sr_np1', 'w1_dyn_overtake1', True),
                           ('bottle_w1_dyn2sr_g1', 'w1_dyn_overtake1', False)):
        CA = DC.CASES[case]
        o = {}
        for nm, cls in (('orig', ORIG), ('cand', CAND)):
            rows = with_cls(cls, TR.replay, run) if rec else with_cls(cls, recon_rows, run, case)
            o[nm] = stats(rows, CA)
        out[run] = o
        a, b = o['orig'], o['cand']
        print(f"   {run}（{'實際掃描' if rec else '重建掃描'}）：速度誤差 p50/p95/max 原 {a['err_p50']:.3f}/{a['err_p95']:.3f}/{a['err_max']:.3f}、候選 {b['err_p50']:.3f}/{b['err_p95']:.3f}/{b['err_max']:.3f}；"
              f">0.5 m/s 原 {a['n_gt_0p5']}／候選 {b['n_gt_0p5']}；>0.8 m/s 發布 原 {a['fast'][:4]}／候選 {b['fast'][:4]}；首次發布（起步後）原 {a['first_pub_after_start']}／候選 {b['first_pub_after_start']} s；"
              f"假動態 原 {a['n_false']}／候選 {b['n_false']}")
        chk(f'1 {run}：候選 >0.5 m/s 誤差筆數不增、首次發布不晚 0.2 s 以上', b['n_gt_0p5'] <= a['n_gt_0p5'] and (a['first_pub_after_start'] is None or (b['first_pub_after_start'] is not None and b['first_pub_after_start'] <= a['first_pub_after_start'] + 0.2 + 1e-9)),
            (a['n_gt_0p5'], b['n_gt_0p5'], a['first_pub_after_start'], b['first_pub_after_start']))
    g = out['bottle_w1_dyn2sr_g1']
    chk('1 dyn2sr_g1：候選消除 22.4 s 附近 > 0.8 m/s 的發布', not any(21.9 <= t <= 22.8 for t, _ in g['cand']['fast']) and any(21.9 <= t <= 22.8 for t, _ in g['orig']['fast']), (g['orig']['fast'], g['cand']['fast']))
    return out


def case_statics():
    out = {}
    legs1 = [(b[1], b[2]) for b in G.table_boxes() if b[0] != 'table_top']
    legs2 = [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']
    scene = legs1 + [(SC.OBST['c'], SC.OBST['sz'])] + legs2
    for nm, cls in (('orig', ORIG), ('cand', CAND)):
        R = with_cls(cls, BTR.replay, 'bottle_w1c_1', legs1, scene, t_max=None)
        ev = [(t, d) for t, q, d, _, _ in R if d]
        vmax = max((float(np.hypot(x['vx'], x['vy'])) for _, d in ev for x in d), default=0.0)
        out[f'w1c_1_{nm}'] = dict(n_dyn_scans=len(ev), v_max=vmax, first=ev[0][0] if ev else None)
    for nm, cls in (('orig', ORIG), ('cand', CAND)):
        rows = with_cls(cls, TR.replay, 'bottle_w1_dyn1sr_g1')
        ob = [(round(r['t'], 2), round(float(np.hypot(d['vx'], d['vy'])), 3)) for r in rows for d in r['dyn'] if d['x'] < -1.5 and abs(d['y']) < 0.4]
        out[f'obst_dyn1sr_g1_{nm}'] = dict(n=len(ob), v_max=max((v for _, v in ob), default=0.0), sample=ob[:4])
        rows = with_cls(cls, TR.replay, 'bottle_w1_sr_nominal1')
        out[f'nominal_{nm}'] = dict(n_dyn_scans=sum(1 for r in rows if r['dyn']))
    print(f"   bottle_w1c_1（漏第二桌地圖、離線射線）：假動態掃描 原 {out['w1c_1_orig']['n_dyn_scans']}（最高 {out['w1c_1_orig']['v_max']:.2f} m/s）／候選 {out['w1c_1_cand']['n_dyn_scans']}（最高 {out['w1c_1_cand']['v_max']:.2f} m/s）")
    print(f"   dyn1sr_g1 未入圖方塊 OBST 假移動：原 {out['obst_dyn1sr_g1_orig']}／候選 {out['obst_dyn1sr_g1_cand']}")
    print(f"   名目 sr_nominal1 假動態掃描：原 {out['nominal_orig']['n_dyn_scans']}／候選 {out['nominal_cand']['n_dyn_scans']}")
    chk('2 未知靜態物：候選假動態掃描數不增（w1c_1、OBST、名目）', out['w1c_1_cand']['n_dyn_scans'] <= out['w1c_1_orig']['n_dyn_scans'] and out['obst_dyn1sr_g1_cand']['n'] <= out['obst_dyn1sr_g1_orig']['n']
        and out['nominal_cand']['n_dyn_scans'] <= out['nominal_orig']['n_dyn_scans'], out)
    return out


def main():
    res = dict(synth=case_synth(), movers=case_movers(), statics=case_statics())
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'tracker_kfclock_replay.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
