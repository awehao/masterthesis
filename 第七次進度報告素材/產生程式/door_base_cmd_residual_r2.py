#!/usr/bin/env python3
"""底盤表觀速度殘差 r2（只讀；Codex 20261025_020934 更正：統一時間語意）。取代 door_base_cmd_offset_phases.py 與
door_w1cd_pause_door_analysis.py 中的速度／命令比較（r1 檔保留）。

時間語意（isaac_door_k2_sim_ev4 主迴圈）：第 k 列的 base_xyth 與 base_cmd_body 都在 world.step 之前取得；本步寫入 API 的是
set_linear_velocity(R(yaw_k)·cmd_k)（世界座標根速度）⇒ 位移 pose[k+1]−pose[k] 由第 k 步命令產生。
同一物理區間 [k0, k1)：
  指令位移 D_cmd ＝ Σ_{k=k0}^{k1−1} R(yaw_k)·cmd_body_k·dt_k（dt_k＝sim_t[k+1]−sim_t[k]）；實際位移 D ＝ pose[k1]−pose[k0]；
  表觀速度殘差 ＝ (D − D_cmd)/T，轉到 k0 的本體座標。只稱「表觀速度殘差」，不假設加性模型或受力來源。
逐相位：每 0.1 s（10 步）一個區間取殘差，報中位數與 IQR；另報整段（單一區間）殘差。
接觸前判定：相位內每一步 leaf_contacts／wall_contacts 都讀得到（dict；None＝讀不到 ⇒ 不算已確認），且所有法向力值為 0（配對項存在但力為 0 仍算無接觸力）。
W1c／W1d 暫停各段：同法；另報 0.1 s 滑動窗口實際速度最大值（不再用整段淨位移代表全程速度；段長 < 0.1 s ⇒ None）。
輸出：results/door/base_cmd_residual_r2.json。

    python3 evaluation/door_base_cmd_residual_r2.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ('door_k3m_pause_p4b_1', 'door_wide_w1b_30_1', 'door_wide_w1c_30_1', 'door_wide_w1d_30_1')
OUT = os.path.join(HERE, 'results', 'door', 'base_cmd_residual_r2.json')


def rot_body(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [v[0] * c + v[1] * s, -v[0] * s + v[1] * c]


def interval(st, I, k0, k1):
    """[k0, k1) 列索引區間：回傳 (T, D_world, Dcmd_world, yaw0)。"""
    dcx = dcy = 0.0
    for k in range(k0, k1):
        yaw = st[k][I['base_xyth']][2]
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy = st[k][I['base_cmd_body']][:2]
        dt = st[k + 1][I['sim_t']] - st[k][I['sim_t']]
        dcx += (vx * c - vy * s) * dt
        dcy += (vx * s + vy * c) * dt
    p0, p1 = st[k0][I['base_xyth']], st[k1][I['base_xyth']]
    T = st[k1][I['sim_t']] - st[k0][I['sim_t']]
    return T, (p1[0] - p0[0], p1[1] - p0[1]), (dcx, dcy), p0[2]


def seg_stats(st, I, a, b):
    ks = [k for k, x in enumerate(st) if a - 1e-9 <= x[I['sim_t']] <= b + 1e-9 and k + 1 < len(st)]
    if len(ks) < 2:
        return None
    k0, k1 = ks[0], ks[-1]
    T, D, Dc, y0 = interval(st, I, k0, k1)
    whole = rot_body(((D[0] - Dc[0]) / T * 1e3, (D[1] - Dc[1]) / T * 1e3), y0)
    win, vmax = [], None
    for j in range(k0, k1 - 9, 10):
        t_, d_, dc_, y_ = interval(st, I, j, j + 10)
        win.append(rot_body(((d_[0] - dc_[0]) / t_ * 1e3, (d_[1] - dc_[1]) / t_ * 1e3), y_))
    for j in range(k0, k1 - 9):                                               # 0.1 s 滑動窗口的實際速度最大值；段長 < 0.1 s ⇒ None
        p0, p1 = st[j][I['base_xyth']], st[j + 10][I['base_xyth']]
        v_ = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (st[j + 10][I['sim_t']] - st[j][I['sim_t']]) * 1e3
        vmax = v_ if vmax is None else max(vmax, v_)
    win = np.array(win) if win else np.zeros((0, 2))
    lc = [st[k][I['leaf_contacts']] for k in range(k0, k1 + 1)]
    wc = [st[k][I['wall_contacts']] for k in range(k0, k1 + 1)]
    readable = all(isinstance(x, dict) for x in lc) and all(isinstance(x, dict) for x in wc)
    fmax = max([0.0] + [float(v) for x in lc if isinstance(x, dict) for d in x.values() for v in d.values()]
               + [float(v) for x in wc if isinstance(x, dict) for v in x.values()])
    contact_free = readable and fmax == 0.0
    return {'t': [st[k0][I['sim_t']], st[k1][I['sim_t']]], 'n_steps': k1 - k0,
            'residual_whole_body_mm_s': whole, 'residual_whole_norm': math.hypot(*whole),
            'residual_0p1s_median': None if not len(win) else np.median(win, 0).tolist(),
            'residual_0p1s_iqr_norm': None if not len(win) else float(np.hypot(*(np.percentile(win, 75, 0) - np.percentile(win, 25, 0)))),
            'net_disp_mm': math.hypot(*D) * 1e3, 'net_speed_mm_s': math.hypot(*D) / T * 1e3, 'max_speed_0p1s_mm_s': vmax,
            'cmd_disp_body_mm': rot_body((Dc[0] * 1e3, Dc[1] * 1e3), y0), 'contact_free_confirmed': contact_free, 'contact_max_n': fmax, 'contact_readable': readable}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res = {'schema': 'base_cmd_residual/2', 'method': __doc__.split('輸出')[0].strip(), 'phases': {}, 'pause_segments': {}}
    for R in RUNS:
        run = os.path.join(HERE, 'runs', R)
        r = json.load(open(os.path.join(run, 'room_run.json')))
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        st = r['steps']
        t = json.load(open(os.path.join(run, 'task.json')))
        ph = [(e['to'], e['t']) for e in t['policy_events'] if 'to' in e]
        t_end = next((e['t'] for e in t['policy_events'] if e.get('abort') or e.get('to') == 'DONE'), st[-1][I['sim_t']])
        out = []
        for k, (nm, a) in enumerate(ph):
            b = ph[k + 1][1] if k + 1 < len(ph) else t_end
            sgs = seg_stats(st, I, a, b)
            if sgs and sgs['n_steps'] >= 20:
                out.append(dict(phase=nm, **sgs))
        res['phases'][R] = out
        print(R)
        for o in out:
            print(f"  {o['phase']:10s} {o['t'][0]:6.2f}-{o['t'][1]:6.2f} 殘差(整段) {np.round(o['residual_whole_body_mm_s'], 2)} |{o['residual_whole_norm']:.2f}| "
                  f"0.1s 中位 {np.round(o['residual_0p1s_median'], 2)} IQR {o['residual_0p1s_iqr_norm']:.2f}  無接觸已確認 {o['contact_free_confirmed']}（最大 {o['contact_max_n']:.2f} N）")
        if R in ('door_wide_w1c_30_1', 'door_wide_w1d_30_1'):
            s = json.load(open(os.path.join(run, 'align_solver.json')))
            tr = s['stats']['k3m_pause']['supervisor_outcome']['transitions']
            pe = dict(ph)
            segs = [('SETTLE', pe['SETTLE'], pe['PAUSE']), ('PAUSE→ZERO_PUB', pe['PAUSE'], tr['ZERO_PUB'])]
            if R.endswith('w1d_30_1'):
                seq1 = s['stats']['pause_reanchor']['record']['first_published']['source_seq']
                ta_ = next(float(x[I['sim_t']]) for x in st if x[I['applied_src']][2] >= seq1)
                segs += [('ZERO_PUB→新命令套用', tr['ZERO_PUB'], ta_), ('套用→FAIL', ta_, tr['FAIL']), ('FAIL→任務中止', tr['FAIL'], t_end)]
            else:
                segs += [('ZERO_PUB→STOPPED', tr['ZERO_PUB'], tr['STOPPED']), ('STOPPED→中止', tr['STOPPED'], t_end)]
            res['pause_segments'][R] = [dict(seg=n, **seg_stats(st, I, a, b)) for n, a, b in segs]
            for o in res['pause_segments'][R]:
                print(f"    {o['seg']:18s} 殘差(整段) {np.round(o['residual_whole_body_mm_s'], 2)} 淨速 {o['net_speed_mm_s']:.3f} 0.1s 滑動最大 {o['max_speed_0p1s_mm_s']} mm/s")
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
