#!/usr/bin/env python3
"""v3：v2 的版本化複本，修 v2 全序列核出的真違反：C0 末點精確落在 P0、C1 初速差分以 V_HAND 起算（導航參考線是空間取樣，非時間序列）；C2 反轉列＋抓握位姿以 quintic 重時間化（起停到零、收到抓握位姿）；C6／C7 改 quintic。
v2（Codex 20261030 m6d）：v1 的版本化複本，只改：導航末段以 C1 進場方向直線接 P0（共用方向，無偏航差）；C1 改三次時間律自滾動速度 V_HAND 減到 0；速度／加速度差分跨階段連續（交界計入）；ok 納入全部運動限值；偏航參考上限 0.0738 另核。
v1：回程關門有界候選 v1（離線、無力學；Howard 10/30 定案：去程不變，回程推開穿過後再關門）。接 door_return_rev_v2c 的 S8 終點。
參考＝mission_dd_7 門段實測列（room_run.json）的時間反轉（同趟、10 列取 1＝20 Hz）；抓放獨立於反轉：
  C0_nav      S8 終點→繞門尖外側折線到 P0 (0.15,−1.50)（導航參考線：只核靜態淨空與離門扇距離，朝向轉折不算閉合率；實際由 GMPC 決定）
  C1_arc      **移動轉彎段**（取代去程 R1 反向的近原地轉）：P0 起三次 Bézier＋quintic 時間律，偏航 105°→RG4 起點偏航，手臂自 STOW 依 φ∈[0.3,1] 展到 RG4 起點姿態；前進、轉向、展臂重疊
  C2_rev_rel  去程 RG3 退離反向（張爪接近外側把手；門 60°）
  C3_close    固定位姿閉爪（1 s）
  C4_rev_open 去程 OPEN 反向：推關，門 q_open→--q-end（預設 2°），底盤向門靠近
  C5_open     固定位姿張爪（1 s）
  C6_retreat  沿工具 −z 退 0.06 m（IK 追蹤）
  C7_stow     手臂內插回 STOW_F
監看同 door_return_rev_v2：GeometryMonitor＋SwitchAwareGeom＋backend_v3(full)＋登錄 v_floor／ds_err；目標橫桿 C2–C6＝out、其餘 None。
只核近關閉 --q-end，不核 0°、不宣稱上閂。實測列含感測雜訊與真實閉迴路殘差，反向後的閉合率會計入預測。
界線：推關受力、把手受壓、門動力學與追蹤未驗證；C0 為參考線。C3–C5 閉爪接觸沿 release_outer_c 規則另核 bar_out ≥ −1 mm（同 bar_in 指定接觸容差）。
    python3 -u evaluation/door_return_close_v1.py [--q-end 2.0]   # 寫 results/door/door_return_close_v1.json（不覆寫）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402

import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402

RUN = os.path.join(HERE, 'runs', 'mission_dd_7')
GCFG = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')
PREV = os.path.join(HERE, 'results', 'door', 'door_return_rev_v2c.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_return_close_v3.json')
OUT_B = os.path.join(HERE, 'results', 'door', 'door_return_close_v3b.json')   # --c2-stretch > 1：C2 只放慢時間軸，路徑不變
V_HAND, W_REF, A_LIN, A_ANG = 0.03, 0.0738, 0.50, 2.00
STOW_F = np.array([0.0, 0.0, 0.15, 0.0, -1.10, 0.0])
V_BOX, W_BOX, ARM_RATE, A_ARM = 0.035255, 0.1999, 1.0, 19.984
DT, AGE = 0.05, 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q-end', type=float, default=2.0)
    ap.add_argument('--c2-stretch', type=float, default=1.0)
    ap.add_argument('--export', default=None, help='只匯出參考列到此 JSON')
    a = ap.parse_args()
    out_path = OUT_B if a.c2_stretch > 1.0 else OUT
    if os.path.exists(out_path) and not a.export:
        raise SystemExit(f'{out_path} 已存在，不覆寫')
    t00 = time.time()
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    t_ = json.load(open(os.path.join(RUN, 'task.json')))
    c = r['steps_cols']
    s = r['steps']
    it, ib, iq, id_, iow = [c.index(k) for k in ('sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'owner(0=nav,1=wb,2=glide)')]
    T = np.array([x[it] for x in s])
    Xm = np.array([list(x[ib]) + list(x[iq]) for x in s], float)
    Qm = np.degrees(np.array([x[id_] for x in s], float))
    Own = np.array([x[iow] for x in s])
    ev = {}
    for e in t_['policy_events']:
        if 'to' in e:
            ev.setdefault(e['to'], e['t'])
    t_open, t_settle, t_rg2, t_rg3, t_rg4 = ev['OPEN'], ev['SETTLE'], ev['RG2'], ev['RG3'], ev['RG4']
    i_wb2nav = int(np.argmax((T > t_rg4) & (Own == 0)))              # 去程 R1 交導航
    i_nav2wb = int(np.argmax((T > T[i_wb2nav]) & (Own != 0)))        # 去程 R2 交回全身
    idx = lambda t0, t1: np.where((T >= t0) & (T <= t1))[0][::5]      # noqa: E731  100 Hz → 20 Hz
    plan = []

    def add(name, rule, tgt, x, q, qfi):
        plan.append({'name': name, 'rule': rule, 'target': tgt, 'x': np.asarray(x, float), 'q': float(q), 'qfi': float(qfi)})
    prev = json.load(open(PREV))
    b_s8 = np.asarray(prev['exit']['end_x'], float)
    q60 = float(Qm[np.searchsorted(T, t_rg2)])
    P0 = np.array([0.15, -1.50])
    y0 = math.radians(105.0)
    P_in = P0 - 0.40 * np.array([math.cos(y0), math.sin(y0)])  # 導航末段 0.40 m 直線＝C1 進場方向
    wp = [b_s8[:2], np.array([0.90, -1.55]), np.array([0.55, -1.86]), P_in, P0]
    for p0, p1 in zip(wp[:-1], wp[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(p1 - p0)) / 0.02)))
        h = math.atan2(*(p1 - p0)[::-1])
        for k in range(n):
            add('C0_nav', 'R_stow', None, np.r_[p0 + (p1 - p0) * k / n, h, STOW_F], q60, PI.OPEN)
    add('C0_nav', 'R_stow', None, np.r_[P0, y0, STOW_F], q60, PI.OPEN)
    i_rg4 = int(np.searchsorted(T, t_rg4))
    xT = Xm[i_rg4]
    dy = math.atan2(math.sin(xT[2] - y0), math.cos(xT[2] - y0))
    L0 = float(np.linalg.norm(xT[:2] - P0))
    e0, e1 = np.array([math.cos(y0), math.sin(y0)]), np.array([math.cos(xT[2]), math.sin(xT[2])])
    c1, c2 = P0 + e0 * L0 / 3, xT[:2] - e1 * L0 / 3
    us = np.linspace(0, 1, 401)
    bez = np.array([(1 - u) ** 3 * P0 + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * xT[:2] for u in us])
    sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(bez, axis=0).T))]
    T_arc = max(2.0 * sl[-1] / (0.9 * V_BOX), 1.875 * abs(dy) / W_REF, 2.5 * float(np.max(np.abs(xT[3:] - STOW_F))) / (0.7 * 0.5))
    n_arc = int(math.ceil(T_arc / DT))
    for k in range(1, n_arc + 1):
        tau = k / n_arc
        f = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        s_ = sl[-1] * (3 * tau ** 2 - 2 * tau ** 3) + V_HAND * n_arc * DT * (tau - 2 * tau ** 2 + tau ** 3)   # 三次：起速 V_HAND、末速 0
        w = float(np.clip((k / n_arc - 0.3) / 0.7, 0, 1))
        w = 10 * w ** 3 - 15 * w ** 4 + 6 * w ** 5
        add('C1_arc', 'R_stow', None, np.r_[np.interp(s_, sl, bez[:, 0]), np.interp(s_, sl, bez[:, 1]), y0 + dy * f, STOW_F + (xT[3:] - STOW_F) * w], q60, PI.OPEN)
    i_g = int(np.searchsorted(T, t_settle))
    _r2 = [Xm[i] for i in idx(t_rg3, t_rg4)[::-1]] + [Xm[i_g]]
    _q2 = [Qm[i] for i in idx(t_rg3, t_rg4)[::-1]] + [Qm[i_g]]
    _sa = np.r_[0.0, np.cumsum([np.linalg.norm(np.r_[_r2[j + 1][:2] - _r2[j][:2], (_r2[j + 1][3:] - _r2[j][3:]) * 0.2]) for j in range(len(_r2) - 1)])]
    _L2 = float(np.hypot(*(_r2[-1][:2] - _r2[0][:2])))
    n2 = int(math.ceil(1.875 * max(float(np.sum(np.hypot(*np.diff(np.array(_r2)[:, :2], axis=0).T))), _L2) / (0.9 * V_BOX) / DT * a.c2_stretch))
    _R2 = np.array(_r2)
    for k in range(1, n2 + 1):
        u = (10 * (k / n2) ** 3 - 15 * (k / n2) ** 4 + 6 * (k / n2) ** 5) * _sa[-1]
        add('C2_rev_rel', 'release_outer', 'out', [float(np.interp(u, _sa, _R2[:, j])) for j in range(9)], float(np.interp(u, _sa, _q2)), PI.OPEN)
    for k in range(1, 21):
        add('C3_close', 'release_outer_c', 'out', Xm[i_g], Qm[i_g], PI.OPEN + (PI.CLOSED - PI.OPEN) * k / 20)
    rows = idx(t_open, t_settle)[::-1]
    rows = [i for i in rows if Qm[i] >= a.q_end]
    for i in rows:
        add('C4_rev_open', 'release_outer_c', 'out', Xm[i], Qm[i], PI.CLOSED)
    xe, qe = Xm[rows[-1]], float(Qm[rows[-1]])
    for k in range(1, 21):
        add('C5_open', 'release_outer_c', 'out', xe, qe, PI.CLOSED + (PI.OPEN - PI.CLOSED) * k / 20)
    Tg = PI.C.K.fk(xe, 'link_tcp')
    rows_r = []
    xe2, why = PI.track(xe, [PI.tz(Tg, -0.06 * (10 * ((k + 1) / 72) ** 3 - 15 * ((k + 1) / 72) ** 4 + 6 * ((k + 1) / 72) ** 5)) for k in range(72)], lambda k: qe, PI.OPEN, 'release_outer', rows_r, None)
    block0 = None if not why else {'stage': 'C6_retreat', 'kind': 'track', 'why': why}
    for rr in rows_r:
        add('C6_retreat', 'release_outer', 'out', rr['x'], qe, PI.OPEN)
    x_last = np.asarray(rows_r[-1]['x'], float) if rows_r else xe
    n7 = int(math.ceil(1.875 * float(np.max(np.abs(x_last[3:] - STOW_F))) / (0.45 * DT)))
    for k in range(1, n7 + 1):
        x = x_last.copy()
        x[3:] = x_last[3:] + (STOW_F - x_last[3:]) * (10 * (k / n7) ** 3 - 15 * (k / n7) ** 4 + 6 * (k / n7) ** 5)
        add('C7_stow', 'R_stow', None, x, qe, PI.OPEN)

    if a.export:                                                                        # 匯出 20 Hz 列（不核對、不寫結果）
        json.dump([{'stage': r['name'], 'rule': r['rule'], 'target': r['target'], 'x': [float(v) for v in r['x']], 'q_deg': r['q'], 'qfi': r['qfi']} for r in plan], open(a.export, 'w'))
        print(f'匯出 {len(plan)} 列 → {a.export}')
        return
    g = json.load(open(GCFG))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    res, block, tt, prevx, pv = {}, block0, 0.0, None, None
    junc, seen = [], set()
    for k, p in enumerate(plan):
        n, x = p['name'], p['x']
        R = res.setdefault(n, {'n': 0, 'rule': p['rule'], 'target': p['target'], 'n_check_fail': 0, 'first_check_fail': None, 'n_stop': 0, 'first_stop': None, 'n_warn': 0,
                               'd_pred_min': {}, 'jm_min': 9.0, 'v_max': 0.0, 'w_max': 0.0, 'arm_rate_max': 0.0, 'arm_acc_max': 0.0})
        if n not in seen:
            seen.add(n)
            _new = False                                                                # v2：交界計入差分
            if prevx is not None:
                dx = x - prevx
                junc.append({'into': n, 'dpos_mm': float(np.hypot(dx[0], dx[1])) * 1e3, 'dyaw_deg': abs(math.degrees(math.atan2(math.sin(dx[2]), math.cos(dx[2])))), 'darm_rad': float(np.max(np.abs(dx[3:])))})
        else:
            _new = False
        R['n'] += 1
        tt += DT
        if n == 'C1_arc' and R['n'] == 1:                                               # 導航參考線是空間取樣 ⇒ 初速以 V_HAND 起算
            pv = np.r_[V_HAND * math.cos(y0), V_HAND * math.sin(y0), np.zeros(7)]
        if prevx is not None and n != 'C0_nav' and not _new:
            dv = (x - prevx) / DT
            dv[2] = math.atan2(math.sin(x[2] - prevx[2]), math.cos(x[2] - prevx[2])) / DT
            R['v_max'] = max(R['v_max'], float(np.hypot(dv[0], dv[1])))
            R['w_max'] = max(R['w_max'], abs(float(dv[2])))
            R['arm_rate_max'] = max(R['arm_rate_max'], float(np.max(np.abs(dv[3:]))))
            if pv is not None:
                R['a_lin_max'] = max(R.get('a_lin_max', 0.0), float(np.hypot(*((dv[:2] - pv[:2]) / DT))))
                R['a_ang_max'] = max(R.get('a_ang_max', 0.0), abs(float((dv[2] - pv[2]) / DT)))
                R['arm_acc_max'] = max(R['arm_acc_max'], float(np.max(np.abs((dv[3:] - pv[3:]) / DT))))
            pv = dv
        prevx = x.copy()
        ok, why, info = PI.check(x, p['q'], p['qfi'], p['rule'])
        if not ok:
            R['n_check_fail'] += 1
            R['first_check_fail'] = R['first_check_fail'] or {'k': k, 'why': why, 'q': p['q'], 'x': np.round(x, 3).tolist()}
            block = block or {'stage': n, 'kind': 'check', 'why': why, 'q': p['q'], 'x': np.round(x, 3).tolist()}
        if info is not None and 'jm' in info:
            R['jm_min'] = min(R['jm_min'], float(info['jm']))
        if info is not None and p['rule'] == 'release_outer_c':
            _bo = min((dd.get('bar_out', 9.0) for dd in info.get('fingers', {}).values()), default=9.0)
            R['bar_out_min_mm'] = min(R.get('bar_out_min_mm', 9e3), _bo * 1e3)
            if _bo < -0.001:
                R['n_check_fail'] += 1
                block = block or {'stage': n, 'kind': 'bar_out', 'why': f'穿入 {_bo * 1e3:.2f} mm', 'q': p['q']}
        if n == 'C0_nav':                                                               # 導航參考線：朝向轉折非實際運動，只核靜態
            R['leaf_min_m'] = min(R.get('leaf_min_m', 9.0), float(np.min(np.linalg.norm(x[:2] - np.outer(np.linspace(0, 1, 41), [math.cos(-math.radians(p['q'])), math.sin(-math.radians(p['q']))]), axis=1))))
            continue
        dec, det = mon.step(tt, x, math.radians(p['q']), p['qfi'], AGE, p['target'])
        if dec == 'stop':
            R['n_stop'] += 1
            bad = {cl: v['d_pred'] for cl, v in (det.get('classes') or {}).items() if v['d_pred'] < cfg.d_min[cl]}
            R['first_stop'] = R['first_stop'] or {'k': k, 'why': det.get('why'), 'q': p['q'], 'bad': bad, 'who': {cl: det['classes'][cl]['who'] for cl in bad}}
            block = block or {'stage': n, 'kind': 'monitor', 'why': det.get('why'), 'q': p['q'], 'bad': bad}
        elif dec == 'warn':
            R['n_warn'] += 1
        for cl, v in (det.get('classes') or {}).items():
            R['d_pred_min'][cl] = min(R['d_pred_min'].get(cl, 9.0), float(v['d_pred']))
    x_end = plan[-1]['x']
    viol = {n: [k for k, lim in (('v_max', V_BOX), ('w_max', W_BOX), ('arm_rate_max', ARM_RATE), ('arm_acc_max', A_ARM), ('a_lin_max', A_LIN), ('a_ang_max', A_ANG)) if R.get(k, 0.0) > lim * (1 + 1e-9)]
            for n, R in res.items() if n != 'C0_nav'}
    viol = {k: v for k, v in viol.items() if v}
    if res['C1_arc']['w_max'] > W_REF:
        viol.setdefault('C1_arc', []).append('w_ref')
    out = {'schema': 'door_return_close_v3', 'c2_stretch': a.c2_stretch, 'ok': block is None and not viol, 'limit_violations': viol, 'first_block': block, 'q_end_request_deg': a.q_end, 'q_end_deg': qe, 'q_open_deg': q60,
           'stages': res, 'junctions': junc, 'total_T_s': tt, 'n_rows': len(plan), 'end_base': np.round(x_end[:3], 4).tolist(),
           'source': {'run': 'mission_dd_7', 't_OPEN': t_open, 't_SETTLE': t_settle, 't_RG3': t_rg3, 't_RG4': t_rg4, 't_wb2nav': float(T[i_wb2nav]), 't_nav2wb': float(T[i_nav2wb])},
           'limits': {'v_box': V_BOX, 'w_box': W_BOX, 'arm_rate': ARM_RATE, 'a_arm': A_ARM}, 'wall_s': time.time() - t00,
           'not_modeled': ['推關受力與把手受壓', '門動力學', '追蹤與延遲', 'C0 實際由 GMPC 決定', '房內家具'], 'note_C1': '沿用近原地轉向暫時段；不符 Howard R1 平順要求'}
    json.dump(out, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"ok {out['ok']}；限值違反 {viol}；first_block {block}；{len(plan)} 列 {tt:.1f} s；門 {q60:.2f}→{qe:.2f}°；終點底盤 {out['end_base']}；牆鐘 {out['wall_s']:.0f} s")
    for n, R in res.items():
        print(f"{n:12s} {R['rule'][:13]:13s} {str(R['target']):4s} n {R['n']:4d} chkF {R['n_check_fail']:3d} stop {R['n_stop']:3d} warn {R['n_warn']:3d} pred {[(k_[:2], round(v * 1e3, 1)) for k_, v in R['d_pred_min'].items()]} v {R['v_max']:.4f} w {R['w_max']:.3f} arm {R['arm_rate_max']:.3f} acc {R['arm_acc_max']:.2f} a {R.get('a_lin_max', 0):.3f}/{R.get('a_ang_max', 0):.3f}")
    for n in res:
        if res[n]['first_stop'] or res[n]['first_check_fail']:
            print(n, 'first_stop', res[n]['first_stop'], 'first_check', res[n]['first_check_fail'])
    print('junctions', [(j['into'], round(j['dpos_mm'], 1), round(j['dyaw_deg'], 1), round(j['darm_rad'], 3)) for j in junc])


if __name__ == '__main__':
    main()
