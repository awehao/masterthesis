#!/usr/bin/env python3
"""連續協同通行：聯合搜尋候選的同一條完整軌跡核驗 v1（Codex 20261026_213833 §3、§6-3；離線靜態，無 Isaac）。

輸入：door_coopen_joint_v2.json 的 full[*]（握回 0° 者；可用參數指定索引，預設全部）。同一條軌跡依序為：
  A 持握開門 0 → q_rel（逐 1°，候選軌跡本身）；
  B 門靜止於 q_rel、仍握持，底盤沿出口路徑前進 fwd（每 1 cm，IK 重解）；
  C 張爪（3 點）＋沿工具 −z 退 60 mm（每 1 cm），底盤續沿出口路徑前進；
  D 繞回門扇自由端（沿 u 0.30 m、沿 −n 0.20 m，每 2 mm），底盤偏航線性 +DYAW（預設 +90°）並續前進；
  E 關節內插到 STOW（40 點），底盤續前進；F 底盤沿出口路徑到 y ≥ 0.50（交還點），手臂 STOW。
底盤位置一律以出口路徑弧長**連續線性插值**（回應 v3 以最近 1 cm 路徑點取底盤的指正）。
核驗（全身，非只底盤）：
  1. 每點非手指／手指／自碰 ≥ 5 mm、底盤 ≥ 20 mm（門檻），另報扣 10 mm 底盤誤差後的底盤餘裕；
  2. 門角：A、B 用當下門角；C–F 用 q_rel + δ，δ ∈ {−2, −1, −0.5, 0, 0.25, 0.5, 1, 2, 5}°，報各段自 0 連續可容許範圍；
  3. 距離下界：相鄰取樣點底盤位移 Δb、連桿最大位移 Δl（FK 點雲最大位移）、門扇在 Δq 內的最大位移（0.95 m × Δq），
     下界＝min(兩點距離) − max(Δb, Δl, 門位移)/2（Lipschitz 1，保守）；
  4. 時間化（E2：合速度 0.05、每軸 0.035 m/s、0.20 rad/s；手臂 1.0 rad/s；門 1 °/s；TCP 0.05 m/s）：逐段時間、總時長，
     放手前後底盤速度向量（B 末 vs C 初、A 末 vs B 初）與所需加速度（≤ 0.5 m/s² 計時）；
  5. 開門期間出口進展：A 段逐度真實測地距離 D_q_rel（不可通行記 None）與 D 的總變化、末 10° 變化。
報：每段通過／失敗、最小距離與下界、餘裕（距離與餘裕分欄）、δ 容許範圍、時間、速度連續性。
**界線**：靜態運動學、剛性夾持、無接觸力；門餘轉以 δ 包絡代替（不推論張爪後餘轉小）；距離下界只涵蓋取樣間運動，不含模型誤差。

    python3 evaluation/door_coopen_verify_v1.py [idx ...]   # 寫 results/door/door_coopen_verify_v1.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_joint_v2 as J2                      # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_release_set_v1 as RS                # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_joint_v2.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_verify_v1.json')
C = RV.C
DS = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0)
TH = {'nonfinger': 0.005, 'finger': 0.005, 'self': 0.005, 'base': 0.020}
E_BASE = 0.010
V_AXIS, V_LIN, W_ANG, A_LIN, ARM_RATE, DOOR_RATE, V_TCP = 0.035, 0.05, 0.20, 0.5, 1.0, 1.0, 0.05
Q_CLOSED, OPEN = RV.Q_FINGER, C.FINGER_OPEN
DYAW = math.radians(90.0)


class Path:
    def __init__(self, P):
        self.P = np.asarray(P, float)
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.P, axis=0), axis=1))])

    def at(self, s):
        s = float(np.clip(s, 0.0, self.s[-1]))
        return np.array([np.interp(s, self.s, self.P[:, 0]), np.interp(s, self.s, self.P[:, 1])])

    def s_at_y(self, y):
        k = int(np.argmax(self.P[:, 1] >= y)) if np.any(self.P[:, 1] >= y) else len(self.P) - 1
        return float(self.s[k])


def geo(qf, q_deg, qfi):
    d = TJ.GEO([float(v) for v in qf], math.radians(q_deg), qfi)
    return {k: (float(d[k][0]), d[k][1]) for k in TH}


def links_world(qf):
    out = []
    for n in C.ARM_LINKS:
        T = C.K.fk(np.asarray(qf, float), n)
        out.append((T[:3, :3] @ C.CL[n][::30].T).T + T[:3, 3])
    return np.concatenate(out)


def run(cand):
    g = tuple(cand['grasp'])
    q_rel = float(cand['q_rel_deg'])
    q = math.radians(q_rel)
    path = Path(cand['exit_path'])
    A = cand['trajectory']
    Tg = EG.grasp_T(q, *g)
    u = np.array([math.cos(q), -math.sin(q), 0.0])
    n = np.array([-math.sin(q), -math.cos(q), 0.0])
    pts = []                                   # 每點：phase, base(3), qa(6), q_finger, q_door, tcp(或 None)
    for p in A:
        pts.append({'phase': 'A_hold', 'base': np.array(p['base']), 'qa': np.array(p['qa']), 'qf': Q_CLOSED, 'q': p['q_deg'], 'tcp': None})
    yaw0 = pts[-1]['base'][2]
    seed = pts[-1]['qa']
    s = 0.0
    fail = None

    def solve(ph, T, s_, yaw, qfi, tcp):
        nonlocal seed, fail
        b = path.at(s_)
        base = np.array([b[0], b[1], yaw])
        q9 = C.ik(base, T, seed, only_prev=True)
        if q9 is None or float(np.max(np.abs(np.asarray(q9)[C.IDX] - seed))) > TJ.LIM['joint_step_rad']:
            fail = {'phase': ph, 's_m': s_, 'why': 'ik' if q9 is None else 'joint_step'}
            return False
        seed = np.asarray(q9)[C.IDX]
        pts.append({'phase': ph, 'base': base, 'qa': seed.copy(), 'qf': qfi, 'q': q_rel, 'tcp': tcp, 's': s_})
        return True
    fwd = float(cand['fwd_m'])
    for k in range(1, int(round(fwd / 0.01)) + 1):
        if not solve('B_fwd_hold', Tg, 0.01 * k, yaw0, Q_CLOSED, Tg[:3, 3].tolist()):
            break
    s = fwd
    if fail is None:
        for k in range(3):
            if not solve('C_open', Tg, s, yaw0, Q_CLOSED + (OPEN - Q_CLOSED) * (k + 1) / 3, Tg[:3, 3].tolist()):
                break
    if fail is None:
        for k in range(1, 7):
            s += 0.005
            T = Tg @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.01 * k], [0, 0, 0, 1.0]])
            if not solve('C_retract', T, s, yaw0, OPEN, T[:3, 3].tolist()):
                break
    if fail is None:
        p0 = (Tg @ np.array([0, 0, -0.06, 1.0]))[:3]
        seq = [u] * 150 + [-n] * 100
        p_ = p0.copy()
        for i, vec in enumerate(seq):
            p_ = p_ + vec * 0.002
            s += 0.001
            T = np.eye(4)
            T[:3, :3] = Tg[:3, :3]
            T[:3, 3] = p_
            if not solve('D_unwrap', T, s, yaw0 + DYAW * (i + 1) / len(seq), OPEN, p_.tolist()):
                break
    if fail is None:
        qa0 = seed.copy()
        for k in range(1, 41):
            s += 0.0075
            b = path.at(s)
            qa = (1 - k / 40) * qa0 + (k / 40) * TJ.STOW
            pts.append({'phase': 'E_stow', 'base': np.array([b[0], b[1], yaw0 + DYAW]), 'qa': qa, 'qf': OPEN, 'q': q_rel, 'tcp': None, 's': s})
        s_end = max(s, path.s_at_y(0.50))
        for s_ in np.arange(s + 0.02, s_end + 1e-9, 0.02):
            b = path.at(s_)
            pts.append({'phase': 'F_pass', 'base': np.array([b[0], b[1], yaw0 + DYAW]), 'qa': TJ.STOW.copy(), 'qf': OPEN, 'q': q_rel, 'tcp': None, 's': float(s_)})
    # 核驗
    rows, prevW = [], None
    for i, p in enumerate(pts):
        TJ.cool()
        qf = list(p['base']) + list(p['qa'])
        if p['phase'].startswith('A') or p['phase'].startswith('B'):
            gd = geo(qf, p['q'], p['qf'])
            tol = None
        else:
            gds = {dq: geo(qf, p['q'] + dq, p['qf']) for dq in DS}
            gd = gds[0.0]
            ok = {dq: all(gds[dq][c][0] >= TH[c] for c in TH) for dq in DS}
            pos = 0.0
            for dq in sorted(x for x in DS if x > 0):
                if not ok[dq]:
                    break
                pos = dq
            neg = 0.0
            for dq in sorted((x for x in DS if x < 0), reverse=True):
                if not ok[dq]:
                    break
                neg = dq
            tol = (pos, neg, ok[0.0])
        W = links_world(qf)
        dl = 0.0 if prevW is None else float(np.max(np.linalg.norm(W - prevW, axis=1)))
        db = 0.0 if i == 0 else float(np.linalg.norm(p['base'][:2] - pts[i - 1]['base'][:2]))
        dd = 0.0 if i == 0 else 0.95 * math.radians(abs(p['q'] - pts[i - 1]['q']))
        prevW = W
        rows.append({'i': i, 'phase': p['phase'], 'q_deg': p['q'], 'base': p['base'].tolist(), 'qa': p['qa'].tolist(), 'q_finger': p['qf'],
                     'dist': {c: gd[c][0] for c in TH}, 'who': {c: gd[c][1] for c in TH}, 'lip_m': max(db, dl, dd) / 2.0, 'tol': tol})
    for r, r2 in zip(rows, rows[1:] + [None]):
        lip = max(r['lip_m'], r2['lip_m'] if r2 else 0.0)
        r['lower'] = {c: r['dist'][c] - lip for c in TH}
    # 時間化
    dts = []
    for a, b in zip(pts, pts[1:]):
        dbv = b['base'][:2] - a['base'][:2]
        dyw = abs((b['base'][2] - a['base'][2] + math.pi) % (2 * math.pi) - math.pi)
        dqa = float(np.max(np.abs(b['qa'] - a['qa'])))
        dtcp = np.linalg.norm(np.subtract(b['tcp'], a['tcp'])) if (a['tcp'] is not None and b['tcp'] is not None) else 0.0
        dts.append(max(abs(b['q'] - a['q']) / DOOR_RATE, abs(dbv[0]) / V_AXIS, abs(dbv[1]) / V_AXIS, np.linalg.norm(dbv) / V_LIN,
                       dyw / W_ANG, dqa / ARM_RATE, dtcp / V_TCP, 1e-3))
    vel = [(pts[k + 1]['base'][:2] - pts[k]['base'][:2]) / dts[k] for k in range(len(dts))]
    phases = [p['phase'] for p in pts]

    def last_idx(ph):
        return max(i for i, x in enumerate(phases) if x == ph)

    def first_idx(ph):
        return min(i for i, x in enumerate(phases) if x == ph)
    junction = {}
    for a_ph, b_ph in (('A_hold', 'B_fwd_hold'), ('B_fwd_hold', 'C_open'), ('C_retract', 'D_unwrap')):
        if a_ph in phases and b_ph in phases:
            va, vb = vel[last_idx(a_ph) - 1], vel[min(first_idx(b_ph), len(vel) - 1)]
            dv = float(np.linalg.norm(vb - va))
            junction[f'{a_ph}->{b_ph}'] = {'v_before': va.tolist(), 'v_after': vb.tolist(), 'dv_m_s': dv, 't_min_s': dv / A_LIN}
    # 出口進展
    F = RS.field(q_rel)
    _, Dtrue = J2.d_map(F)
    Dq = [J2.D_at(Dtrue, p['base']) for p in A]
    Dq = [None if not math.isfinite(v) else v for v in Dq]
    fin = [(p['q_deg'], v) for p, v in zip(A, Dq) if v is not None]
    summ = {}
    for ph in ('A_hold', 'B_fwd_hold', 'C_open', 'C_retract', 'D_unwrap', 'E_stow', 'F_pass'):
        r = [x for x in rows if x['phase'] == ph]
        if not r:
            continue
        mn = {c: min(x['dist'][c] for x in r) for c in TH}
        lo = {c: min(x['lower'][c] for x in r) for c in TH}
        rec = {'n': len(r), 'min_dist_mm': {c: v * 1000 for c, v in mn.items()}, 'lower_bound_mm': {c: v * 1000 for c, v in lo.items()},
               'margin_mm': {c: (mn[c] - TH[c]) * 1000 for c in TH}, 'base_margin_after_10mm_mm': (lo['base'] - TH['base'] - E_BASE) * 1000,
               'pass_threshold': all(lo[c] >= TH[c] for c in TH)}
        if r[0]['tol'] is not None:
            rec['delta_plus_ok_min_deg'] = min(x['tol'][0] for x in r)
            rec['delta_minus_ok_max_deg'] = max(x['tol'][1] for x in r)
        summ[ph] = rec
    total_t = float(sum(dts))
    return {'grasp': list(g), 'q_rel_deg': q_rel, 'fail': fail, 'n_points': len(pts), 'summary': summ, 'junction': junction,
            'time_s': {'total': total_t, 'by_phase': {ph: float(sum(dt for dt, p in zip(dts, pts[1:]) if p['phase'] == ph)) for ph in summ}},
            'exit_progress': {'D_by_q': [[p['q_deg'], v] for p, v in zip(A, Dq)], 'D_first_finite': fin[0] if fin else None,
                              'D_last': fin[-1] if fin else None, 'D_drop_last10_m': (dict(fin).get(q_rel - 10) - fin[-1][1]) if (fin and dict(fin).get(q_rel - 10) is not None) else None},
            'rows': rows}


def main():
    global OUT
    if os.path.exists(OUT) and not (len(sys.argv) > 1 and sys.argv[1].endswith('.json')):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    src = SRC
    args = sys.argv[1:]
    if args and args[0].endswith('.json'):
        src = args.pop(0)
        OUT = OUT.replace('.json', '_' + os.path.basename(src).replace('door_coopen_joint_v2_', ''))
        if os.path.exists(OUT):
            raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(src))
    full = [f for f in d['full'] if f.get('trajectory')]
    idx = [int(a) for a in args] or list(range(len(full)))
    res = []
    for i in idx:
        r = run(full[i])
        res.append(r)
        s = r['summary']
        print(f"候選 {i}（φ{r['grasp'][0]} s{r['grasp'][1]} q{r['q_rel_deg']:.0f}）：失敗 {r['fail']}；總時長 {r['time_s']['total']:.0f} s；"
              + '；'.join(f"{ph} 下界過門檻 {v['pass_threshold']} 底盤扣誤差餘裕 {v['base_margin_after_10mm_mm']:.1f} mm"
                         + (f" δ+≤{v['delta_plus_ok_min_deg']} δ−≥{v['delta_minus_ok_max_deg']}" if 'delta_plus_ok_min_deg' in v else '')
                         for ph, v in s.items())
              + f"；交界 {({k: round(v['dv_m_s'], 3) for k, v in r['junction'].items()})}；末 10° D 降 {r['exit_progress']['D_drop_last10_m']}", flush=True)
    json.dump({'schema': 'door_coopen_verify/1', 'source': os.path.relpath(src, HERE), 'thresholds_m': TH, 'e_base_m': E_BASE, 'deltas_deg': DS,
               'candidates': res, 'not_modeled': ['接觸力、夾持穩定性', '門餘轉動力學（δ 包絡代替）', '模型與定位誤差（只有底盤 10 mm 壓力等級）', '控制器追蹤']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
