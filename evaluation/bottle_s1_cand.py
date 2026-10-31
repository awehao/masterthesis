#!/usr/bin/env python3
"""S1 候選離線閉迴路與幾何（Codex 20261031_210846：完整任務、實際軌跡幾何、導航資料來源、三欄指標）。非物理：假受控對象、理想底盤、合成接觸。
場景：桌瓶同 S0b；0.40×0.40×0.50 m 方塊障礙在路徑上；起點 x −2.4；出口區圓心 (1.0, 0)、半徑 0.3。障礙以外接圓給 CBF（真值，r＝0.283＋靜態邊距 0.47）；**無 LiDAR、無 Raw-scan Shield**。
參考路徑 y(x) 升餘弦繞行（工程給定，非規劃器輸出）。幾何含障礙方塊 ↔ 手臂／持瓶／0.39 m 圓盤模型。
    python3 evaluation/bottle_s1_cand.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s0b_path_check as PC                      # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

X0, X_END = -3.6, 0.85
OBST = dict(c=np.array([-2.0, 0.0, 0.25]), sz=np.array([0.40, 0.40, 0.50]))
R_OBST, M_STATIC, R_DISC = 0.2 * math.sqrt(2.0), 0.47, 0.39
PATH = (-3.4, -0.6, -0.92)         # 升餘弦繞行，對稱於障礙；CLEAR_MIN 由 path_clearance 硬性核對
PATH_P2 = (-3.4, -0.6, -0.81)      # 貼近路徑：CBF 實際介入
Q_TUCK = (0.0, 0.0, 0.035, 0.0, -1.5708, 0.0)   # 起始與行進姿態：K2_READY（Howard 指定，10/31），同門任務起始
FAST = dict(q_tuck=Q_TUCK, v_nav=0.35, v_c=0.10, v_carry=0.25, x_slow=-0.90, x_unfold=-1.55, t_unfold=3.4, t_blend=3.0, t_insert=1.5)   # Howard 指定速度；展臂在高速段進行，只有朝瓶段降速
CLEAR_MIN = 0.03                   # 參考路徑對 r_eff 的最小餘裕（m）：路徑穿進 keep-out 時底盤與參考會一起停在邊界（r1 死鎖）
EXIT_C, EXIT_R = (1.0, 0.0), 0.30


def obstacles(t):
    return [dict(x=float(OBST['c'][0]), y=float(OBST['c'][1]), radius=R_OBST, margin=M_STATIC, static=True)]


def make(**kw):
    p = S1.DetourPath(*kw.pop('path', PATH))
    return S1.S1Controller(T.K, T.ID, T.T_G, T.T_READY, T.Q_READY, T.TOP, T.R_B, X0, T.WIN, p, EXIT_C, EXIT_R, kw.pop('obs', obstacles), x_exit_stop=X_END, required=T.REQ, **kw)


def scene_boxes():
    return [(OBST['c'], OBST['sz'])] + [(b[1], b[2]) for b in G.table_boxes()]


def make_lidar(boxes=None, **kw):
    """掃描鏈版本：障礙只來自離線射線 → 追蹤器；底盤命令經 Shield。"""
    ch = NC.NavChain(map_boxes=[(b[1], b[2]) for b in G.table_boxes() if b[0] != 'table_top'])
    bx = scene_boxes() if boxes is None else boxes
    fn = kw.pop('scan_fn', lambda t, q: NC.ray_boxes(q[:2], float(q[2]), bx, G.LIDAR_Z))
    return make(path=kw.pop('path', PATH_P2), obs=kw.pop('obs', ch.obstacles), chain=ch, scan_fn=fn, **SC_FAST(kw))


def SC_FAST(kw):
    return dict(FAST, **kw)


def path_clearance(path=PATH):
    p = S1.DetourPath(*path)
    xs = np.linspace(X0, X_END, 4000)
    return float(min(math.hypot(x - OBST['c'][0], p.y(float(x)) - OBST['c'][1]) for x in xs) - R_OBST - M_STATIC)


def obstacle_geom(f, oc):
    box, out = (oc, OBST['sz']), dict(arm=9.9, finger=9.9, held=9.9, base_disc=9.9)
    rel = None
    for h in f.hist[::5]:
        W = G.link_world(h['q'], G.Q_OPEN)
        for n, P in W.items():
            k = 'finger' if n in G.C.FINGERS else 'arm'
            out[k] = min(out[k], float(G.box_dist(P, *box).min()))
        dx, dy = max(abs(h['q'][0] - box[0][0]) - box[1][0] / 2, 0.0), max(abs(h['q'][1] - box[0][1]) - box[1][1] / 2, 0.0)
        out['base_disc'] = min(out['base_disc'], math.hypot(dx, dy) - R_DISC)
        if h['phase'] in ('LIFT', 'CARRY'):
            Tt = T.K.fk(h['q'], 'link_tcp')
            if rel is None:
                T0 = np.eye(4)
                T0[:3, 3] = [T.BX, T.BY, T.TOP]
                rel = np.linalg.inv(Tt) @ T0
            Tb = Tt @ rel
            Pw = (Tb[:3, :3] @ G.bottle_points().T).T + Tb[:3, 3]
            out['held'] = min(out['held'], float(G.box_dist(Pw, *box).min()))
    return {k: (round(v * 1e3, 1) if v < 9.0 else None) for k, v in out.items()}


def metrics(name, f, oc=None):
    oc = OBST['c'] if oc is None else oc                # 障礙實際位置（偏移案例不可用名目位置算餘裕）
    r = f.c.report()
    q = np.array([h['q'][:3] for h in f.hist])
    t = np.array([h['t'] for h in f.hist])
    vx, vy = np.gradient(q[:, 0], 0.01), np.gradient(q[:, 1], 0.01)
    sp = np.convolve(np.hypot(vx, vy), np.ones(5) / 5, mode='same')
    ax = np.gradient(sp, 0.01)
    L = f.c.cbf_log
    bound = [e for e in L if e['cbf_active'] > 0 and e['du'] >= 1e-3]
    ph = {}
    for e in bound:
        ph[e['phase']] = ph.get(e['phase'], 0) + 1
    d_c = np.hypot(q[:, 0] - oc[0], q[:, 1] - oc[1])
    tc, tl = r['t_close'], r['t_lift_start']
    mk = (t >= tc) & (t <= tl) if tc is not None and tl is not None else np.zeros(len(t), bool)
    mins, sm, _, _, marg = PC.check(f)
    keys = [k for k in mins if 'table_top（LIFT）' not in k]
    k0 = min(keys, key=lambda k_: mins[k_][0])
    ms = np.array(r['solve_ms_all'])
    return dict(case=name,
                task=dict(success=r['success'], exit_code=S1.exit_code(r), clear_bottle=r['clear_bottle'], judge=r['result'], exit_reached=r['exit_reached'], t_exit_in=r['t_exit_in'], mission_done=r['mission_done'], stop=(r['stop'] or {}).get('why')),
                safety=dict(forbidden=r['forbidden_pulses'], cbf_changed=len(bound), cbf_ticks=len(L), cbf_phase=ph, k0_slack=round(max([e['eps0'] for e in L], default=0.0), 6), clear_r_eff_mm=round(float(d_c.min() - R_OBST - M_STATIC) * 1e3, 1), clear_disc_model_mm=round(float(d_c.min() - R_OBST - R_DISC) * 1e3, 1),
                            obstacle_geom=obstacle_geom(f, oc), slip_mm_deg=[round(v, 3) for v in r['slip_max_mm_deg']], geom_min=dict(key=k0, mm=round(mins[k0][0] * 1e3, 1)), self_mm=round(sm[0] * 1e3, 1), joint_margin=round(marg, 3), fallback=r['n_fallback'], yaw_peak_deg=round(math.degrees(float(np.abs(q[:, 2]).max())), 2)),
                efficiency=dict(t_total=round(float(t[-1]), 2), t_grasp_window=[tc, tl], v_grasp=[round(float(sp[mk].min()), 4), round(float(sp[mk].max()), 4)] if mk.any() else None, near_stop_grasp_s=round(float((sp[mk] < 0.01).sum()) * 0.01, 2) if mk.any() else None, near_stop_total_s=round(float((sp < 0.01).sum()) * 0.01, 2), v_peak=round(float(sp.max()), 4), a_peak=round(float(np.abs(ax).max()), 3), x_end=round(float(q[-1, 0]), 4), y_end=round(float(q[-1, 1]), 4), n_gate=r['n_gate'],
                                ms=dict(p50=round(float(np.median(ms)), 1), p95=round(float(np.percentile(ms, 95)), 1), p99=round(float(np.percentile(ms, 99)), 1), max=round(float(ms.max()), 1)), n_over_50ms=r['n_over_50ms'], solver=r['qp_counts'], gi_fail=r['gi_fail']))


def main():
    res = []
    pc = path_clearance()
    print(f'參考路徑對 r_eff 餘裕 {pc * 1e3:.1f} mm')
    if pc < CLEAR_MIN:
        print(f'參考路徑餘裕低於 {CLEAR_MIN * 1e3:.0f} mm ⇒ 拒跑')
        return 2
    tight = PATH_P2
    print(f'貼近路徑 P2，對 r_eff 餘裕 {path_clearance(tight) * 1e3:.1f} mm')
    D_SHIFT = 0.12
    shifted = lambda t: [dict(obstacles(t)[0], y=float(OBST['c'][1]) - D_SHIFT)]
    for name, kw, ckw in (('P1 名目路徑', {}, {}), ('P1 伺服慢未知 α×0.6', dict(alpha_scale=0.6), {}), ('P1 底盤打折 ×0.8', dict(v_scale=0.8), {}),
                          ('P2 貼近路徑', {}, dict(path=tight)), ('P2 伺服慢未知 α×0.6', dict(alpha_scale=0.6), dict(path=tight)),
                          ('N1 障礙實際偏 0.12 m（路徑未更新）', {}, dict(obs=shifted))):
        c = make(log_cbf=True, **ckw)
        t_lim = 110.0
        f = T.Fake(c, x0=X0, **kw).run(t_lim, (lambda t, s: (s.c._trigger_stop('sim_limit') if t >= 100.0 else None) or {}))
        m = metrics(name, f, OBST['c'] - np.array([0.0, D_SHIFT, 0.0]) if name.startswith('N1') else None)
        res.append(m)
        print(f"\n[{name}]\n  任務 {m['task']}\n  安全 {m['safety']}\n  效率 {m['efficiency']}", flush=True)
    out = os.path.join(HERE, 'results', 'bottle', 's1_cand_r5.json')
    json.dump(dict(note='假受控對象、理想底盤、合成接觸；真值障礙注入 CBF、無 LiDAR、無 Raw-scan Shield；參考路徑工程給定；r1 路徑穿進 keep-out 死鎖、r3 N1 餘裕誤用名目障礙位置、r4 無線上淨空拒絕（皆保留）', obstacle=dict(c=OBST['c'].tolist(), sz=OBST['sz'].tolist(), r_cbf=R_OBST, margin=M_STATIC), exit=dict(c=EXIT_C, r=EXIT_R), path=PATH, cases=res), open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫入', os.path.relpath(out, os.path.dirname(HERE)))


if __name__ == '__main__':
    sys.exit(main())
