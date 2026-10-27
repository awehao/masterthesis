#!/usr/bin/env python3
"""v3（door_coopen_release_bench6_v2.py 的版本化修正；Codex r40 §1–§3）：
  (1) 存完整構型：張爪、退開、STOW 每點都記 base／qa／q_finger；
  (2) 依求解器設定重新時間化（底盤本體每軸 0.035255 m/s、偏航 0.1999 rad/s、線加速度 0.5 m/s²、角加速度 2.0 rad/s²；手臂每軸 0.9999 rad/s、19.984 rad/s²；
      張爪每點下限 0.1 s；起訖靜止；加速度以前向／反向兩次時間縮放滿足），再**依新時間**重算三種門餘轉情境的門角並重核全身淨空；
  (3) 指定手指—橫桿配對：兩指點雲對橫桿膠囊（door_geom_v2.bar_capsule_dist）距離逐點核：t＝0 初始值明列；首次 ≥ 1 mm（離開）後不得 < 0（重新穿入）；
  (4) 關節位置餘裕以 arm_limits.LITE6_SAFE 逐點核（≥ 0.05 rad）。
  取樣構型序列仍由 v2 的全身最小範數求得（門角用舊時間）；重新時間化後只重核淨空與門角，不重解 IK（構型與門角無關，因已放手）。
以下為 v2 說明。
v2（door_coopen_release_bench6_v1.py 的版本化修正）：v1 底盤位置固定、只動手臂 ⇒ W1 走 2 cm 即 IK 失敗 8、單步過大 7（手臂獨自補償，構型條件差）。
v2 退臂改用全身加權最小範數（door_coopen_pass_close_v3.solve_step：底盤 x,y 便宜、偏航 4 倍、手臂 25 倍；Newton 收斂），底盤可平移；
單步關節 ≤ 0.10 rad、底盤 ≤ 0.03 m；姿態候選同 v1；其餘判準同 v1。收臂（STOW 內插）時底盤固定在退臂終點。
以下為 v1 說明。
A 包（Codex r39 §下一候選 1）：bench_6 實錄開門末態能否「張爪 → 解除 → 退開」（door_coopen_release_plan_v1 的版本化改寫；只改起點與底盤處置）。
起點：bench_6 STOPPED 後、門已靜止的一個物理步（預設 T_PICK 110.0 s；門角約 47.9°）的實測底盤位姿、手臂角與門角；
  起點 TCP＝實測關節正運動學（不重設抓取關係、不瞬間對準名目抓取位姿）；門扇座標 u、n 依該步實測門角。
底盤：位置固定（不沿出口路徑前進），偏航候選 {0, ±2°}；其餘（張爪 3 點 0.3 s、路徑點 W1＝TCP＋0.08 n、W2＝1.07 u＋0.08 n、W3＝1.07 u − 0.15 n、
  每步 1 cm、姿態候選 {不變、繞世界 z ±10°、繞工具 x ±10°}、IK 前一點種子且單步 ≤ 0.10 rad、門角依壓力情境、非手指／手指（張爪開度）／自碰 ≥ 5 mm、
  底盤 ≥ 30 mm、之後關節內插到 STOW 40 點）同原版。壓力情境：static、k3b_stress、reverse_0p5（原版定義；只作壓力，不作本抓法上界）。
報：每情境是否全程成立、首個失敗的時刻／階段／部位、最小餘裕。不成立 ⇒ 即 A 包本候選的第一個阻點。

    python3 evaluation/door_coopen_release_bench6_v1.py [T_PICK]   # 寫 results/door/door_coopen_release_bench6_v1.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_v3 as PC3               # noqa: E402  v2：全身解
import door_coopen_release_plan_v1 as RP               # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_release_bench6_v3.json')
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6')
C = RP.C


def plan(b0, qa0, q_deg, scen):
    dq = RP.SCEN[scen]
    q = math.radians(q_deg)
    u = np.array([math.cos(q), -math.sin(q), 0.0])
    n = np.array([-math.sin(q), -math.cos(q), 0.0])
    base0 = np.array(b0, float)
    seed = np.array(qa0, float)
    Tg = C.K.fk(np.concatenate([base0, seed]), 'link_tcp')
    yaw = float(base0[2])
    t, rows = 0.0, []
    for k in range(3):
        t += 0.1
        qfi = RP.Q_CLOSED + (RP.OPEN - RP.Q_CLOSED) * (k + 1) / 3
        m, gg = RP.dmin(list(base0) + list(seed), q_deg + dq(t), qfi)
        rows.append({'t': t, 'phase': 'open', 'q_door': q_deg + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0],
                     'base': [float(v) for v in base0], 'qa': [float(v) for v in seed], 'q_finger': qfi})
        if m[0] < 0 or gg['base'][0] < RP.BASE_PLAN:
            return rows, {'t': t, 'phase': 'open', 'who': f'{m[1]}:{m[2]}', 'margin_mm': m[0] * 1e3}
    p, R = Tg[:3, 3].copy(), Tg[:3, :3].copy()
    W = [p + 0.08 * n, 1.07 * u * np.array([1, 1, 0]) + np.array([0, 0, p[2]]) + 0.08 * n,
         1.07 * u * np.array([1, 1, 0]) + np.array([0, 0, p[2]]) - 0.15 * n]
    for wi, w in enumerate(W):
        it = 0
        while np.linalg.norm(w - p) > 0.005 and it < 80:
            it += 1
            TJ.cool()
            p_new = p + (w - p) * min(1.0, 0.01 / np.linalg.norm(w - p))
            best, why_all = None, {}
            for rot in (np.eye(3), RP.Rax([0, 0, 1], 0.17), RP.Rax([0, 0, 1], -0.17), RP.Rax(R[:, 0], 0.17), RP.Rax(R[:, 0], -0.17)):
                R2 = rot @ R
                for dyw in (0.0,):
                    T = np.eye(4)
                    T[:3, :3], T[:3, 3] = R2, p_new
                    xcur = np.concatenate([base0, seed])
                    xn = PC3.solve_step(xcur, T, np.zeros(9))
                    if xn is None:
                        why_all['ik'] = why_all.get('ik', 0) + 1
                        continue
                    qa, base = xn[3:], xn[:3]
                    if float(np.max(np.abs(qa - seed))) > TJ.LIM['joint_step_rad'] or float(np.linalg.norm(base[:2] - base0[:2])) > 0.03:
                        why_all['step'] = why_all.get('step', 0) + 1
                        continue
                    dyw = float(base[2] - base0[2])
                    dt = max(0.01 / RP.V_TCP, float(np.max(np.abs(qa - seed))) / RP.ARM_RATE, abs(dyw) / RP.W_ANG)
                    m, gg = RP.dmin(list(base) + list(qa), q_deg + dq(t + dt), RP.OPEN)
                    if gg['base'][0] < RP.BASE_PLAN:
                        why_all['base'] = why_all.get('base', 0) + 1
                        continue
                    if best is None or m[0] > best[0][0]:
                        best = (m, qa, R2, base, dt, gg)
            if best is None:
                return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': 'ik_or_step_or_base', 'detail': why_all, 'tcp': p.tolist()}
            m, seed, R, base, dt, gg = best
            base0 = np.array(base, float)
            yaw = float(base[2])
            t += dt
            p = p_new
            rows.append({'t': t, 'phase': f'W{wi + 1}', 'q_door': q_deg + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0],
                         'tcp': p.tolist(), 'qa': seed.tolist(), 'base': base.tolist(), 'q_finger': RP.OPEN})
            if m[0] < 0:
                return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': f'{m[1]}:{m[2]}', 'margin_mm': m[0] * 1e3, 'tcp': p.tolist()}
        if np.linalg.norm(w - p) > 0.005:
            return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': 'not_reached'}
    qa0s = seed.copy()
    for k in range(1, 41):
        TJ.cool()
        qa = (1 - k / 40) * qa0s + (k / 40) * TJ.STOW
        t += 0.05
        base = np.array([base0[0], base0[1], yaw])
        m, gg = RP.dmin(list(base) + list(qa), q_deg + dq(t), RP.OPEN)
        rows.append({'t': t, 'phase': 'stow', 'q_door': q_deg + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0],
                     'base': [float(v) for v in base], 'qa': [float(v) for v in qa], 'q_finger': RP.OPEN})
        if m[0] < 0 or gg['base'][0] < RP.BASE_PLAN:
            return rows, {'t': t, 'phase': 'stow', 'who': f'{m[1]}:{m[2]}', 'margin_mm': m[0] * 1e3}
    return rows, None


VB, WB, AB, AW = 0.035255, 0.1999, 0.5, 2.0
VA, AA = 0.9999, 19.984


def retime(cfgs):
    """cfgs：[(base3, qa6, phase)]（含 t＝0 起點）。回傳各點時間（起訖靜止、速度與加速度上限）。"""
    n = len(cfgs)
    seg = []
    for k in range(1, n):
        b0, b1 = np.asarray(cfgs[k - 1][0]), np.asarray(cfgs[k][0])
        yaw = float(b0[2])
        dw = np.array([b1[0] - b0[0], b1[1] - b0[1]])
        bx = math.cos(yaw) * dw[0] + math.sin(yaw) * dw[1]
        by = -math.sin(yaw) * dw[0] + math.cos(yaw) * dw[1]
        dq = np.abs(np.asarray(cfgs[k][1]) - np.asarray(cfgs[k - 1][1]))
        d = np.array([abs(bx), abs(by), abs(b1[2] - b0[2])] + list(dq))
        vmax = np.array([VB, VB, WB] + [VA] * 6)
        amax = np.array([AB, AB, AW] + [AA] * 6)
        dt = max(float(np.max(d / vmax)), 1e-3, 0.1 if cfgs[k][2] == 'open' else 0.0)
        seg.append([d, dt, amax])
    for _ in range(50):                                   # 前向／反向縮放直到加速度成立（含起訖靜止）
        changed = False
        vel = [s_[0] / s_[1] for s_ in seg]
        for k in range(len(seg)):
            vp = np.zeros(9) if k == 0 else vel[k - 1]
            vn = np.zeros(9) if k == len(seg) - 1 else vel[k + 1]
            need = max(float(np.max(np.abs(vel[k] - vp) / seg[k][2])), float(np.max(np.abs(vn - vel[k]) / seg[k][2])))
            if need > seg[k][1] * 1.0001:
                seg[k][1] = max(seg[k][1] * 1.2, math.sqrt(float(np.max(seg[k][0] / seg[k][2]))))
                changed = True
                vel[k] = seg[k][0] / seg[k][1]
        if not changed:
            break
    ts = [0.0]
    for s_ in seg:
        ts.append(ts[-1] + s_[1])
    return ts


def finger_bar(base, qa, q_deg, qfi):
    import door_geom_v2 as G2
    K1 = RP.RV.K1
    qf = [float(v) for v in list(base) + list(qa)]
    W = C.world(qf, list(C.FINGERS), qfi)
    return {f: float(np.min(G2.bar_capsule_dist(K1.D2, W[f], math.radians(q_deg), K1.H))) for f in C.FINGERS}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t_pick = float(sys.argv[1]) if len(sys.argv) > 1 else 110.0
    r6 = json.load(open(os.path.join(RUN, 'room_run.json')))
    c = {k: i for i, k in enumerate(r6['steps_cols'])}
    s = min(r6['steps'], key=lambda z: abs(z[c['sim_t']] - t_pick))
    b0, qa0, q_deg = list(s[c['base_xyth']]), list(s[c['q_arm_meas']]), math.degrees(s[c['door_q_rad']])
    res = {'schema': 'door_coopen_release_bench6/1', 'start': {'sim_t': s[c['sim_t']], 'step': s[c['step']], 'base': b0, 'qa': qa0, 'q_deg': q_deg,
                                                                'finger_contact_n': s[c['finger_contact_n']]}, 'results': {}}
    from ammr_wholebody_mpc import arm_limits as AL
    for sc in ('static', 'k3b_stress', 'reverse_0p5'):
        rows, fail = plan(b0, qa0, q_deg, sc)
        mm = min(r['min_margin_m'] for r in rows) if rows else None
        res['results'][sc] = {'ok': fail is None, 'fail': fail, 'n': len(rows), 'min_margin_mm': None if mm is None else mm * 1e3, 'rows': rows}
        if fail is not None:
            continue
        cfgs = [(b0, qa0, 'start')] + [(r_['base'], r_['qa'], r_['phase']) for r_ in rows]
        ts = retime(cfgs)
        dq = RP.SCEN[sc]
        rech, fb, exited, re_pen, jlm = [], [], False, [], 1e9
        fb0 = finger_bar(b0, qa0, q_deg, RP.Q_CLOSED)
        for k, r_ in enumerate(rows):
            tk = ts[k + 1]
            qd = q_deg + dq(tk)
            m, gg = RP.dmin(list(r_['base']) + list(r_['qa']), qd, r_['q_finger'])
            f = finger_bar(r_['base'], r_['qa'], qd, r_['q_finger'])
            fmin = min(f.values())
            if fmin >= 0.001:
                exited = True
            if exited and fmin < 0.0:
                re_pen.append((k, tk, fmin))
            fb.append(fmin)
            qa_ = np.asarray(r_['qa'])
            jlm = min(jlm, float(min(min(qa_ - AL.LITE6_SAFE.lower), min(AL.LITE6_SAFE.upper - qa_))))
            rech.append({'t': tk, 'q_door': qd, 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0], 'finger_bar_min_m': fmin})
        bad = [x for x in rech if x['min_margin_m'] < 0 or x['base_m'] < RP.BASE_PLAN]
        res['results'][sc]['retimed'] = {'t_end_s': ts[-1], 'recheck_ok': not bad, 'first_bad': bad[:3], 'min_margin_mm': min(x['min_margin_m'] for x in rech) * 1e3,
                                          'finger_bar_t0_m': fb0, 'finger_bar_min_after_exit_m': min((f for f, x in zip(fb, rech)), default=None),
                                          'finger_exit_t': next((x['t'] for x in rech if x['finger_bar_min_m'] >= 0.001), None),
                                          'repenetration': re_pen[:5], 'joint_limit_margin_min_rad': jlm, 'joint_limit_ok': jlm >= 0.05,
                                          'limits': {'base_body_axis_m_s': VB, 'yaw_rad_s': WB, 'a_lin': AB, 'a_ang': AW, 'arm_rad_s': VA, 'arm_rad_s2': AA},
                                          'rows': rech}
        rt = res['results'][sc]['retimed']
        print(f"  {sc} 重新時間化：{rt['t_end_s']:.1f} s；重核淨空 {'成立' if rt['recheck_ok'] else '不成立'}（最小餘裕 {rt['min_margin_mm']:.1f} mm）；"
              f"手指—橫桿 t0 {({k: round(v * 1e3, 2) for k, v in fb0.items()})} mm、離開於 {rt['finger_exit_t']} s、重新穿入 {len(re_pen)}；關節限位餘裕 {jlm:.3f} rad", flush=True)
        print(f"情境 {sc}：{'成立' if fail is None else '失敗於 ' + json.dumps(fail, ensure_ascii=False, default=float)[:300]}；{len(rows)} 點；最小餘裕 {res['results'][sc]['min_margin_mm']}", flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
