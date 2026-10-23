#!/usr/bin/env python3
"""K3-M「正常受控暫停／減速保持」的離線閉迴路核對（無 Isaac、無接觸；規格 K3_controlled_pause_spec.md；依 Codex 20261023_221949 第 3 節）。

只處理**健康狀態下的預定暫停**（量測、求解、雙指接觸、走廊與幾何監看都正常）；不處理故障後跟隨。
受控體與控制器同 door_k3m_offline_replay.py r2（wgmpc_closed_loop_sim.run、solve_sp 本體、free4 手臂模型、底盤增益 0.9／0.9／0.82、
延遲 1.4 週期、補償 1.6／1.0、權重同跑法；門角由 TCP 經剛性夾持推得＝門跟隨夾爪）。free 模式開門，到觸發點後切換：
  - **參考凍結**：目標固定為暫停瞬間最後一筆已發布目標（END_HOLD 語意），DoorTargetK3 不再推進；**不是**每輪追隨實測門角。
  - **底盤律（明確的 QP 等式，不是零速度成本）**：
      減速段：u_base（本體）＝把**量測的底盤本體速度**（位姿差分）朝 0 移動一步，每週期線速度 ≤ 0.8·a_base_lin·dt、角速度 ≤ 0.8·a_base_ang·dt
              （留 20% 給 QP 加速度框）；以 cfg.base_hold 等式套入整個時域；
      保持段：量測線速度 ≤ 停車伺服 v_max（5 mm/s）且角速度 ≤ w_max 時，錨點＝當下位姿，改用既有 park_hold_cmd 停車伺服。
  - 手臂持續由 W-GMPC 追凍結目標（吸收底盤剩餘位移）。
**完成判準**（只用量測）：底盤線速度 ≤ 1 mm/s、角速度 ≤ 0.005 rad/s、門角速率 ≤ 0.002 rad/s、手臂套用速率 ≤ 1e-3 rad/s，連續 1.0 s；
  完成時記錄最後套用命令的大小（底盤 ≤ 伺服界、手臂 ≈ 0）⇒ 之後才停止發布，不以 0.2 s 命令逾時空窗當作停止機制。
觸發：P1＝q_ref 到 8° 後 1.0 s（對應 SETTLE→END_HOLD）；P2＝推得門角 ≥ 4°（開門中、底盤仍在動；只作離線驗證）。
逐週期記錄：底盤速度、手臂速率、推得門角、閉合殘差、四類淨空（door_k3m_geom_monitor 外接球後端）、夾爪殼—橫桿與手指—橫桿
  （相對**觸發瞬間名目值**的變化）、套用命令。
**界線**：門沒有動力學（不會自行續轉，只隨夾爪動）；無接觸力、無夾持柔順、無安全層與 E2；不是停止安全證明。

（觀察窗 8 s；先前 5 s 版本輸出保留為 door_k3m_offline_pause_v0_after5s.json——P2 在 5 s 內尚未滿足完成判準。）

    python3 evaluation/door_k3m_offline_pause.py   # 寫 results/door/door_k3m_offline_pause.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_offline_replay as RP                  # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
from park_fixed import HoldServo, park_hold_cmd       # noqa: E402

CL, KC, T3, GA, KP = RP.CL, RP.KC, RP.T3, RP.GA, RP.KP
OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_pause.json')
DT = 0.05
AFTER_S = 8.0
DONE_HOLD_S = 1.0


def run_case(tag, trigger, ident, K, q0, s0, T_hold, anchor0, q_finger, cfgp):
    cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05)
    H0 = KC._H_of(0.0)
    G_T_H = np.linalg.inv(T_hold) @ H0
    a0 = math.atan2(H0[1, 3], H0[0, 3])
    servo = HoldServo()
    a_lin, a_ang = 0.8 * cfg.a_base_lin * cfg.dt, 0.8 * cfg.a_base_ang * cfg.dt
    st = {'dtk': T3.DoorTargetK3(T_hold, H0, cfgp.target_cfg()), 'T': T_hold, 't_reach': None, 'paused': None, 'stage': 'free',
          'anchor': None, 'prev': None, 'log': {}, 'u_hold': None}

    def pre(t, q, cfg_):
        qi = RP.door_angle(q, G_T_H, a0)
        vb = None
        if st['prev'] is not None and t > st['prev'][0]:
            dtt = t - st['prev'][0]
            dx, dy = q[0] - st['prev'][1][0], q[1] - st['prev'][1][1]
            c, s_ = math.cos(q[2]), math.sin(q[2])
            vb = np.array([(c * dx + s_ * dy) / dtt, (-s_ * dx + c * dy) / dtt,
                           math.atan2(math.sin(q[2] - st['prev'][1][2]), math.cos(q[2] - st['prev'][1][2])) / dtt])
        st['prev'] = (t, np.array(q[:3], float))
        if st['paused'] is None:
            T, _ = st['dtk'].step(KC._H_of(qi), qi, DT)
            st['T'] = T
            if st['dtk'].reached and st['t_reach'] is None:
                st['t_reach'] = t
            if trigger(t, qi, st):
                st['paused'] = {'t': t, 'q_door_deg': math.degrees(qi), 'T': st['T'].tolist(), 'base_pose': list(map(float, q[:3]))}
                st['stage'] = 'decel'
        if st['paused'] is not None:
            if st['stage'] == 'decel':
                v = np.zeros(3) if vb is None else vb
                lin = math.hypot(v[0], v[1])
                if lin <= servo.v_max and abs(v[2]) <= servo.w_max:
                    st['stage'] = 'hold'
                    st['anchor'] = tuple(float(x) for x in q[:3])
                else:
                    sc = max(0.0, lin - a_lin) / lin if lin > 1e-12 else 0.0
                    wz = math.copysign(max(0.0, abs(v[2]) - a_ang), v[2])
                    st['u_hold'] = (float(v[0] * sc), float(v[1] * sc), float(wz))
            if st['stage'] == 'hold':
                st['u_hold'] = park_hold_cmd(st['anchor'], [float(x) for x in q[:3]], servo)
            cfg_.base_hold = tuple(st['u_hold'])
        st['log'][round(t, 6)] = (qi, st['stage'], None if vb is None else vb.tolist(), st['u_hold'])

    def T_of(t):
        return st['T']
    t_end = 30.0
    rec = CL.run(cfg, K, q0, s0, T_of, t_end=t_end, reach_p=0.005, reach_r=0.02, hold_s=1e9, pre_solve=pre,
                 delay_cycles=1.4, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=0.0)['rec']
    if st['paused'] is None:
        return {'tag': tag, 'triggered': False}
    tp = st['paused']['t']
    fn = GM.default_backend()
    rows, nom = [], None
    prev_q = None
    done_t, still_since = None, None
    for i, t in enumerate(rec['t']):
        if t < tp - 1e-9 or t > tp + AFTER_S + 1e-9:
            prev_q = np.asarray(rec['q'][i], float)
            continue
        q = np.asarray(rec['q'][i], float)
        qi, stage, vb, uh = st['log'][round(t, 6)]
        ua = np.asarray(rec['u_app'][i], float)
        d = fn(q, qi, q_finger)
        parts, _ = GA.parts_clearance(q, qi, q_finger)
        shell_bar = parts['uflite_gripper_link']['min_to_bar_m']
        finger_bar = min(parts[f]['min_to_bar_m'] for f in GA.FINGERS)
        if nom is None:
            nom = {'shell_bar': shell_bar, 'finger_bar': finger_bar}
        Hi = KC._tcp(list(q[:3]), list(q[3:9])) @ G_T_H
        Hq = KC._H_of(qi)
        qdot = None if prev_q is None else (RP.door_angle(q, G_T_H, a0) - RP.door_angle(prev_q, G_T_H, a0)) / DT
        row = {'t': t, 'stage': stage, 'q_door_deg': math.degrees(qi), 'q_door_dot': qdot, 'v_body': vb, 'u_hold': uh,
               'u_app_base': ua[:3].tolist(), 'u_app_arm_absmax': float(np.abs(ua[3:]).max()),
               'closure_mm': float(np.linalg.norm(Hi[:3, 3] - Hq[:3, 3]) * 1e3),
               **{f'd_{c}': d[c][0] for c in GM.CLASSES}, 'shell_bar_m': shell_bar, 'finger_bar_m': finger_bar,
               'd_shell_bar_vs_nom_mm': (shell_bar - nom['shell_bar']) * 1e3, 'd_finger_bar_vs_nom_mm': (finger_bar - nom['finger_bar']) * 1e3}
        rows.append(row)
        still = (vb is not None and math.hypot(vb[0], vb[1]) <= 0.001 and abs(vb[2]) <= 0.005 and qdot is not None and abs(qdot) <= 0.002
                 and row['u_app_arm_absmax'] <= 1e-3)
        if still:
            still_since = t if still_since is None else still_since
            if done_t is None and t - still_since >= DONE_HOLD_S - 1e-9:
                done_t = t
        else:
            still_since = None
        prev_q = q
    last = rows[-1] if rows else None
    ext = lambda k, f: f(r[k] for r in rows if r[k] is not None)       # noqa: E731
    return {'tag': tag, 'triggered': True, 'pause': st['paused'], 't_hold_stage': next((r['t'] for r in rows if r['stage'] == 'hold'), None),
            'anchor_hold': st['anchor'], 'done_t': done_t, 'done_after_s': None if done_t is None else done_t - tp,
            'base_disp_after_pause_mm': None if not rows else float(np.linalg.norm(np.array(rec['q'][int(np.searchsorted(rec['t'], rows[-1]['t']))][:2])
                                                                                   - np.array(st['paused']['base_pose'][:2])) * 1e3),
            'door_change_deg': None if not rows else rows[-1]['q_door_deg'] - rows[0]['q_door_deg'],
            'min': {k: ext(k, min) for k in ('d_nonfinger', 'd_finger', 'd_base', 'd_self', 'd_shell_bar_vs_nom_mm', 'd_finger_bar_vs_nom_mm')},
            'max_closure_mm': ext('closure_mm', max), 'max_arm_rate': ext('u_app_arm_absmax', max),
            'last_applied': None if last is None else {'base': last['u_app_base'], 'arm_absmax': last['u_app_arm_absmax']},
            'rows': rows}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    ident = json.load(open(RP.IDENT))
    t = json.load(open(os.path.join(RP.RUN, 'task.json')))
    r = json.load(open(os.path.join(RP.RUN, 'room_run.json')))
    gate = json.load(open(os.path.join(RP.RUN, 'park_gate.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    anchor = [float(x) for x in gate['gate']['anchor']]
    T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
    s_open = int(next(e['step'] for e in t['trace'] if e.get('phase') == 'OPEN'))
    so = next(x for x in r['steps'] if int(x[I['step']]) == s_open)
    q0 = np.array(list(so[I['base_xyth']]) + list(so[I['q_arm_meas']]), float)
    s0 = q0[3:9] - np.asarray(ident['bias_rad']) / np.asarray(ident['alpha'])
    K = CL.WholeBodyKinematics.from_urdf_file(RP.C.URDF)
    CL.BASE_GAIN = np.array([0.9, 0.9, 0.82])
    cfgp = KP.K3AConfig()
    cases = [('P1_end', lambda tt, qi, st: st['t_reach'] is not None and tt >= st['t_reach'] + 1.0 - 1e-9),
             ('P2_mid', lambda tt, qi, st: math.degrees(qi) >= 4.0)]
    res = {'schema': 'door_k3m_offline_pause/1', 'source_run': 'door_k3a_ev_1', 'anchor': anchor, 'servo': HoldServo().__dict__,
           'decel_fraction_of_accel_box': 0.8, 'done_criteria': {'v_lin_mps': 0.001, 'w_radps': 0.005, 'door_rate_radps': 0.002,
                                                                  'arm_rate_radps': 1e-3, 'hold_s': DONE_HOLD_S},
           'cases': [run_case(tag, trig, ident, K, q0, s0, T_hold, anchor, q_finger, cfgp) for tag, trig in cases],
           'note': '門沒有動力學（不會自行續轉）、無接觸力、無安全層與 E2；離線閉迴路運動學核對，不是停止安全證明。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for c in res['cases']:
        if not c['triggered']:
            print(c['tag'], '未觸發')
            continue
        m = c['min']
        print(f"== {c['tag']}：暫停 t {c['pause']['t']:.2f} s、門角 {c['pause']['q_door_deg']:.2f}°；進保持段 {c['t_hold_stage']}；完成 {c['done_after_s']} s 後；"
              f"暫停後底盤位移 {c['base_disp_after_pause_mm']:.1f} mm、門角變化 {c['door_change_deg']:+.3f}°")
        print(f"   淨空最小：非手指 {m['d_nonfinger'] * 1e3:.1f}、手指 {m['d_finger'] * 1e3:.1f}、底盤 {m['d_base'] * 1e3:.0f}、自碰 {m['d_self'] * 1e3:.1f} mm；"
              f"殼—桿相對名目 {m['d_shell_bar_vs_nom_mm']:+.2f} mm、指—桿相對名目 {m['d_finger_bar_vs_nom_mm']:+.2f} mm；"
              f"閉合最大 {c['max_closure_mm']:.2f} mm；末筆套用 {c['last_applied']}")


if __name__ == '__main__':
    main()
