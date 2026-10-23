#!/usr/bin/env python3
"""K3-M 前置離線核對：底盤可動的小角度協同開門，W-GMPC 閉迴路運動學重播（無 Isaac、無接觸；依 Codex 20261023_200048）。

受控體與控制器沿用 `wgmpc_closed_loop_sim.run`（手臂設定點模型＝free4 辨識 α／bias；底盤 q⁺＝q＋dt·B(θ)u，
命令→實測增益 0.9／0.9／0.82 同 motm_offline；控制器＝`wgmpc_core_sp.solve_sp` 本體，20 Hz）。
延遲：端到端 1.4 週期（發布 0.6）、補償 state 1.6／cmd 1.0、發布歷史查表——同 K3-A 跑法的求解器參數。
權重：N 5、w_s 0.001、w_a 0.05、w_s_arm 0.05（同 run_door_k3a*.sh）；其餘核心預設。

**門跟隨夾爪（剛性夾持）**：每輪由實際 TCP 經夾持關係 G_T_H＝T_hold⁻¹·H(0) 推得把手位姿，取其繞鉸鏈（原點、−z）的轉角為門角 q_impl；
DoorTargetK3（R∩L∩G，首步不推進）以 q_impl 為實測門角產生下一輪目標（任務端到求解端一個控制週期的落後），到 8° 後保持 4 s
（SETTLE 1 s＋END_HOLD 3 s 的時長）。淨空以 q_impl 擺門（門被夾爪帶動，不會跑在夾爪前面）。
（初版用「門理想跟隨 q_ref、與夾爪無關」會在手臂落後時讓門扇穿入夾爪，是建模假象，已改；初版輸出保留為 door_k3m_offline_replay_v0_ideal_door.json，不採用。）
兩組設定（只差底盤與近目標整形，其餘相同）：
  fixed：base_fixed（整個時域 u_base＝0，近似 PARK_HOLD 錨點誤差 0 的情形）、近目標整形停用（同 PARK_HOLD 實作）；
  free ：底盤三軸由 W-GMPC 共同決定（無 base_fixed／base_hold）、近目標整形 γ＝1.0（同跑法 --near-target-gamma 1.0）。
每輪紀錄：底盤相對錨點位移（錨點座標：前進＝錨點 yaw 方向）、偏航、底盤命令與套用速度、手臂關節餘裕、相鄰輪關節變化、TCP 追蹤誤差、求解成功。
每 4 輪（0.2 s）做一次淨空：點雲／盒／膠囊近似，非手指剛體（含相機支架、夾爪殼）對門扇與門框、手指對門板與立柱、底盤圓盤對門扇／門框。

**界線**：沒有接觸力、沒有夾持約束、沒有 Isaac 積分器細節、沒有安全層；門角是理想跟隨，不是被動接觸產生。
結果只用來篩選規格的底盤包絡、速度需求與淨空，不代表 Isaac 會收斂，也不預測卸載是否改善。

    python3 evaluation/door_k3m_offline_replay.py   # 寫 results/door/door_k3m_offline_replay.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wgmpc_closed_loop_sim as CL                    # noqa: E402
import door_k3au_offline_geom as GA                   # noqa: E402
import door_k3ax_offline_geom as GX                   # noqa: E402
import door_k2_check as KC                            # noqa: E402
import door_k3a_policy as KP                          # noqa: E402
import door_target_k3 as T3                           # noqa: E402

S = CL.S
C = GA.C
OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay.json')
RUN = os.path.join(HERE, 'runs', 'door_k3a_ev_1')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
DT = 0.05
HOLD_AFTER_S = 4.0
CLEAR_EVERY = 4


def reference(T_hold, cfg):
    dtk = T3.DoorTargetK3(T_hold, KC._H_of(0.0), cfg.target_cfg())
    ref, q_meas = [], 0.0
    while True:
        Tg, _ = dtk.step(KC._H_of(q_meas), q_meas, DT)
        ref.append((len(ref) * DT, dtk.q_ref, Tg))
        q_meas = dtk.q_ref
        if dtk.reached:
            break
    return ref


def door_angle(q, G_T_H, a0):
    H = KC._tcp(list(q[:3]), list(q[3:9])) @ G_T_H
    d = math.atan2(H[1, 3], H[0, 3]) - a0
    return -math.atan2(math.sin(d), math.cos(d))           # 鉸鏈軸 −z：q > 0 ＝ 繞 −z


def one(name, base_fixed, gamma, ident, K, q0, s0, ref, q_finger, anchor, T_hold, cfgp):
    cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05, base_fixed=base_fixed)
    H0 = KC._H_of(0.0)
    G_T_H = np.linalg.inv(T_hold) @ H0
    a0 = math.atan2(H0[1, 3], H0[0, 3])
    st = {'dtk': T3.DoorTargetK3(T_hold, H0, cfgp.target_cfg()), 'T': T_hold, 'log': {}, 't_reach': None, 'conflict': None}

    def pre(t, q, _cfg):
        qi = door_angle(q, G_T_H, a0)
        try:
            T, _ = st['dtk'].step(KC._H_of(qi), qi, DT)
            st['T'] = T
        except T3.DoorTargetK3Conflict as e:
            st['conflict'] = st['conflict'] or {'t': t, 'why': str(e)}
        if st['dtk'].reached and st['t_reach'] is None:
            st['t_reach'] = t
        st['log'][round(t, 6)] = (qi, st['dtk'].q_ref)

    def T_of(t):
        return st['T']                                      # 上一輪 pre 算出的目標（一個控制週期落後）
    t_end = ref[-1][0] + 3.0 + HOLD_AFTER_S
    rec = CL.run(cfg, K, q0, s0, T_of, t_end=t_end, reach_p=-1.0, reach_r=-1.0, hold_s=1e9, pre_solve=pre,
                 delay_cycles=1.4, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=gamma)['rec']
    ax, ay, ath = anchor
    ca, sa = math.cos(ath), math.sin(ath)
    rows, prev = [], None
    for i, t in enumerate(rec['t']):
        q = np.asarray(rec['q'][i], float)
        qd, qr = st['log'].get(round(t, 6), (door_angle(q, G_T_H, a0), None))
        dx, dy = q[0] - ax, q[1] - ay
        fwd, lat = ca * dx + sa * dy, -sa * dx + ca * dy
        ua = np.asarray(rec['u_app'][i], float)
        row = {'t': t, 'q_door_deg': math.degrees(qd), 'q_ref_deg': None if qr is None else math.degrees(qr),
               'base_fwd_m': fwd, 'base_lat_m': lat,
               'base_dyaw_deg': math.degrees(math.atan2(math.sin(q[2] - ath), math.cos(q[2] - ath))),
               'u_app_base': ua[:3].tolist(), 'u_app_arm_max': float(np.abs(ua[3:]).max()),
               'joint_margin_rad': float(C.jm(q[3:9])[0]), 'ep_m': rec['ep'][i], 'er_rad': rec['er'][i], 'ok': rec['ok'][i],
               'dq_arm_max_rad': None if prev is None else float(np.abs(q[3:9] - prev[3:9]).max())}
        if i % CLEAR_EVERY == 0:
            parts, bc = GA.parts_clearance(q, qd, q_finger)
            nf = {n: v for n, v in parts.items() if n not in GA.FINGERS}
            row['non_finger_env_min_m'] = min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for v in nf.values())
            row['non_finger_env_who'] = min(nf.items(), key=lambda kv: min(kv[1]['min_to_door_parts_m'], kv[1]['min_to_bar_m']))[0]
            row['fingers_to_door_parts_min_m'] = min(parts[f]['min_to_door_parts_m'] for f in GA.FINGERS)
            row['to_posts_min_m'] = min(GX.post_dist(q, qd, q_finger, ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link',
                                                                        'link_eef')).values())
            row['base_clear_m'] = bc
        rows.append(row)
        prev = q

    def ext(key, f=max, absval=False):
        v = [abs(r[key]) if absval else r[key] for r in rows if r.get(key) is not None]
        return None if not v else f(v)
    summ = {'n_cycles': len(rows), 'n_solve_fail': sum(1 for r in rows if not r['ok']), 't_reach_s': st['t_reach'],
            'target_conflict': st['conflict'], 'q_door_final_deg': rows[-1]['q_door_deg'],
            'lead_absmax_deg': max([abs(r['q_ref_deg'] - r['q_door_deg']) for r in rows if r['q_ref_deg'] is not None] or [0.0]),
            'base_fwd_range_m': [ext('base_fwd_m', min), ext('base_fwd_m', max)],
            'base_lat_range_m': [ext('base_lat_m', min), ext('base_lat_m', max)],
            'base_disp_max_m': max(math.hypot(r['base_fwd_m'], r['base_lat_m']) for r in rows),
            'base_dyaw_absmax_deg': ext('base_dyaw_deg', absval=True),
            'u_base_lin_max': max(math.hypot(r['u_app_base'][0], r['u_app_base'][1]) for r in rows),
            'u_base_ang_max': max(abs(r['u_app_base'][2]) for r in rows),
            'u_arm_max': ext('u_app_arm_max'), 'joint_margin_min_rad': ext('joint_margin_rad', min),
            'dq_arm_step_max_rad': ext('dq_arm_max_rad'),
            'ep_max_m': ext('ep_m'), 'er_max_rad': ext('er_rad'),
            'ep_final_m': rows[-1]['ep_m'], 'er_final_rad': rows[-1]['er_rad'],
            'non_finger_env_min_m': ext('non_finger_env_min_m', min), 'fingers_to_door_parts_min_m': ext('fingers_to_door_parts_min_m', min),
            'to_posts_min_m': ext('to_posts_min_m', min), 'base_clear_min_m': ext('base_clear_m', min),
            'limits': {'v_base_lin': cfg.v_base_lin, 'v_base_ang': cfg.v_base_ang, 'v_arm': cfg.v_arm}}
    summ['ok_clearance'] = (summ['non_finger_env_min_m'] >= GA.ENV_M and summ['fingers_to_door_parts_min_m'] >= GA.ENV_M
                            and summ['to_posts_min_m'] >= GA.ENV_M and summ['base_clear_min_m'] >= GA.BASE_M)
    return {'name': name, 'base_fixed': base_fixed, 'near_target_gamma': gamma, 'summary': summ, 'rows': rows}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    ident = json.load(open(IDENT))
    t = json.load(open(os.path.join(RUN, 'task.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    gate = json.load(open(os.path.join(RUN, 'park_gate.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    anchor = [float(x) for x in gate['gate']['anchor']]
    T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
    q2 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR']
    qf = C.ik(np.array(anchor), T_hold, np.array(q2))
    if qf is None:
        raise SystemExit('HOLD 目標 IK 失敗')
    q0 = np.asarray(qf, float)
    alpha, bias = np.asarray(ident['alpha'], float), np.asarray(ident['bias_rad'], float)
    s0 = q0[3:9] - bias / alpha                                  # 模型平衡：x＝s＋b/α
    K = CL.WholeBodyKinematics.from_urdf_file(C.URDF)
    CL.BASE_GAIN = np.array([0.9, 0.9, 0.82])
    cfgp = KP.K3AConfig()
    ref = reference(T_hold, cfgp)
    res = {'schema': 'door_k3m_offline_replay/1', 'source_run': 'door_k3a_ev_1', 'anchor': anchor, 'q0': q0.tolist(),
           'reference': {'n': len(ref), 't_end_s': ref[-1][0], 'q_end_deg': math.degrees(ref[-1][1]), 'hold_after_s': HOLD_AFTER_S},
           'delay': {'delay_cycles': 1.4, 'd_pub_cycles': 0.6, 'comp_state': 1.6, 'comp_cmd': 1.0, 'comp_mode': 'hist'},
           'base_gain': CL.BASE_GAIN.tolist(), 'weights': {'N': 5, 'w_s': 0.001, 'w_a': 0.05, 'w_s_arm': 0.05},
           'thresholds': {'env_m': GA.ENV_M, 'base_m': GA.BASE_M}, 'configs': [],
           'note': '無接觸、門角理想跟隨、無安全層的運動學閉迴路重播；只支援規格的底盤包絡、速度需求與取樣淨空，不是碰撞安全或收斂證明，'
                   '也不預測卸載。'}
    for name, bf, g in (('fixed', True, 0.0), ('free', False, 1.0)):
        res['configs'].append(one(name, bf, g, ident, K, q0, s0, ref, q_finger, anchor, T_hold, cfgp))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for c in res['configs']:
        s = c['summary']
        print(f"== {c['name']}：{s['n_cycles']} 輪、失敗 {s['n_solve_fail']}、到達 {s['t_reach_s']} s、末門角 {s['q_door_final_deg']:.3f}°、"
              f"最大超前 {s['lead_absmax_deg']:.3f}°、衝突 {s['target_conflict']}")
        print(f"   底盤前進 {np.round(np.array(s['base_fwd_range_m']) * 1e3, 1)} mm、"
              f"側向 {np.round(np.array(s['base_lat_range_m']) * 1e3, 1)} mm、最大位移 {s['base_disp_max_m'] * 1e3:.1f} mm、偏航 {s['base_dyaw_absmax_deg']:.2f}°")
        print(f"   底盤速度 lin {s['u_base_lin_max']:.4f}／{s['limits']['v_base_lin']:.4f} m/s、ang {s['u_base_ang_max']:.4f}；手臂 {s['u_arm_max']:.3f} rad/s；"
              f"關節餘裕最小 {s['joint_margin_min_rad']:.3f}；每輪關節變化最大 {s['dq_arm_step_max_rad']:.4f} rad")
        print(f"   追蹤 ep 最大 {s['ep_max_m'] * 1e3:.2f} mm／末 {s['ep_final_m'] * 1e3:.2f}、er 最大 {math.degrees(s['er_max_rad']):.3f}°；淨空：非手指 "
              f"{s['non_finger_env_min_m'] * 1e3:.1f}、手指對門扇 {s['fingers_to_door_parts_min_m'] * 1e3:.1f}、立柱 {s['to_posts_min_m'] * 1e3:.1f}、"
              f"底盤 {s['base_clear_min_m'] * 1e3:.0f} mm；ok {s['ok_clearance']}")


if __name__ == '__main__':
    main()
