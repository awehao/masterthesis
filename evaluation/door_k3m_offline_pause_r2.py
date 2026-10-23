#!/usr/bin/env python3
"""K3-M 受控暫停離線閉迴路核對 r2（依 Codex 20261023_223218 第 2 節三項必修；純函式見 door_k3m_pause.py）。r1（door_k3m_offline_pause.py）與其輸出不改。

受控體與控制器同 r1（wgmpc_closed_loop_sim.run、solve_sp、free4 手臂模型、延遲 1.4 週期、補償 1.6／1.0；門跟隨夾爪，沒有門動力學）。r2 的差異：
  ① 參考：每週期已發布目標記入 PauseReference（seq, t, T）；觸發當輪**不再推進** DoorTargetK3，凍結＝上一輪已發布目標，
     並核對凍結後使用的目標與該筆逐元素相同。
     P1 觸發改為正式 SETTLE 條件（|q − 8°| ≤ 0.5°、|q̇| ≤ 0.002 rad/s，連續 1.0 s；本工具沒有漂移與接觸量，這兩項不在核對內）。
  ② 底盤：減速段與保持段的 cfg.base_hold 都經 decel_step，起點為**上一物理步實際套用**的底盤命令（包住 plant_step 取得，不改受控體），
     逐週期核對 feasible(u_prev_applied, base_hold)。保持段目標＝park_hold_cmd（停車伺服），同樣由 applied 承接。
  ③ 完成：PauseSequencer（LOWSPEED→ZERO_PUB→CONFIRM→STOPPED→DONE）；每週期把**本輪求解是否成功**交給序列器（包住 solve_sp）。
     ZERO_PUB 起發布值改為「底盤 decel_step 朝 0、手臂 0」；CONFIRM 用每物理步的實際套用值與來源序號
     （離線以數值相等反查最近一筆已發布命令；線上須由執行端回報來源序號）；STOPPED 後視為停止發布（套用值保持 0，等同命令逾時後的 0），觀察 1 s。
     報告取**各階段轉換當下**的量（不取觀察窗末）。
界線：門沒有動力學、無接觸力、無安全層與 E2；P2 只作離線驗證。不是停止安全證明。

    python3 evaluation/door_k3m_offline_pause_r2.py   # 寫 results/door/door_k3m_offline_pause_r2.json（已存在則拒絕覆寫）
"""
import dataclasses
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_offline_replay as RP                  # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_pause as PZ                            # noqa: E402
from park_fixed import HoldServo, park_hold_cmd       # noqa: E402

CL, KC, T3, GA, KP = RP.CL, RP.KC, RP.T3, RP.GA, RP.KP
OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_pause_r2.json')
DT = 0.05
AFTER_S = 12.0
TIMEOUT_S = 8.0                                                           # 規格 §3 離線建議期限


def run_case(tag, trigger, ident, K, q0, s0, T_hold, q_finger, cfgp):
    cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05)
    H0 = KC._H_of(0.0)
    G_T_H = np.linalg.inv(T_hold) @ H0
    a0 = math.atan2(H0[1, 3], H0[0, 3])
    servo = HoldServo()
    lim = PZ.BaseLimits(a_lin=cfg.a_base_lin, a_ang=cfg.a_base_ang, dt=cfg.dt)
    ref = PZ.PauseReference()
    seq_ = PZ.PauseSequencer(PZ.PauseCfg(timeout_s=TIMEOUT_S))
    st = {'dtk': T3.DoorTargetK3(T_hold, H0, cfgp.target_cfg()), 'T': T_hold, 'seq': 0, 'paused': None, 'stage': 'free',
          'anchor': None, 'prev': None, 'qprev': None, 'settle_t0': None, 'reached': False, 'u_hold': None,
          'applied': np.zeros(9), 'pubs': [], 'phys': [], 'cyc': [], 'infeasible': 0, 'freeze_mismatch': 0, 'zero_mode': False,
          'm': None, 'vb': None}

    def pre(t, q, cfg_):
        qi = RP.door_angle(q, G_T_H, a0)
        vb, qdot = None, None
        if st['prev'] is not None and t > st['prev'][0]:
            dtt = t - st['prev'][0]
            dx, dy = q[0] - st['prev'][1][0], q[1] - st['prev'][1][1]
            c, s_ = math.cos(q[2]), math.sin(q[2])
            vb = np.array([(c * dx + s_ * dy) / dtt, (-s_ * dx + c * dy) / dtt,
                           math.atan2(math.sin(q[2] - st['prev'][1][2]), math.cos(q[2] - st['prev'][1][2])) / dtt])
            qdot = (qi - st['qprev']) / dtt
        st['prev'], st['qprev'], st['vb'] = (t, np.array(q[:3], float)), qi, vb
        if st['paused'] is None:
            fire = trigger(t, qi, qdot, st)
            if fire:
                info = ref.freeze()                                       # 凍結＝上一輪已發布目標；本輪不推進
                st['paused'] = {'t': t, 'q_door_deg': math.degrees(qi), 'frozen': info, 'base_pose': list(map(float, q[:3]))}
                st['stage'] = 'decel'
                st['T'] = ref.target()
            else:
                T, _ = st['dtk'].step(KC._H_of(qi), qi, DT)
                st['T'] = T
                st['reached'] = st['reached'] or bool(st['dtk'].reached)
        if st['paused'] is not None:
            if not np.array_equal(st['T'], st['pubs'][st['paused']['frozen']['seq'] - 1][2]):
                st['freeze_mismatch'] += 1
            up = st['applied'][:3]
            v = np.zeros(3) if vb is None else vb
            if st['stage'] == 'decel' and math.hypot(v[0], v[1]) <= servo.v_max and abs(v[2]) <= servo.w_max:
                st['stage'], st['anchor'] = 'hold', tuple(float(x) for x in q[:3])
            if st['zero_mode']:
                goal = np.zeros(3)
            elif st['stage'] == 'hold':
                goal = np.asarray(park_hold_cmd(st['anchor'], [float(x) for x in q[:3]], servo), float)
            else:
                goal = np.zeros(3)
            uh = PZ.decel_step(up, goal, lim)
            if not PZ.feasible(up, uh, lim):
                st['infeasible'] += 1
            st['u_hold'] = uh
            cfg_.base_hold = tuple(float(x) for x in uh)
        st['m'] = {'t': t, 'qi': qi, 'qdot': qdot, 'vb': None if vb is None else vb.tolist(), 'stage': st['stage']}

    solve0, plant0 = CL.S.solve_sp, CL.plant_step

    def solve_w(*a, **kw):
        r = solve0(*a, **kw)
        m = st['m']
        pub, zero = True, False
        if st['paused'] is not None and seq_.state not in ('DONE', 'FAIL'):
            arm_rate = float(np.abs(st['applied'][3:]).max())
            meas = {'v_body': m['vb'] if m['vb'] is not None else [float('nan')] * 3,
                    'door_rate': m['qdot'] if m['qdot'] is not None else float('nan'), 'arm_rate_applied': arm_rate}
            act = seq_.cycle(m['t'], meas, bool(r.ok), base_cmd_zero=bool(st['zero_mode'] and not np.any(st['u_hold'])))
            pub, zero = act['publish'], act['zero']
            if act['zero'] and not st['zero_mode']:
                st['zero_mode'] = True                                    # 下一輪起底盤目標改 0；本輪手臂即 0
            if act['zero'] or not pub:
                u0 = np.zeros(9)
                u0[:3] = st['u_hold'] if pub else 0.0
                r = dataclasses.replace(r, u0=u0)
        elif st['paused'] is not None:
            r = dataclasses.replace(r, u0=np.zeros(9))
            pub = False
        st['seq'] += 1
        u_out = np.asarray(r.u0 if r.ok else np.zeros(9), float).copy()
        if pub:
            st['pubs'].append((st['seq'], m['t'], st['T'].copy(), u_out))
            ref.published(st['seq'], m['t'], st['T'])
            seq_.note_published(st['seq'], zero)
        st['cyc'].append({'t': m['t'], 'ok': bool(r.ok), 'pub': pub, 'state': seq_.state if st['paused'] is not None else 'pre',
                          'stage': m['stage'], 'q_door_deg': math.degrees(m['qi']), 'qdot': m['qdot'], 'vb': m['vb'],
                          'base_hold': None if st['u_hold'] is None else st['u_hold'].tolist(), 'seq': st['seq']})
        return r

    def plant_w(q, s, u, cfg_, n):
        st['applied'] = np.asarray(u, float).copy()
        if st['paused'] is not None:
            src = next((p[0] for p in reversed(st['pubs']) if np.array_equal(p[3], st['applied'])), None)
            st['phys'].append((st['m']['t'], st['applied'].tolist(), src))
            if seq_.state == 'CONFIRM':
                seq_.applied(st['m']['t'], st['applied'][:3].tolist(), st['applied'][3:].tolist(), src)
        return plant0(q, s, u, cfg_, n)

    CL.S.solve_sp, CL.plant_step = solve_w, plant_w
    try:
        rec = CL.run(cfg, K, q0, s0, lambda t: st['T'], t_end=30.0 + AFTER_S, reach_p=0.005, reach_r=0.02, hold_s=1e9, pre_solve=pre,
                     delay_cycles=1.4, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=0.0)['rec']
    finally:
        CL.S.solve_sp, CL.plant_step = solve0, plant0
    if st['paused'] is None:
        return {'tag': tag, 'triggered': False}
    tp = st['paused']['t']
    fn = GM.default_backend()
    rows, nom = [], None
    for i, t in enumerate(rec['t']):
        if t < tp - 1e-9 or t > tp + AFTER_S + 1e-9:
            continue
        q = np.asarray(rec['q'][i], float)
        c = next(x for x in st['cyc'] if abs(x['t'] - t) < 1e-9)
        qi = math.radians(c['q_door_deg'])
        d = fn(q, qi, q_finger)
        parts, _ = GA.parts_clearance(q, qi, q_finger)
        sb, fb = parts['uflite_gripper_link']['min_to_bar_m'], min(parts[f]['min_to_bar_m'] for f in GA.FINGERS)
        nom = nom or {'sb': sb, 'fb': fb}
        Hi = KC._tcp(list(q[:3]), list(q[3:9])) @ G_T_H
        ua = np.asarray(rec['u_app'][i], float)
        rows.append({**c, 'base_disp_mm': float(np.linalg.norm(q[:2] - np.array(st['paused']['base_pose'][:2])) * 1e3),
                     'u_app_base': ua[:3].tolist(), 'u_app_arm_absmax': float(np.abs(ua[3:]).max()),
                     'closure_mm': float(np.linalg.norm(Hi[:3, 3] - KC._H_of(qi)[:3, 3]) * 1e3),
                     **{f'd_{k}': d[k][0] for k in GM.CLASSES}, 'd_shell_bar_vs_nom_mm': (sb - nom['sb']) * 1e3,
                     'd_finger_bar_vs_nom_mm': (fb - nom['fb']) * 1e3})

    def at(tt):
        if tt is None:
            return None
        r_ = min(rows, key=lambda x: abs(x['t'] - tt))
        ph = [p for p in st['phys'] if p[0] <= tt + 1e-9]
        return {'t': r_['t'], 'after_pause_s': r_['t'] - tp, 'base_disp_mm': r_['base_disp_mm'], 'q_door_deg': r_['q_door_deg'],
                'door_change_deg': r_['q_door_deg'] - st['paused']['q_door_deg'], 'vb': r_['vb'], 'qdot': r_['qdot'],
                'last_applied_base': None if not ph else ph[-1][1][:3], 'last_applied_arm_absmax': None if not ph else max(abs(x) for x in ph[-1][1][3:]),
                'last_applied_src_seq': None if not ph else ph[-1][2]}
    trans = {s_: next((x[0] for x in seq_.log if x[1] == s_), None) for s_ in ('ZERO_PUB', 'CONFIRM', 'STOPPED', 'DONE', 'FAIL')}
    ext = lambda k, f: f(r[k] for r in rows if r.get(k) is not None)       # noqa: E731
    return {'tag': tag, 'triggered': True, 'pause': st['paused'], 'freeze_mismatch_cycles': st['freeze_mismatch'],
            'infeasible_base_hold_cycles': st['infeasible'], 'solve_fail_cycles_after_pause': sum(1 for x in rows if not x['ok']),
            't_hold_stage': next((r['t'] for r in rows if r['stage'] == 'hold'), None),
            'final_state': seq_.state, 'fail_why': seq_.why, 'zero_seq0': seq_.zero_seq0, 'transitions': {k: at(v) for k, v in trans.items()},
            'min': {k: ext(k, min) for k in ('d_nonfinger', 'd_finger', 'd_base', 'd_self', 'd_shell_bar_vs_nom_mm', 'd_finger_bar_vs_nom_mm')},
            'max_closure_mm': ext('closure_mm', max), 'max_base_disp_mm': ext('base_disp_mm', max), 'rows': rows}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    ident = json.load(open(RP.IDENT))
    t = json.load(open(os.path.join(RP.RUN, 'task.json')))
    r = json.load(open(os.path.join(RP.RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
    s_open = int(next(e['step'] for e in t['trace'] if e.get('phase') == 'OPEN'))
    so = next(x for x in r['steps'] if int(x[I['step']]) == s_open)
    q0 = np.array(list(so[I['base_xyth']]) + list(so[I['q_arm_meas']]), float)
    s0 = q0[3:9] - np.asarray(ident['bias_rad']) / np.asarray(ident['alpha'])
    K = CL.WholeBodyKinematics.from_urdf_file(RP.C.URDF)
    CL.BASE_GAIN = np.array([0.9, 0.9, 0.82])
    cfgp = KP.K3AConfig()
    goal = cfgp.target_cfg().theta_goal_rad

    def p1(tt, qi, qdot, st):                                             # 正式 SETTLE 條件（無漂移與接觸量）連續 1.0 s
        if not st['reached'] or qdot is None:
            return False
        ok = abs(qi - goal) <= math.radians(cfgp.settle_tol_deg) and abs(qdot) <= cfgp.settle_rate
        st['settle_t0'] = (st['settle_t0'] if st['settle_t0'] is not None else tt) if ok else None
        return ok and tt - st['settle_t0'] >= cfgp.settle_hold_s - 1e-9
    cases = [('P1_settle', p1), ('P2_mid', lambda tt, qi, qdot, st: math.degrees(qi) >= 4.0)]
    res = {'schema': 'door_k3m_offline_pause/2', 'source_run': 'door_k3a_ev_1', 'servo': HoldServo().__dict__,
           'limits': dataclasses.asdict(PZ.BaseLimits()), 'pause_cfg': dataclasses.asdict(PZ.PauseCfg(timeout_s=TIMEOUT_S)),
           'cases': [run_case(tag, trig, ident, K, q0, s0, T_hold, q_finger, cfgp) for tag, trig in cases],
           'note': '門沒有動力學、無接觸力、無安全層與 E2；來源序號以數值相等反查（離線近似）。離線閉迴路運動學核對，不是停止安全證明。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for c in res['cases']:
        if not c['triggered']:
            print(c['tag'], '未觸發')
            continue
        m, tr = c['min'], c['transitions']
        print(f"== {c['tag']}：暫停 t {c['pause']['t']:.2f} s、門角 {c['pause']['q_door_deg']:.3f}°、凍結 seq {c['pause']['frozen']['seq']}；"
              f"最終 {c['final_state']}（{c['fail_why']}）；凍結不符 {c['freeze_mismatch_cycles']}、不可行 {c['infeasible_base_hold_cycles']}、"
              f"求解失敗 {c['solve_fail_cycles_after_pause']}；保持段 {c['t_hold_stage']}")
        for k, v in tr.items():
            if v:
                print(f"   {k}: +{v['after_pause_s']:.2f} s 底盤 {v['base_disp_mm']:.2f} mm、門變化 {v['door_change_deg']:+.3f}°、"
                      f"套用底盤 {np.round(v['last_applied_base'], 9).tolist()}、手臂 {v['last_applied_arm_absmax']:.2e}、來源 seq {v['last_applied_src_seq']}")
        print(f"   淨空最小 非手指 {m['d_nonfinger'] * 1e3:.1f}／手指 {m['d_finger'] * 1e3:.1f}／底盤 {m['d_base'] * 1e3:.0f}／自碰 {m['d_self'] * 1e3:.1f} mm；"
              f"殼—桿 {m['d_shell_bar_vs_nom_mm']:+.2f}、指—桿 {m['d_finger_bar_vs_nom_mm']:+.2f} mm；閉合 {c['max_closure_mm']:.2f} mm")


if __name__ == '__main__':
    main()
