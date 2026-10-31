#!/usr/bin/env python3
"""瓶子移動夾取 (2c)(2d)：運動學離線閉迴路（無接觸、非 Isaac）。Codex 20261031 有條件核可 (2)。
受控對象：wgmpc_core_sp.plant_phys_step（手臂設定點模型 α、偏差；底盤運動學），逐物理步 0.01 s；
  延遲：命令算出後 8 個物理步（發布 0.6＋命令 1.0 週期）才生效；控制器以在途命令前推 8 步（pipe 補償）。
控制器 C3：wgmpc_core_sp_fr（時域內未來參考 T_seq／v_ref 序列；bottle_ref 產生），權重 CRUISE／CARRY 同 rt_43 R3（w_p 50、w_vref 1、w_a 0.05），
  PREGRASP～LIFT 改 w_p 200（GRASP_W），
  base_box_scale 2.0；外層進度協調（10 Hz）：底盤速度 v_b＝min(v_c, (窗末 − x)／(到閉爪剩餘時間＋T_need))，
  T_need＝閉爪 0.5＋證據 1.0＋抬升 1.0 s；v_b < v_min ⇒ 記為降級（允許減到 0，不剔除）。
消融 C3−prog：v_b＝v_c 定速。
抓持**代理**：閉爪 0.5 s 後，TCP 對抓取位姿連續 1.0 s 在 3 mm／2° 內 ⇒ grasp_ok（無接觸模型：只驗追蹤與候選可行性，不能驗抓持成功）；
  CLOSE 起 5 s 內未成立 ⇒ 失敗；LIFT 前底盤越過可達窗末 ⇒ 失敗（window_passed）。
執行擾動：plant α×0.6；mismatch＝控制器仍用名目 α（未知失配），known＝控制器也用 α×0.6（已知新參數）。
    python3 evaluation/bottle_offline.py [--cases quick|all]
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                     # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp_fr as FR                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import bottle_reach_scan as RS                                        # noqa: E402

K = RS.K
IDENT = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
DT, PHYS = 0.05, 0.01
NPH = int(round(DT / PHYS))
D_STEPS = 8
SCN = dict(xb=0.0, yb=0.42, h=0.70, x_start=-1.30, x_end=0.60, win=(-0.35, 0.15))
T_CLOSE, T_EV, T_LIFT, DEADLINE = 0.5, 1.0, 1.0, 5.0
TOL_EV = (0.003, math.radians(2.0))
V_MIN = 0.01
GRASP_W = (200.0, 1.0)     # PREGRASP～LIFT 權重（w_p, w_vref）；10/31 離線比較 50／1 vs 200／1 選定（同返程 S8 的 w_p 200），各組須同調參預算後凍結


def make_cfg(alpha_scale=1.0):
    am = S.ArmSetpointModel(alpha=np.asarray(IDENT['alpha'], float) * alpha_scale, bias=np.asarray(IDENT['bias_rad'], float),
                            phys_dt=IDENT['phys_dt_measured_s'])
    c = S.WGMPCConfigSP(N=5, dt=DT, arm_model=am, w_s=1e-3, w_a=0.05, w_s_arm=0.05, row_scaling=True)
    c.w_p, c.w_vref, c.base_box_scale = 50.0, 1.0, 2.0
    return c


def ready_posture():
    """備妥：TCP 在底盤座標 (0.05, 0.22, 0.78)、與抓取同姿態；以 IK 求手臂角。"""
    Tg = BR.grasp_pose(0.0, 0.0, 0.0)
    T = np.eye(4)
    T[:3, :3] = Tg[:3, :3]
    T[:3, 3] = [0.05, 0.22, 0.78]
    g = RS.ik(np.zeros(3), T, RS.SEEDS)
    if not g[0]:
        raise SystemExit('備妥姿態 IK 失敗')
    return g[2], T


def run(v_c, coord=True, plant_alpha=1.0, ctrl_alpha=1.0, t_max=80.0, label='', grasp_w=None):
    """grasp_w＝(w_p, w_vref)：PREGRASP～LIFT 相位改用的權重（None＝全程 R3 權重 50／1）。"""
    cfg_p = make_cfg(plant_alpha)
    cfg_c = make_cfg(ctrl_alpha)
    q_ready, T_ready_rel = ready_posture()
    task = BR.BottleTask(T_grasp=BR.grasp_pose(SCN['xb'], SCN['yb'], SCN['h']), T_ready_rel=T_ready_rel)
    x_we = SCN['xb'] + SCN['win'][1]
    x_ws = SCN['xb'] + SCN['win'][0]
    # 預抓到達時刻：底盤進入可達窗 0.05 m 的時刻（窗外預抓位姿不可達；10/31 首版誤設在窗前 0.10 m），且不短於手臂最小平滑時間
    base_at_pre = np.array([x_ws + 0.05, 0.0, 0.0])
    gp = RS.ik(base_at_pre, task.T_pre, [q_ready] + RS.SEEDS)
    t_arm_min = 1.875 * float(np.abs(np.asarray(gp[2]) - q_ready).max()) / 0.5 + 0.3
    t_pre = max((base_at_pre[0] - SCN['x_start']) / v_c, t_arm_min)
    st = BR.PhaseState(t_cruise0=0.0, t_pre=t_pre)
    q = np.r_[SCN['x_start'], 0.0, 0.0, q_ready]
    s = q_ready.copy()
    pipe = [np.zeros(9)] * D_STEPS
    u_prev = np.zeros(9)
    U_warm = None
    v_b = v_c
    rec = dict(t=[], x=[], vx=[], phase=[], ep=[], er=[], v_b=[], fail=0, solve_ms=[])
    t = 0.0
    ev_t0 = None
    close_t0 = None
    result, why = None, None
    degraded = False
    while t < t_max:
        k = int(round(t / DT))
        Tm = K.fk(q, 'link_tcp')
        # 相位推進（事件）
        ev_ok = False
        if st.phase == 'CLOSE':
            close_t0 = st.t_phase if close_t0 is None else close_t0
            ep, er = BR.pose_err(Tm, task.T_grasp)
            if t - close_t0 >= T_CLOSE and ep <= TOL_EV[0] and er <= TOL_EV[1]:
                ev_t0 = t if ev_t0 is None else ev_t0
                ev_ok = (t - ev_t0) >= T_EV
            else:
                ev_t0 = None
            if t - close_t0 > DEADLINE and not ev_ok:
                result, why = 'fail', 'grasp_deadline'
                break
        ph0 = st.phase
        BR.step_phase(task, st, t, Tm, grasp_evidence_ok=ev_ok, close_done=(st.phase == 'CLOSE' and t - st.t_phase >= T_CLOSE),
                      base_meas=q[:3])
        if st.phase != ph0 and st.phase == 'CARRY':
            pass
        if BR.PHASES.index(st.phase) < BR.PHASES.index('LIFT') and q[0] > x_we:
            result, why = 'fail', 'window_passed'
            break
        if st.phase == 'CARRY' and q[0] >= SCN['x_end']:
            result = 'done'
            break
        # 外層進度協調（10 Hz）
        if k % 2 == 0:
            if coord and BR.PHASES.index(st.phase) <= BR.PHASES.index('CLOSE'):
                if st.phase == 'CRUISE':
                    t_rem = max(st.t_pre - t, 0.0) + 0.3 + task.t_insert
                elif st.phase == 'PREGRASP':
                    ep, _ = BR.pose_err(Tm, task.T_pre)
                    t_rem = ep / 0.05 + 0.3 + task.t_insert
                elif st.phase == 'INSERT':
                    t_rem = max(task.t_insert - (t - st.t_phase), 0.0) + 0.2
                else:
                    t_rem = 0.0
                need = T_CLOSE + T_EV + T_LIFT if st.phase != 'CLOSE' else max(T_CLOSE + T_EV - (t - st.t_phase), 0.0) + T_LIFT
                v_need = (x_we - 0.02 - q[0]) / max(t_rem + need, 1e-3)
                v_b = float(np.clip(min(v_c, v_need), 0.0, v_c))
                if v_b < V_MIN:
                    degraded = True
            else:
                v_b = v_c
        # 參考（對應預測狀態時刻 t＋D）
        t_eff = t + D_STEPS * PHYS
        ts = np.array([t_eff - 1.0, t_eff + 30.0])
        x_eff = q[0] + v_b * D_STEPS * PHYS
        plan = (ts, np.array([x_eff - v_b * 1.0, x_eff + v_b * 30.0]), 0.0)
        T_seq, vref = BR.horizon(task, st, plan, t_eff, cfg_c.N, DT)
        th = q[2]
        vb_body = np.stack([[math.cos(th) * v[0] + math.sin(th) * v[1], -math.sin(th) * v[0] + math.cos(th) * v[1], 0.0] for v in vref])
        cfg_c.base_vref = vb_body
        if grasp_w is not None:
            gw = st.phase in ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT')
            cfg_c.w_p, cfg_c.w_vref = (grasp_w if gw else (50.0, 1.0))
        # 延遲補償：以控制器模型把 (q,s) 推過在途命令
        qp, sp = q.copy(), s.copy()
        for u_in in pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg_c, 1)
        t0 = time.perf_counter()
        r = FR.solve_sp_fr(K, S.make_z(qp, sp), u_prev, T_seq, cfg_c, U_warm=U_warm)
        rec['solve_ms'].append((time.perf_counter() - t0) * 1e3)
        if r.ok:
            u, U_warm = r.u0, r.U
        else:
            rec['fail'] += 1
            u, U_warm = np.zeros(9), None
        u_prev = u.copy()
        for _ in range(NPH):
            pipe.append(u.copy())
            ua = pipe.pop(0)
            q, s = S.plant_phys_step(q, s, ua, cfg_p, 1)
        ep, er = BR.pose_err(K.fk(q, 'link_tcp'), T_seq[0])
        rec['t'].append(t); rec['x'].append(float(q[0])); rec['phase'].append(st.phase)
        rec['ep'].append(ep); rec['er'].append(er); rec['v_b'].append(v_b)
        t += DT
    if result is None:
        result, why = 'fail', 'timeout'
    x = np.asarray(rec['x']); tt = np.asarray(rec['t'])
    vx = np.gradient(x, tt) if len(x) > 2 else np.zeros_like(x)
    ph = np.asarray(rec['phase'])
    m_g = np.isin(ph, ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT'))
    ev = {e['to']: round(e['t'], 2) for e in st.events}
    ep_close = None
    if 'CLOSE' in ev:
        i = int(np.searchsorted(tt, ev['CLOSE']))
        ep_close = round(rec['ep'][min(i, len(rec['ep']) - 1)] * 1e3, 2)
    out = dict(label=label, grasp_w=grasp_w, v_c=v_c, coord=coord, plant_alpha=plant_alpha, ctrl_alpha=ctrl_alpha, result=result, why=why, degraded=degraded,
               t_end=round(t, 2), events=ev, t_pre=round(t_pre, 2), n_fail=rec['fail'],
               v_min_grasp=round(float(vx[m_g].min()), 4) if m_g.any() else None,
               t_slow_grasp=round(float(np.sum(m_g & (vx < 0.01)) * DT), 2),
               decel_peak=round(float(-np.min(np.gradient(vx, tt))), 3) if len(vx) > 2 else None,
               ep_close_start_mm=ep_close, ep_p95_mm=round(float(np.percentile(rec['ep'], 95)) * 1e3, 2),
               x_at_close=None if 'CLOSE' not in ev else round(float(np.interp(ev['CLOSE'], tt, x)), 3),
               solve_ms_p95=round(float(np.percentile(rec['solve_ms'], 95)), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='quick', choices=['quick', 'all'])
    a = ap.parse_args()
    cases = [(0.05, True, 1.0, 1.0, 'C3 正常')]
    if a.cases == 'all':
        cases = []
        for v in (0.03, 0.05, 0.07):
            cases += [(v, True, 1.0, 1.0, 'C3 正常'), (v, False, 1.0, 1.0, 'C3−prog 正常'),
                      (v, True, 0.6, 1.0, 'C3 伺服慢（未知失配）'), (v, True, 0.6, 0.6, 'C3 伺服慢（已知）'),
                      (v, False, 0.6, 1.0, 'C3−prog 伺服慢（未知失配）')]
    res = []
    for v, co, pa, ca, lab in cases:
        r = run(v, co, pa, ca, label=lab, grasp_w=GRASP_W)
        res.append(r)
        print(f"{lab:<20} v_c {v:.2f}：{r['result']}/{r['why']} 降級 {r['degraded']} 結束 {r['t_end']} s；事件 {r['events']}；閉爪時 x {r['x_at_close']}；"
              f"抓取期 v_min {r['v_min_grasp']}、<0.01 m/s {r['t_slow_grasp']} s、減速峰 {r['decel_peak']}；閉爪起誤差 {r['ep_close_start_mm']} mm；失敗 {r['n_fail']}；求解 p95 {r['solve_ms_p95']} ms", flush=True)
    out = os.path.join(HERE, 'results', 'bottle', f'offline_c3_{a.cases}.json')
    json.dump({'scenario': SCN, 'model': '設定點模型＋底盤運動學；延遲 8 物理步；無接觸（抓持為代理）', 'results': res}, open(out, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(out, WS))


if __name__ == '__main__':
    sys.exit(main())
