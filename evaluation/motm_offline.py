#!/usr/bin/env python3
"""MotM 協同權重的離線篩選（無 Isaac）。

受控體與延遲：沿用 wgmpc_closed_loop_sim.run（已對照 Isaac 的翻號率／振盪），
設定與 run_nav_handover.sh 的求解節點相同；底盤命令增益 0.9／0.9／0.82（rec8）。

兩個情境：
  approach  底盤在停位前 d0 處，照 √ 減速剖面朝停位走；TCP 目標固定在把手。
            手臂名目姿態 = 停位時的抓取姿態。
  pull      底盤在停位、TCP 在把手；目標以 v_pull 沿世界 −y 走 200 mm。
            底盤參考速度 = 目標速度（本體座標），手臂名目姿態 = 收回 X mm 的姿態。

**界線**：沒有接觸、沒有抽屜動力學（拉開時目標是預先給定的軌跡，不是閉在
實測開度上），沒有 Isaac 的積分器細節。只用來篩配置。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
PARK = np.array([-0.136412, 0.560, 1.297349])
QG = np.array([-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813])
BASE_GAIN = 0.9
DELAY = 1.4
COMP = 1.6


def body_vel(world_xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * world_xy[0] + s * world_xy[1],
                     -s * world_xy[0] + c * world_xy[1]])


def ik_arm(K, base, q_arm0, T_goal, tcp='link_tcp', iters=200):
    """底盤固定，只解手臂（阻尼最小平方）。回傳手臂六軸。"""
    q = np.r_[base, q_arm0].astype(float)
    lim = np.array(K.joint_limits())
    for _ in range(iters):
        T = K.fk(q, tcp)
        e = np.r_[T_goal[:3, 3] - T[:3, 3],
                  0.5 * (np.cross(T[:3, 0], T_goal[:3, 0])
                         + np.cross(T[:3, 1], T_goal[:3, 1])
                         + np.cross(T[:3, 2], T_goal[:3, 2]))]
        if np.linalg.norm(e[:3]) < 1e-7 and np.linalg.norm(e[3:]) < 1e-7:
            break
        J = K.jacobian(q, tcp)[:, 3:]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        q[3:] = np.clip(q[3:] + dq, lim[0, 3:], lim[1, 3:])
    return q[3:]


def make_cfg(**kw):
    i = json.load(open(IDENT))
    am = S.ArmSetpointModel(alpha=i['alpha'], bias=i['bias_rad'],
                            phys_dt=i['phys_dt_measured_s'])
    base = dict(N=5, dt=0.05, arm_model=am, w_s=1e-3, w_s_arm=0.05)
    base.update(kw)
    return S.WGMPCConfigSP(**base)


def simulate(K, cfg, q0, T_of_t, coord_of_t, t_end):
    """用已對照過 Isaac 的閉環模擬器（wgmpc_closed_loop_sim.run）：
    延遲 1.0 週期＋發布 0.6、發布歷史查表補償 1.6／1.0、近目標整形 γ 1.0、
    底盤命令增益 0.9 —— 與 run_nav_handover.sh 的求解節點設定相同。"""
    import wgmpc_closed_loop_sim as CL
    CL.BASE_GAIN = np.array([BASE_GAIN, BASE_GAIN, 0.82])

    def hook(t, q, c):
        for k, v in coord_of_t(t, q).items():
            setattr(c, k, v)
    r = CL.run(cfg, K, q0, q0[3:].copy(), T_of_t, t_end=t_end,
               reach_p=-1.0, delay_cycles=1.0, d_pub_cycles=0.6,
               comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=1.0,
               pre_solve=hook)['rec']
    return {'t': r['t'], 'ep': r['ep'], 'u': r['u_req'], 'q': r['q'],
            'fail': ~r['ok'].astype(bool)}


def flip_rate(U):
    """手臂各軸命令翻號率（|u| > 0.01 rad/s 才算），取最大軸。"""
    out = 0.0
    for j in range(3, 9):
        x = U[:, j]
        m = np.abs(x) > 0.01
        sg = np.sign(x[m])
        if len(sg) > 2:
            out = max(out, float(np.mean(sg[1:] != sg[:-1])))
    return out


def scen_approach(K, w, d0=0.10, a_ref=0.002, v_cap=0.030, t_end=10.0):
    yaw = PARK[2]
    hd = np.array([math.cos(yaw), math.sin(yaw)])
    q0 = np.r_[PARK[:2] - d0 * hd, yaw, QG]
    T = K.fk(np.r_[PARK, QG], 'link_tcp')
    cfg = make_cfg(w_a=w['w_a'], w_p=w['w_p'])

    def coord(t, q):
        d = float(hd @ (PARK[:2] - q[:2]))
        v = min(v_cap, math.sqrt(2 * a_ref * max(d, 0.0)))
        return dict(w_vref=w['w_vref'], base_vref=(v, 0.0, 0.0),
                    w_qn=w['w_qn'], arm_q_nom=tuple(QG))
    r = simulate(K, cfg, q0, lambda t: T, coord, t_end)
    vb = r['u'][:, 0]
    d_end = float(hd @ (PARK[:2] - r['q'][-1, :2]))
    i_ok = np.where(r['ep'] < 0.002)[0]
    return dict(ep_max_after_2s=float(r['ep'][r['t'] >= 2.0].max()),
                t_first_2mm=None if len(i_ok) == 0 else float(r['t'][i_ok[0]]),
                base_d_end_mm=d_end * 1e3,
                base_v_min_first6s=float(vb[r['t'] < 6.0].min()),
                flips=flip_rate(r['u']), fails=int(r['fail'].sum()))


def scen_pull(K, w, X=0.06, v_pull=0.018, acc=0.05, travel=0.200):
    T0 = K.fk(np.r_[PARK, QG], 'link_tcp')
    ax = np.array([0.0, -1.0, 0.0])
    Tr = T0.copy()
    Tr[:3, 3] = T0[:3, 3] + X * ax
    q_ret = ik_arm(K, PARK, QG, Tr)
    t_ramp = v_pull / acc
    t_end = travel / v_pull + t_ramp + 2.0

    def s_of_t(t):
        if t < t_ramp:
            return 0.5 * acc * t * t, acc * t
        s = 0.5 * acc * t_ramp ** 2 + v_pull * (t - t_ramp)
        return (min(s, travel), v_pull if s < travel else 0.0)

    def T_of_t(t):
        T = T0.copy()
        T[:3, 3] = T0[:3, 3] + s_of_t(t)[0] * ax
        return T
    cfg = make_cfg(w_a=w['w_a'], w_p=w['w_p'])

    def coord(t, q):
        v = s_of_t(t)[1]
        vb = body_vel(v * ax[:2], q[2])
        return dict(w_vref=w['w_vref'], base_vref=(vb[0], vb[1], 0.0),
                    w_qn=w['w_qn'], arm_q_nom=tuple(q_ret))
    q0 = np.r_[PARK, QG]
    r = simulate(K, cfg, q0, T_of_t, coord, t_end)
    qf = r['q'][-1]
    base_dy = float(qf[1] - PARK[1])
    tcp_dy = float(K.fk(qf, 'link_tcp')[1, 3] - T0[1, 3])
    return dict(ep_max=float(r['ep'].max()), ep_p95=float(np.percentile(r['ep'], 95)),
                arm_share_mm=(-(tcp_dy - base_dy)) * 1e3,
                base_dy_mm=-base_dy * 1e3, flips=flip_rate(r['u']),
                fails=int(r['fail'].sum()), X_mm=X * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    K = WholeBodyKinematics.from_urdf_file(URDF)
    grid = dict(w_a=[0.05, 0.01, 0.003], w_p=[50.0, 200.0],
                w_vref=[0.01, 0.03], w_qn=[0.0, 0.3, 1.0])
    rows = []
    for vals in itertools.product(*grid.values()):
        w = dict(zip(grid.keys(), vals))
        ra = scen_approach(K, w)
        rp = scen_pull(K, w)
        rows.append(dict(w=w, approach=ra, pull=rp))
        print(f"{w}  approach ep>2s {ra['ep_max_after_2s']*1e3:5.2f}mm "
              f"d_end {ra['base_d_end_mm']:5.1f}mm flips {ra['flips']:.2f} | "
              f"pull ep_max {rp['ep_max']*1e3:5.2f} p95 {rp['ep_p95']*1e3:5.2f}mm "
              f"arm {rp['arm_share_mm']:5.1f}/{rp['X_mm']:.0f}mm "
              f"flips {rp['flips']:.2f} fails {ra['fails']+rp['fails']}",
              flush=True)
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
