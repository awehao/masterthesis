#!/usr/bin/env python3
"""W-GMPC 的**離線閉迴路模擬**（無 Isaac、無 GPU）。

受控對象用已辨識並驗證過的模型：
  手臂：每物理步 x⁺ = x + α⊙(s − x) + b，設定點 s⁺ = s + u_a·dt_p
        （free4 辨識，50 ms 預測最差關節 RMSE 0.44 mrad）
  底盤：q⁺ = q + dt·B(θ)u（free4 實測追隨比 0.96–1.06）

控制器就是 `wgmpc_core_sp.solve_sp` 本體，20 Hz（模擬時間）。

**界線**：此處沒有接觸、沒有 Isaac 的積分器細節，也沒有安全層與 E2。
離線收斂**不等於** Isaac 會收斂 —— 只用來快速篩配置，最後仍須實跑確認。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import task_error             # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS_DT = 0.01


def make_cfg(ident, **kw):
    am = S.ArmSetpointModel(alpha=ident['alpha'], bias=ident['bias_rad'],
                            phys_dt=ident['phys_dt_measured_s'])
    base = dict(N=5, dt=0.05, arm_model=am, row_scaling=True)
    base.update(kw)
    return S.WGMPCConfigSP(**base)


BASE_GAIN = np.array([1.0, 1.0, 1.0])     # 由 --base-gain 覆寫


def plant_step(q, s, u, cfg, n_phys):
    """把命令 u 保持 n_phys 個物理步。**手臂部分與節點共用核心的同一段碼。**

    `BASE_GAIN`：底盤的命令→實測增益。rec8 實測迴歸斜率
    vx 0.894、vy 0.920、**wz 0.819** —— 核心把底盤當成完美追隨（1.0）。
    """
    uu = np.asarray(u, float).copy()
    uu[:3] = uu[:3] * BASE_GAIN
    return S.plant_phys_step(q, s, uu, cfg, n_phys)


def run(cfg, K, q0, s0, T_des, t_end=60.0, reach_p=0.005, reach_r=0.02,
        hold_s=2.0, tcp='link_tcp', delay_cycles=0.0,
        compensate=0.0, comp_mode='pipe', jitter_cycles=0.0,
        seed=0, measure_comp=False, u_prev_mode='applied',
        d_pub_cycles=0.0, comp_state=None, comp_cmd=None):
    """`delay_cycles`：命令從算出到真正套用的延遲（以控制週期計）。

    實跑量到的端到端延遲 ≈ 1.40 個週期（rec7：發布延遲 0.60 ＋ cmd_age 0.60
    ＋ 一個物理步 0.01 s）。核心的預測模型**沒有這個延遲** ——
    它假設 u0 立刻作用。

    **兩種時間必須分開**（先前用同一個 D 做兩件事是錯的）：

        量測 ──D_pub──> 發布 ──D_cmd──> 生效
        └──────── D_state = D_pub + D_cmd ────────┘

    * `d_pub_cycles`：量測 → 發布（求解與傳遞到發布）。
      離線原本求解是**瞬時**的 ⇒ D_pub = 0，於是 D_state ≡ D_cmd，
      這個區別被掩蓋。加入 d_pub 才驗得出來。
    * `comp_state`：狀態要往前推多久（應等於 D_state）。
    * `comp_cmd`：由發布歷史倒查命令時用的偏移（應等於 D_cmd）。
    `compensate` 保留為兩者同值的簡寫。
    """
    n_phys = int(round(cfg.dt / cfg.arm_model.phys_dt))
    n_del = int(round(delay_cycles * n_phys))     # 以物理步計的延遲
    q, s = np.asarray(q0, float).copy(), np.asarray(s0, float).copy()
    pipe = [np.zeros(S.NU)] * max(n_del, 0)       # 命令管線（延遲）
    u_prev = np.zeros(S.NU)
    U_warm = None
    t = 0.0
    hold_t0 = None
    rec = {'t': [], 'ep': [], 'er': [], 'sat': [], 'ok': [],
           'u_req': [], 'u_app': []}
    reached_held = False
    n_fail = 0
    cs = compensate if comp_state is None else comp_state
    cc = compensate if comp_cmd is None else comp_cmd
    n_comp = int(round(cs * n_phys))          # 狀態前推的物理步數
    n_pub = int(round(d_pub_cycles * n_phys))  # 量測 → 發布（物理步）
    # **命令三態分離**（與線上語意對齊）：
    #   requested  求解端算出並發布的命令（= 節點的 request／發布歷史）
    #   in_flight  已發布、尚未生效的命令（延遲管線內）
    #   applied    執行端 E2 之後**真正套用**的命令（線上 u_prev 的唯一權威來源）
    # 先前這裡用 `u_prev = u`（剛求解的 requested），**與線上不同** ——
    # 有延遲時兩者本來就不同，會直接改變下一輪的加速度約束與平滑成本。
    hist = []            # (發布的模擬時間, u_requested)
    applied_last = np.zeros(S.NU)     # 最近一筆**實際套用**的命令
    pubq = [np.zeros(S.NU)] * max(n_pub, 0)   # 量測→發布的佇列
    rng = np.random.default_rng(seed)
    d_true = []          # 每輪真正的延遲（物理步），供 measure_comp 使用
    while t < t_end:
        # **延遲抖動**：rec8 實測端到端在 0.80–1.60 個週期之間擺動
        if jitter_cycles > 0.0:
            n_del_now = max(0, int(round(
                (delay_cycles + rng.uniform(-jitter_cycles, jitter_cycles))
                * n_phys)))
        else:
            n_del_now = n_del
        d_true.append(n_del_now)
        # measure_comp：用**上一輪實際延遲**當本輪補償（節點可由 cmd_age 量到）
        if measure_comp and len(d_true) > 1:
            n_comp = d_true[-2]
        # ---- 延遲補償：從**預測狀態**求解 ----
        if n_comp > 0:
            q_s, s_s = q.copy(), s.copy()
            # 依序套用尚未作用的在途命令；不足的部分以最後一筆保持
            if comp_mode == 'hist':
                # **發布歷史查表**（節點可直接照搬）：
                # 時刻 τ 作用的命令是 τ − D 時「已發布的最新一筆」。
                # 要把狀態由 t 推到 t + D，就用 [t − D, t) 這段的已發布命令。
                Dt = cc * cfg.dt
                inflight = []
                for j in range(n_comp):
                    tau = t + j * cfg.arm_model.phys_dt - Dt
                    uu = np.zeros(S.NU)
                    for (tp, up) in hist:
                        if tp <= tau + 1e-12:
                            uu = up
                        else:
                            break
                    inflight.append(uu)
            elif comp_mode == 'hold':
                # **簡化版**：以最後一筆已送出的命令保持往前推。
                # 節點端只需記住上一筆命令，不必維護逐物理步的管線。
                inflight = [u_prev.copy()] * n_comp
            else:
                inflight = list(pipe[:n_comp])
                while len(inflight) < n_comp:
                    inflight.append(u_prev.copy())
            for u_in in inflight:
                q_s, s_s = plant_step(q_s, s_s, u_in, cfg, 1)
        else:
            q_s, s_s = q, s
        r = S.solve_sp(K, S.make_z(q_s, s_s), u_prev, T_des, cfg,
                       U_warm=U_warm)
        if not r.ok:
            n_fail += 1
            u = np.zeros(S.NU)
            U_warm = None
        else:
            u = r.u0
            U_warm = r.U
        e = task_error(K, q, T_des, tcp)      # 判準用**實際**狀態，不用預測
        ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        rec['t'].append(t); rec['ep'].append(ep); rec['er'].append(er)
        rec['sat'].append(float(np.abs(u[3:]).max() / cfg.v_arm))
        rec['ok'].append(bool(r.ok))
        rec['u_req'].append(u.copy())
        rec['u_app'].append(applied_last.copy())
        if ep <= reach_p and er <= reach_r:
            if hold_t0 is None:
                hold_t0 = t
            elif t - hold_t0 >= hold_s:
                reached_held = True
                break
        else:
            hold_t0 = None
        # 命令先進管線，實際作用的是 n_del 個物理步之前那一筆
        for _ in range(n_phys):
            # **發布延遲**：本輪算出的命令要 n_pub 個物理步後才發布，
            # 在那之前管線裡進的是上一筆已發布的命令。
            pubq.append(u.copy())
            u_pub = pubq.pop(0) if n_pub > 0 else u
            pipe.append(u_pub.copy())
            # 管線長度 = 當前延遲；多出來的最舊者就是本步實際套用的命令
            u_app = u.copy()
            while len(pipe) > max(n_del_now, 0):
                u_app = pipe.pop(0)
            q, s = plant_step(q, s, u_app, cfg, 1)
            applied_last = u_app          # **實際套用**，供下一輪當 u_prev
        # 發布歷史記的是**發布時刻**（= 求解時刻 + D_pub），不是求解時刻
        hist.append((t + d_pub_cycles * cfg.dt, u.copy()))
        if len(hist) > 64:
            hist.pop(0)
        # **與線上一致**：u_prev 取「E2 之後實際套用」的那一筆，
        # 不是剛求解的 requested。
        u_prev = (applied_last.copy() if u_prev_mode == 'applied'
                  else u.copy())
        t += cfg.dt
    for k in rec:
        rec[k] = np.array(rec[k])
    return {'reached_held': reached_held, 'n_fail': n_fail,
            't_reach': (float(hold_t0) if reached_held else None),
            'rec': rec}


def summarise(res, tag=''):
    r = res['rec']
    ep, er, sat = r['ep'], r['er'], r['sat']
    return {
        'tag': tag, 'reached_held': res['reached_held'],
        't_reach_s': res['t_reach'], 'n_cycles': int(len(ep)),
        'err_p_min': float(ep.min()), 'err_p_p50': float(np.percentile(ep, 50)),
        'err_p_final': float(ep[-1]),
        'err_r_min': float(er.min()), 'err_r_p50': float(np.percentile(er, 50)),
        'err_r_final': float(er[-1]),
        'sat_frac': float(np.mean(sat > 0.95)),
        'n_in_tol': int(np.sum((ep <= 0.005) & (er <= 0.02))),
        'n_fail': res['n_fail'],
        # requested 與 applied 的差距：若為 0 表示延遲沒被模出來
        'req_vs_app_p50': (float(np.percentile(np.abs(
            r['u_req'] - r['u_app']).max(axis=1), 50))
            if len(r.get('u_req', [])) else float('nan')),
    }


def load_scene(run_dir, ident_path):
    ident = json.load(open(ident_path))
    K = WholeBodyKinematics.from_urdf_file(
        os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    w = json.load(open(os.path.join(run_dir, 'wg2_out.json')))
    q0 = np.array(w['start_q'], float)
    T0 = K.fk(q0, w['args']['tcp'])
    T_des = np.eye(4)
    T_des[:3, 3] = np.array(w['target_tcp'], float)
    T_des[:3, :3] = T0[:3, :3]
    return ident, K, q0, T_des, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run',
                    default=os.path.join(HERE, 'runs',
                                         'wgmpc_wg2_rec7_011217'))
    ap.add_argument('--ident',
                    default=os.path.join(HERE, 'results',
                                         'wgmpc_arm_sp_ident_free4.json'))
    ap.add_argument('--t-end', type=float, default=60.0)
    ap.add_argument('--delay-cycles', type=float, default=0.0)
    ap.add_argument('--sweep-delay', action='store_true')
    ap.add_argument('--sweep-comp', action='store_true')
    ap.add_argument('--sweep-jitter', action='store_true')
    ap.add_argument('--sweep-base-gain', action='store_true')
    ap.add_argument('--sweep-split', action='store_true')
    ap.add_argument('--sweep-measured', action='store_true')
    ap.add_argument('--compensate', type=float, default=0.0)
    a = ap.parse_args()
    ident, K, q0, T_des, w = load_scene(a.run, a.ident)
    cfg = make_cfg(ident)
    # 初始設定點 = 實測關節角（執行端 setpoint_init 的定義）
    s0 = q0[3:].copy()
    if a.sweep_delay:
        print('=== 延遲掃描（現行配置，找能重現 rec7 的延遲）===')
        print(f"  {'延遲(週期)':>10} {'到達保持':>8} {'err_p p50':>10}"
              f" {'err_r p50':>10} {'飽和比例':>8} {'in_tol':>7}")
        out = []
        for dc in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            r = run(cfg, K, q0, s0, T_des, t_end=a.t_end, delay_cycles=dc)
            m = summarise(r, f'delay{dc}')
            out.append(m)
            print(f"  {dc:10.1f} {str(m['reached_held']):>8}"
                  f" {m['err_p_p50']:10.4f} {m['err_r_p50']:10.4f}"
                  f" {m['sat_frac']:8.3f} {m['n_in_tol']:7d}")
        print()
        pub = [x for x in w['log'] if x.get('published')]
        ep = np.array([x['err_p'] for x in pub])
        er = np.array([x['err_r'] for x in pub])
        U = np.array([x['stages']['request']['body'] for x in pub], float)
        print(f"  rec7 實跑   {'False':>8} {np.percentile(ep,50):10.4f}"
              f" {np.percentile(er,50):10.4f}"
              f" {np.mean(np.abs(U[:,3:]).max(axis=1)/0.9992>0.95):8.3f}"
              f" {int(((ep<=0.005)&(er<=0.02)).sum()):7d}")
        return 0
    if a.sweep_measured:
        print('=== 用 rec9 **逐筆量到**的值：D_pub 0.60、D_cmd 1.00 '
              '⇒ D_state 1.60 ===')
        print('  （先前節點用單一 D = 1.4 同時當 state 與 cmd）')
        print(f"  {'補償設定':>30} {'到達保持':>8} {'到達s':>7}"
              f" {'err_p 末':>10} {'err_r 末':>10} {'飽和':>7} {'同時達標':>8}")
        for cs_, cc_, lab in (
                (0.0, 0.0, '不補償'),
                (1.4, 1.4, '混用單一 D=1.4（rec8/rec9 節點）'),
                (1.6, 1.6, '混用單一 D=1.6'),
                (1.0, 1.0, '混用單一 D=1.0'),
                (1.6, 1.0, '**分開 state 1.6 / cmd 1.0（量到值）**')):
            r = run(cfg, K, q0, s0, T_des, t_end=a.t_end,
                    delay_cycles=1.0, d_pub_cycles=0.6,
                    comp_state=cs_, comp_cmd=cc_, comp_mode='hist')
            m = summarise(r, '')
            print(f"  {lab:>30} {str(m['reached_held']):>8}"
                  f" {str(round(m['t_reach_s'],2) if m['t_reach_s'] else '-'):>7}"
                  f" {m['err_p_final']:10.5f} {m['err_r_final']:10.5f}"
                  f" {m['sat_frac']:7.3f} {m['n_in_tol']:8d}")
        return 0
    if a.sweep_split:
        print('=== 兩種時間分開 vs 混用（D_pub 0.6、D_cmd 0.6 ⇒ D_state 1.2）===')
        print('  rec8 實測：發布延遲 p50 0.020 s（0.40 週期）、'
              'cmd_age p50 0.020 s（0.40 週期）')
        print(f"  {'補償設定':>28} {'到達保持':>8} {'err_p p50':>10}"
              f" {'err_r p50':>10} {'飽和':>7} {'同時達標':>8}")
        for cs_, cc_, lab in (
                (0.0, 0.0, '不補償'),
                (1.2, 1.2, '混用單一 D=1.2（現行節點）'),
                (0.6, 0.6, '混用單一 D=0.6'),
                (1.2, 0.6, '**分開 state 1.2 / cmd 0.6**')):
            r = run(cfg, K, q0, s0, T_des, t_end=a.t_end,
                    delay_cycles=0.6, d_pub_cycles=0.6,
                    comp_state=cs_, comp_cmd=cc_, comp_mode='hist')
            m = summarise(r, '')
            print(f"  {lab:>28} {str(m['reached_held']):>8}"
                  f" {m['err_p_p50']:10.5f} {m['err_r_p50']:10.5f}"
                  f" {m['sat_frac']:7.3f} {m['n_in_tol']:8d}")
        return 0
    if a.sweep_base_gain:
        global BASE_GAIN
        print('=== 底盤追隨增益的影響（延遲 1.4、補償 1.4、無抖動）===')
        print('  rec8 實測：vx 0.894、vy 0.920、wz 0.819')
        print(f"  {'底盤增益':>18} {'到達保持':>8} {'err_p p50':>10}"
              f" {'err_r p50':>10} {'飽和':>7} {'同時達標':>8}")
        for g, lab in (((1.0, 1.0, 1.0), '完美 1/1/1'),
                       ((0.894, 0.920, 1.0), '僅線性 .89/.92/1'),
                       ((1.0, 1.0, 0.819), '僅角速 1/1/.82'),
                       ((0.894, 0.920, 0.819), 'rec8 實測全部')):
            BASE_GAIN = np.array(g, float)
            r = run(cfg, K, q0, s0, T_des, t_end=a.t_end,
                    delay_cycles=1.4, compensate=1.4, comp_mode='hist')
            m = summarise(r, '')
            print(f"  {lab:>18} {str(m['reached_held']):>8}"
                  f" {m['err_p_p50']:10.5f} {m['err_r_p50']:10.5f}"
                  f" {m['sat_frac']:7.3f} {m['n_in_tol']:8d}")
        BASE_GAIN = np.array([1.0, 1.0, 1.0])
        return 0
    if a.sweep_jitter:
        print('=== 延遲抖動下：固定補償 vs 逐輪實測補償 ===')
        print('  （延遲中心 1.4 週期；rec8 實測端到端 0.80–1.60 週期）')
        print(f"  {'抖動±':>7} {'補償方式':>10} {'到達保持':>8} {'err_p p50':>10}"
              f" {'err_r p50':>10} {'飽和':>7} {'同時達標':>8}")
        for jit in (0.0, 0.2, 0.4):
            for meas, lab in ((False, '固定1.4'), (True, '逐輪實測')):
                r = run(cfg, K, q0, s0, T_des, t_end=a.t_end,
                        delay_cycles=1.4, compensate=1.4, comp_mode='hist',
                        jitter_cycles=jit, measure_comp=meas, seed=7)
                m = summarise(r, '')
                print(f"  {jit:7.1f} {lab:>10} {str(m['reached_held']):>8}"
                      f" {m['err_p_p50']:10.5f} {m['err_r_p50']:10.5f}"
                      f" {m['sat_frac']:7.3f} {m['n_in_tol']:8d}")
        return 0
    if a.sweep_comp:
        print('=== 延遲補償掃描（固定實測延遲 1.4 週期）===')
        print(f"  {'補償(週期)':>10} {'到達保持':>8} {'到達時間s':>10}"
              f" {'err_p 末':>10} {'err_r 末':>10} {'飽和':>7} {'無解':>5}")
        for mode in ('pipe', 'hist', 'hold'):
            print(f'  -- 補償方式「{mode}」--')
            for cp in (0.0, 1.0, 1.4, 1.5, 2.0):
                r = run(cfg, K, q0, s0, T_des, t_end=a.t_end,
                        delay_cycles=1.4, compensate=cp, comp_mode=mode)
                m = summarise(r, f'{mode}{cp}')
                print(f"  {cp:10.1f} {str(m['reached_held']):>8}"
                      f" {str(round(m['t_reach_s'],2) if m['t_reach_s'] else '-'):>10}"
                      f" {m['err_p_final']:10.5f} {m['err_r_final']:10.5f}"
                      f" {m['sat_frac']:7.3f} {m['n_fail']:5d}")
        print()
        print('  **補償不改權重、視界或任何限制** —— 只是從預測狀態求解。')
        return 0
    res = run(cfg, K, q0, s0, T_des, t_end=a.t_end,
              delay_cycles=a.delay_cycles, compensate=a.compensate)
    sm = summarise(res, 'current')
    print('=== 現行配置的離線閉迴路 ===')
    for k, v in sm.items():
        print(f'  {k:14s} {v}')
    print()
    print('=== 對照 rec7 實跑 ===')
    pub = [x for x in w['log'] if x.get('published')]
    ep = np.array([x['err_p'] for x in pub]); er = np.array([x['err_r'] for x in pub])
    U = np.array([x['stages']['request']['body'] for x in pub], float)
    print(f'  err_p p50 {np.percentile(ep,50):.4f}（離線 {sm["err_p_p50"]:.4f}）')
    print(f'  err_r p50 {np.percentile(er,50):.4f}（離線 {sm["err_r_p50"]:.4f}）')
    print(f'  飽和比例 {np.mean(np.abs(U[:,3:]).max(axis=1)/0.9992>0.95):.3f}'
          f'（離線 {sm["sat_frac"]:.3f}）')
    print(f'  in_tol {int(((ep<=0.005)&(er<=0.02)).sum())}'
          f'（離線 {sm["n_in_tol"]}）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
