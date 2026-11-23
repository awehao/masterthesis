#!/usr/bin/env python3
"""K3-M 受控暫停用的離線閉迴路（依 Codex 20261023_225057 第 2 節 1–4）。wgmpc_closed_loop_sim.run 不改。

與 CL.run（comp_mode='hist'、gamma=0）相同的受控體、求解器、延遲與補償，差別只在命令路徑：
  1. **預測與實際分開**：延遲補償的模型外推用 CL.plant_step 推進複本，不碰權威 applied 狀態；只有實際受控體推進更新
     applied、物理步序號與套用回報。
  2. **序號隨管線傳遞**：每筆發布訊息 (seq, t_pub, u) 進入發布延遲（n_pub 物理步）與命令延遲（n_del 物理步）管線；
     執行端每個實際物理步回報 (phys_step, t, u_applied[3+6], src_seq)，src_seq 取自管線，不以數值反查。
  3. **停止發布**：publish=False 的週期不入列（管線該位置為空）；執行端保留最後收到的訊息，
     其年齡（t − t_pub）> max_cmd_age 時走 wb_wheel_limit.limit9_timeout（底盤經輪級減速、手臂立即 0；src_seq＝None），
     不注入人工零值。
  4. 發布時刻記入發布歷史（t + d_pub·dt），供下一輪 hist 補償；未發布的週期不記。
等價性：每輪都發布且不改寫時，軌跡與 CL.run 逐元素相同（含 pre 逐輪改目標的情形；test_door_k3m_pause_loop.py 核對）。
呼叫順序同 CL.run：T_of(t) 在 pre 之前取值（pre 改的目標下一輪才生效）。

hooks：
  pre(t, q, cfg)                         每輪求解前（改目標、cfg.base_hold）
  decide(t, r, applied) -> (publish, u)   求解後；publish=False ⇒ 不入列；u=None ⇒ 用求解結果（失敗時 0，與 CL 同）
  on_applied(phys_step, t, u, src_seq)    每個**實際**物理步
  skip_solve(t) -> bool                   True ⇒ 本輪不求解（停止發布後）；decide 收到 r=None
"""
from __future__ import annotations

import numpy as np

import door_k3m_offline_replay as RP
import wb_wheel_limit as WL

CL = RP.CL
S = CL.S


def run_loop(cfg, K, q0, s0, T_of, t_end, delay_cycles=1.4, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0,
             pre=None, decide=None, on_applied=None, skip_solve=None, max_cmd_age=0.2, wcfg=None, tcp='link_tcp'):
    wcfg = wcfg or WL.WheelLimitConfig()
    pdt = cfg.arm_model.phys_dt
    n_phys = int(round(cfg.dt / pdt))
    n_del = int(round(delay_cycles * n_phys))
    n_comp = int(round(comp_state * n_phys))
    n_pub = int(round(d_pub_cycles * n_phys))
    q, s = np.asarray(q0, float).copy(), np.asarray(s0, float).copy()
    pubq = [None] * max(n_pub, 0)                     # 元素：(seq, t_pub, u) 或 None（該步沒有新訊息）
    pipe = [(0, 0.0, np.zeros(S.NU))] * max(n_del, 0)  # 起始＝序號 0 的零命令（與 CL 的零管線相同）
    held = (0, 0.0, np.zeros(S.NU))                  # 執行端最後收到的訊息
    applied = np.zeros(S.NU)
    u_prev = np.zeros(S.NU)
    U_warm = None
    hist = []
    seq = 0
    j = 0                                            # 實際物理步序號
    t = 0.0
    rec = {'t': [], 'q': [], 's': [], 'ok': [], 'u_req': [], 'u_app': [], 'published': [], 'seq': [], 'phys': []}
    Dt = comp_cmd * cfg.dt
    while t < t_end:
        T_cur = T_of(t)                              # 與 CL.run 同序：先取目標、再呼叫 pre
        if pre is not None:
            pre(t, q, cfg)
        solve = not (skip_solve is not None and skip_solve(t))
        r = None
        if solve:
            q_s, s_s = q.copy(), s.copy()
            for jj in range(n_comp):                 # 模型外推（複本；不碰 applied／phys）
                tau = t + jj * pdt - Dt
                uu = np.zeros(S.NU)
                for (tp, up) in hist:
                    if tp <= tau + 1e-12:
                        uu = up
                    else:
                        break
                q_s, s_s = CL.plant_step(q_s, s_s, uu, cfg, 1)
            r = S.solve_sp(K, S.make_z(q_s, s_s), u_prev, T_cur, cfg, U_warm=U_warm)
            if not r.ok:
                u, U_warm = np.zeros(S.NU), None
            else:
                u, U_warm = r.u0, r.U
        else:
            u = np.zeros(S.NU)
        publish, u_ovr = (True, None) if decide is None else decide(t, r, applied.copy())
        if u_ovr is not None:
            u = np.asarray(u_ovr, float).copy()
        if publish:
            seq += 1
            msg = (seq, t + d_pub_cycles * cfg.dt, u.copy())
        rec['t'].append(t); rec['q'].append(q.copy()); rec['s'].append(s.copy())
        rec['ok'].append(None if r is None else bool(r.ok)); rec['u_req'].append(u.copy()); rec['u_app'].append(applied.copy())
        rec['published'].append(bool(publish)); rec['seq'].append(seq if publish else None)
        for _ in range(n_phys):
            pubq.append(msg if publish else None)
            m_pub = pubq.pop(0) if n_pub > 0 else (msg if publish else None)
            pipe.append(m_pub)
            while len(pipe) > max(n_del, 0):
                m = pipe.pop(0)
                if m is not None:
                    held = m
            t_phys = round(j * pdt, 9)
            if held is not None and t_phys - held[1] <= max_cmd_age + 1e-9:
                u_app, src = held[2].copy(), held[0]
            else:
                lr = WL.limit9_timeout(applied, pdt, wcfg)
                u_app = np.zeros(S.NU) if lr.u_out is None else lr.u_out
                src = None
            q, s = CL.plant_step(q, s, u_app, cfg, 1)    # 實際推進
            applied = u_app
            rec['phys'].append((j, t_phys, applied.tolist(), src))
            if on_applied is not None:
                on_applied(j, t_phys, applied.copy(), src)
            j += 1
        if publish:
            hist.append((t + d_pub_cycles * cfg.dt, u.copy()))
            if len(hist) > 64:
                hist.pop(0)
        u_prev = applied.copy()
        t += cfg.dt
    for k in ('t', 'q', 's', 'u_req', 'u_app'):
        rec[k] = np.array(rec[k])
    return rec
