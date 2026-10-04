#!/usr/bin/env python3
"""W-GMPC 的**預測準確度**核對：它預測接下來會到哪，與實際到哪差多少。

為什麼要做：MPC 與一般回授控制的差別就在它會往前預測 N 步。但到目前為止
從來沒有比對過預測與實際。若預測贏不過最笨的基準，那「預測」這件事
並沒有在發揮作用，這個控制器實質上只是個有約束的單步回授。

做法：用趟次紀錄的 `solve_in`（求解器當輪**實際收到**的 15 維狀態與 u_prev）
重跑 `solve_sp`，取回 `Q_pred`（N+1 步的 9 維狀態預測），再與 sim log 中
t + k·dt 時刻的**實測狀態**比對。

**三個基準**（都用同一組時刻與同一個誤差定義）：
  hold   把當下狀態維持不變（零階保持）—— 最笨
  const  以當下實測速度做等速外推
  mpc    求解器自己的非線性 rollout
贏不過 hold 就代表預測沒有加值。
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wgmpc_core_sp import (make_z, solve_sp,        # noqa: E402
                                              plant_phys_step)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import wgmpc_closed_loop_sim as CL                                     # noqa: E402

ARMJ = ['joint%d' % (i + 1) for i in range(6)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--max-cycles', type=int, default=0, help='0 = 全部')
    g = ap.parse_args()

    w = json.load(open(g.run + '/wg2_out.json'))
    s = json.load(open(g.run + '/sim/wb_run.json'))
    a = w['args']
    ident = json.load(open(a['arm_ident']))
    cfg = CL.make_cfg(ident, N=a['N'], dt=1.0 / a['rate'], tcp=a['tcp'])
    K = WholeBodyKinematics.from_urdf_file(a['urdf'])
    T_des = np.eye(4)
    T_des[:3, 3] = np.array(w['target_tcp'], float)
    T_des[:3, :3] = K.fk(np.array(w['start_q'], float), a['tcp'])[:3, :3]

    ci = {c: i for i, c in enumerate(s['log_cols'])}
    L = np.array(s['log'], float)
    ts = L[:, ci['t']]

    def state_at(t):
        k = int(np.argmin(np.abs(ts - t)))
        q = np.array([L[k, ci['base_x']], L[k, ci['base_y']], L[k, ci['base_yaw']]]
                     + [L[k, ci['%s_act' % j]] for j in ARMJ], float)
        return q, float(ts[k])

    def tcp_of(q):
        return K.fk(q, a['tcp'])[:3, 3]

    pub = [x for x in w['log'] if x.get('published') and x.get('solve_in')]
    if g.max_cycles:
        pub = pub[:g.max_cycles]
    N, dt = a['N'], 1.0 / a['rate']
    t_end = ts[-1]

    rows = []
    for x in pub:
        si = x['solve_in']
        t0 = float(si['snap_sim_t_exact'])
        if t0 + N * dt > t_end:
            continue
        dsc_dt = float(a.get('delay_comp_state_cycles', 0.0)) * dt
        qp = np.array(si['q_pred'], float)
        spp = np.array(si['s_pred'], float)
        up = np.array(si['u_prev'], float)
        if len(spp) != 6:
            continue
        r = solve_sp(K, make_z(qp, spp), up, T_des, cfg, U_warm=None)
        if r.Q_pred is None:
            continue
        # ---- 變體：用**實際被套用的命令**而非求解器的計畫 U 做 rollout ----
        # 這把「模型本身準不準」與「計畫有沒有被照著執行」分開：
        # 若換成實際命令後誤差大幅下降，代表模型沒問題，
        # 問題在計畫與執行不一致（整形、安全層、延遲）。
        qa, sa = qp.copy(), spp.copy()
        Q_act = [qa.copy()]
        nph = int(round(dt / cfg.arm_model.phys_dt))
        for kk in range(N):
            tq = float(si["snap_sim_t_exact"]) + dsc_dt + kk * dt
            j = int(np.argmin(np.abs(ts - tq)))
            u_app = np.array([L[j, ci['vx_cmd']], L[j, ci['vy_cmd']], L[j, ci['wz_cmd']]]
                             + [0.0] * 6, float)
            # 手臂實際套用速率由設定點差分還原
            j2 = min(j + 1, len(L) - 1)
            u_app[3:] = [(L[j2, ci['%s_sp' % jn]] - L[j, ci['%s_sp' % jn]])
                         / max(ts[j2] - ts[j], 1e-9) for jn in ARMJ]
            qa, sa = plant_phys_step(qa, sa, u_app, cfg, nph)
            Q_act.append(qa.copy())
        Q_act = np.array(Q_act)
        q_now, _ = state_at(t0)
        # 等速外推用實測速度
        k0 = int(np.argmin(np.abs(ts - t0)))
        k1 = max(k0 - 1, 0)
        dtm = max(ts[k0] - ts[k1], 1e-9)
        qdot = (np.array([L[k0, ci['base_x']], L[k0, ci['base_y']], L[k0, ci['base_yaw']]]
                         + [L[k0, ci['%s_act' % j]] for j in ARMJ], float)
                - np.array([L[k1, ci['base_x']], L[k1, ci['base_y']], L[k1, ci['base_yaw']]]
                           + [L[k1, ci['%s_act' % j]] for j in ARMJ], float)) / dtm
        # **時間對齊**：q_pred 是延遲補償後、命令**生效時刻**的狀態
        # （t0 + d_state·dt），所以 Q_pred[k] 對應 t0 + d_state·dt + k·dt。
        # 直接拿 t0 + k·dt 比會整體偏掉一個補償量（實測 0.08 s）。
        t_eff = t0 + float(a.get('delay_comp_state_cycles', 0.0)) * dt
        for k in range(0, N + 1):
            q_act, t_act = state_at(t_eff + k * dt)
            p_act = tcp_of(q_act)
            # 基準也用同一組時刻，且都從**實測當下**出發（公平）
            lead = (t_eff + k * dt) - t0
            e_mpc = float(np.linalg.norm(tcp_of(r.Q_pred[k]) - p_act))
            e_hold = float(np.linalg.norm(tcp_of(q_now) - p_act))
            e_const = float(np.linalg.norm(tcp_of(q_now + qdot * lead) - p_act))
            e_actu = float(np.linalg.norm(tcp_of(Q_act[k]) - p_act))
            rows.append((k, e_mpc, e_hold, e_const, e_actu))

    R = np.array(rows, float)
    out = {'run': os.path.basename(g.run.rstrip('/')), 'n_cycles': len(pub),
           'N': N, 'dt': dt, 'per_step': {}}
    print(f"趟次 {out['run']}  視界 N={N} × dt={dt:.2f} s；比對 {len(pub)} 輪")
    dsc = float(a.get('delay_comp_state_cycles', 0.0))
    print(f"  補償 {dsc} 週期 ⇒ Q_pred[k] 對應實際時刻 t0 + {dsc*dt:.3f} + k×{dt:.2f} s")
    print(f"\n{'步':>3} {'自 t0 起 s':>10} {'MPC 預測誤差 mm':>15} {'零階保持':>10} {'等速外推':>10} "
          f"{'模型+實際命令':>13} {'勝過保持':>9}")
    for k in range(0, N + 1):
        m = R[:, 0] == k
        em, eh, ec, ea = (R[m, 1] * 1e3, R[m, 2] * 1e3, R[m, 3] * 1e3, R[m, 4] * 1e3)
        out['per_step'][k] = dict(
            n=int(m.sum()), mpc_p50=float(np.median(em)), hold_p50=float(np.median(eh)),
            const_p50=float(np.median(ec)), actualcmd_p50=float(np.median(ea)),
            win_hold=float(np.mean(em < eh)), win_const=float(np.mean(em < ec)))
        print(f'{k:>3} {dsc*dt+k*dt:10.3f} {np.median(em):15.3f} {np.median(eh):10.3f} '
              f'{np.median(ec):10.3f} {np.median(ea):13.3f} {np.mean(em<eh):9.1%}')
    kN = out['per_step'][N]
    print(f"\n視界末端（自 t0 起 {dsc*dt+N*dt:.3f} s）："
          f"MPC {kN['mpc_p50']:.2f} mm、保持 {kN['hold_p50']:.2f} mm、"
          f"等速 {kN['const_p50']:.2f} mm、模型+實際命令 {kN['actualcmd_p50']:.2f} mm")
    print(f"  相對零階保持的改善倍數 {kN['hold_p50']/max(kN['mpc_p50'],1e-9):.2f}x；"
          f"相對等速外推 {kN['const_p50']/max(kN['mpc_p50'],1e-9):.2f}x")
    print('\n**界線**：這是「模型預測 vs 實際」的開迴路比對。'
          '預測準不代表控制好，預測不準也不代表控制壞 —— '
          '它只回答「預測這件事有沒有在發揮作用」。')
    if g.json_out:
        json.dump(out, open(g.json_out, 'w'), indent=1, ensure_ascii=False)
        print(f'寫出 {g.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
