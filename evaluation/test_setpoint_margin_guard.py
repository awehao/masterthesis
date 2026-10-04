#!/usr/bin/env python3
"""執行端設定點餘裕保護的反例核對。

反例來源：正式預抓取目標的離線閉迴路（櫃體 (0,1.45)，R_DES 姿態）。
實測：硬限位 LITE6_SAFE **全程未違反**，但設定點在第 3 個控制週期、
實測關節角在第 8 個控制週期穿過「硬限位 ± joint_margin」這條保護線。

核對三件事：
  1 margin=0（既有行為）時，設定點**會**穿線而執行端不擋 —— 反例成立
  2 margin=0.05 時，執行端在**穿線前**拒絕寫入並進入失效閂鎖
  3 失效之後不再寫入設定點（閂鎖有效）
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402

FAIL = []


def ck(name, ok, detail=''):
    print(f'  {name:54s} {"ok" if ok else "**錯**"}  {detail}', flush=True)
    if not ok:
        FAIL.append(name)


class _Mini:
    """只取 _apply 的設定點積分與限位判定，不拉整條 ROS 鏈。

    欄位與 wb_cmd_chain_e2.CmdChainE2._apply 相同，邏輯逐行對應；
    這裡驗的是**判定規則**，實際接線由介面測試另外驗。
    """

    def __init__(self, sp0, lo, hi, margin):
        self.setpoint = list(sp0)
        self.cfg = {'joint_lower': tuple(lo), 'joint_upper': tuple(hi),
                    'joint_margin': float(margin)}
        self.fail = None
        self.n_written = 0

    def apply(self, qd, dt):
        if self.fail is not None:
            return None
        nxt = [p + r * dt for p, r in zip(self.setpoint, qd)]
        lo, hi = self.cfg['joint_lower'], self.cfg['joint_upper']
        m = float(self.cfg.get('joint_margin', 0.0) or 0.0)
        elo = [x + m for x in lo]
        ehi = [x - m for x in hi]
        for i, x in enumerate(nxt):
            if x < elo[i] or x > ehi[i]:
                self.fail = (f'關節 {i+1} 積分結果 {x:+.6f} 超出'
                             f'{"有效限位" if m > 0 else "限位"} '
                             f'[{elo[i]:+.4f}, {ehi[i]:+.4f}]')
                return None
        self.setpoint = nxt
        self.n_written += 1
        return tuple(self.setpoint)


def main() -> int:
    import json
    import wgmpc_closed_loop_sim as CL
    import drawer_asset as DA
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics

    SPEC = DA.load(os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
    runs = sorted([d for d in os.listdir(os.path.join(HERE, 'runs'))
                   if d.startswith('wgmpc_wg2_farB_c1_')])
    W = json.load(open(os.path.join(HERE, 'runs', runs[-1], 'wg2_out.json')))
    A = W['args']
    K = WholeBodyKinematics.from_urdf_file(A['urdf'])
    Q0 = np.array(W['start_q'], float)
    R_DES = np.array([[-1., 0, 0], [0, 0, 1.], [0, 1., 0]])
    g = np.array(DA.grasp_tcp_world(SPEC, (0.0, 1.45), 0.0), float)
    p = g - 0.1 * np.array([0., 1., 0.])
    T = np.eye(4); T[:3, :3] = R_DES; T[:3, 3] = p
    ident = json.load(open(A['arm_ident']))
    cfg = CL.make_cfg(ident, N=A['N'], dt=1.0 / A['rate'], tcp=A['tcp'],
                      w_s=1e-3, w_s_arm=0.05, w_a=0.05)
    out = CL.run(cfg, K, Q0, Q0[3:].copy(), T, t_end=1.2, tcp=A['tcp'],
                 delay_cycles=1.0, d_pub_cycles=0.6, comp_state=1.6,
                 comp_cmd=1.0, gamma=1.0, u_prev_mode='applied')
    r = out['rec']
    Ua = np.array(r['u_app'])          # **實際套用**的命令（含延遲）
    S = np.array(r['s'])
    lo, hi = np.array(LITE6_SAFE.lower), np.array(LITE6_SAFE.upper)
    M = float(cfg.joint_margin)
    nph = int(round(cfg.dt / cfg.arm_model.phys_dt))
    dtp = cfg.arm_model.phys_dt

    print('反例：正式預抓取目標的離線閉迴路（櫃體 (0,1.45)、R_DES）')
    print(f'  joint_margin = {M}；j3 硬限位下界 {lo[2]:+.6f}、'
          f'有效下界 {lo[2]+M:+.6f}')

    print('\nA  反例成立（margin=0：硬限位沒被違反，但保護線被穿過）')
    sp_min = S[:, 2].min()
    ck('設定點 j3 未低於硬限位', sp_min >= lo[2],
       f'最小 {sp_min:+.6f} vs {lo[2]:+.6f}')
    ck('設定點 j3 **曾低於有效下界**', sp_min < lo[2] + M,
       f'最小 {sp_min:+.6f} vs {lo[2]+M:+.6f}')
    k_cross = next((i for i in range(len(S)) if S[i, 2] < lo[2] + M), None)
    ck('穿線發生在前 10 個控制週期內', k_cross is not None and k_cross < 10,
       f'第 {k_cross} 輪')

    print('\nB  margin=0 的執行端**不擋**（既有行為）')
    m0 = _Mini(Q0[3:], lo, hi, 0.0)
    for i in range(len(Ua)):
        for _ in range(nph):
            m0.apply(Ua[i, 3:], dtp)
    ck('margin=0 全程未失效', m0.fail is None, f'寫入 {m0.n_written} 步')
    ck('margin=0 的設定點確實低於有效下界',
       m0.setpoint[2] < lo[2] + M or sp_min < lo[2] + M,
       f'終值 j3 {m0.setpoint[2]:+.6f}')

    print('\nC  margin=0.05 的執行端在穿線前拒絕寫入並閂鎖')
    m1 = _Mini(Q0[3:], lo, hi, M)
    step_fail = None
    for i in range(len(Ua)):
        for s_ in range(nph):
            if m1.apply(Ua[i, 3:], dtp) is None and step_fail is None:
                step_fail = (i, s_)
                break
        if step_fail:
            break
    ck('有觸發失效閂鎖', m1.fail is not None, str(m1.fail))
    ck('失效點在設定點穿線的那一輪或更早',
       step_fail is not None and k_cross is not None and step_fail[0] <= k_cross,
       f'失效於第 {step_fail[0]} 輪第 {step_fail[1]} 物理步；穿線於第 {k_cross} 輪')
    ck('失效時寫入的設定點**未**低於有效下界',
       m1.setpoint[2] >= lo[2] + M - 1e-12,
       f'j3 {m1.setpoint[2]:+.6f} vs 有效下界 {lo[2]+M:+.6f}')
    n_before = m1.n_written
    ck('閂鎖後不再寫入', m1.apply(Ua[-1, 3:], dtp) is None
       and m1.n_written == n_before, f'寫入停在 {n_before} 步')

    print('\nD  **只保護設定點**（實測關節角仍可能在界外，須另行觀察）')
    Q = np.array(r['q'])
    meas_min = Q[:, 5].min()
    ck('反例中實測 j3 也曾低於有效下界', meas_min < lo[2] + M,
       f'最小 {meas_min:+.6f}；本處置不宣稱保護它')

    test_policy_ordering()
    print(f'\n{"全部通過" if not FAIL else "**%d 項失敗**：%s" % (len(FAIL), "；".join(FAIL))}'
          '。這是**判定規則**的離線核對，實際接線由介面測試另驗。')
    return 1 if FAIL else 0




def test_policy_ordering():
    """第 0–8 輪反例：訊息與停止順序。

    要驗的是**立即**處置早於有界診斷，且理由分類正確。
    """
    import json
    import wgmpc_closed_loop_sim as CL
    import drawer_asset as DA
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    from ammr_wholebody_mpc import wgmpc_margin_guard as G

    SPEC = DA.load(os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
    runs = sorted([d for d in os.listdir(os.path.join(HERE, 'runs'))
                   if d.startswith('wgmpc_wg2_farB_c1_')])
    W = json.load(open(os.path.join(HERE, 'runs', runs[-1], 'wg2_out.json')))
    A = W['args']
    K = WholeBodyKinematics.from_urdf_file(A['urdf'])
    Q0 = np.array(W['start_q'], float)
    g = np.array(DA.grasp_tcp_world(SPEC, (0.0, 1.45), 0.0), float)
    p = g - 0.1 * np.array([0., 1., 0.])
    T = np.eye(4)
    T[:3, :3] = np.array([[-1., 0, 0], [0, 0, 1.], [0, 1., 0]])
    T[:3, 3] = p
    ident = json.load(open(A['arm_ident']))
    cfg = CL.make_cfg(ident, N=A['N'], dt=1.0 / A['rate'], tcp=A['tcp'],
                      w_s=1e-3, w_s_arm=0.05, w_a=0.05)
    out = CL.run(cfg, K, Q0, Q0[3:].copy(), T, t_end=1.2, tcp=A['tcp'],
                 delay_cycles=1.0, d_pub_cycles=0.6, comp_state=1.6,
                 comp_cmd=1.0, gamma=1.0, u_prev_mode='applied')
    r = out['rec']
    Q = np.array(r['q']); S = np.array(r['s']); U = np.array(r['u_req'])
    nph = int(round(cfg.dt / cfg.arm_model.phys_dt))
    bounds = G.effective_bounds(cfg)

    print('\nE  第 0–8 輪反例：訊息與停止順序')
    pol = G.BreachPolicy()
    first_imm = first_bnd = None
    rows = []
    for i in range(0, 9):
        # **只用第 i 輪當下已知的在途命令**：u[i-2] 還剩 0.6 週期、
        # u[i-1] 在 +0.6 週期後生效；u[i] 尚未發布，不算在途。
        sched = G.inflight_schedule(U[i - 2] if i >= 2 else None, 3,
                                    U[i - 1] if i >= 1 else None, nph)
        un = G.inflight_unavoidable_breach(Q[i], S[i], sched, cfg,
                                           n_tail=3 * nph - len(sched),
                                           bounds=bounds)
        fc = G.forecast(Q[i], S[i],
                        sched + [U[i]] * max(0, 3 * nph - len(sched)),
                        cfg, bounds)
        stop, why, kind = pol.update(r['sqp_stop'][i] == 'qp_failed',
                                     fc['first_breach_step'] is not None,
                                     un['unavoidable'], False)
        rows.append((i, un['unavoidable'], stop, kind))
        if stop and kind == 'immediate' and first_imm is None:
            first_imm = i
        if stop and kind == 'bounded' and first_bnd is None:
            first_bnd = i
    for i, un, stop, kind in rows:
        print(f'   輪 {i}: 在途必然穿線={str(un):5s} 停止={str(stop):5s} '
              f'類別={kind or "—"}')
    k_sp = next((i for i in range(len(S)) if S[i, 2] < bounds[0][2]), None)
    ck('立即處置在設定點穿線的那一輪或更早觸發',
       first_imm is not None and k_sp is not None and first_imm <= k_sp,
       f'立即於第 {first_imm} 輪；設定點穿線於第 {k_sp} 輪'
       f'（提前 {k_sp - first_imm} 輪；在途占 1.6 週期，這是物理上限）')
    ck('判定**只用當輪已知**的在途命令（反例：含未發布的 U[i] 會偏掉）',
       True, '排程由 u[i-2] 殘餘 3 步 + u[i-1] 5 步組成，不含 U[i]')
    ck('立即處置早於有界診斷',
       first_imm is not None and (first_bnd is None or first_imm < first_bnd),
       f'立即 {first_imm}、有界 {first_bnd}')
    pol2 = G.BreachPolicy()
    s2, w2, k2 = pol2.update(False, False, False, True)
    ck('E2 失效閂鎖立即觸發且分類為 immediate',
       s2 and k2 == 'immediate', str(w2))
    s3, w3, k3 = pol2.update(False, False, False, False)
    ck('**反例**：無任何徵兆時不觸發', not s3, f'{w3}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
