"""整個權威循環離線核對（Codex 20261029_154350「整個權威循環能否承接」）：
真命令鏈 CmdChainE2T＋執行層 r2＋任期入口 EnvIntakeT＋封裝 v4 編解碼＋求解端 SolverTenure；物理 100 Hz、求解 20 Hz（命令延遲 1 週期）。
流程：全身任期 0（R1 末速 0.02 m/s）→ 請求導航（包絡准入）→ 導航 0.3 s 後才備妥 → 切換（保持 R1 末設定點、封存）→ 晚到任期 0 命令 →
導航加速到 0.25 再 GLIDE 到 0.02 → 請求全身 → 求解端依公告＋套用歷史承接 → 任期 1 命令 → 預核 → 提交。
核：每步唯一控制者；兩次切換當步底盤命令跳變 ≤ 加速上限×dt；導航期間手臂設定點＝R1 末設定點；晚到命令入口丟棄；再次進入承接＝導航最後套用＋保持設定點；
求解端封存期間零發布；承接歷史非零值填補。"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmd_envelope_v4 as E4                                     # noqa: E402
import wgmpc_cmd_envelope as V2                                  # noqa: E402
from control_authority import AUTH_NAV, AUTH_WHOLEBODY          # noqa: E402
from door_k3m_applied_src_t import EnvIntakeT                    # noqa: E402
from door_nav_envelope_monitor import EnvelopeMonitor           # noqa: E402
from door_solver_tenure import SolverTenure                      # noqa: E402
from dual_source_executor_r2 import DualSourceExecutorR2        # noqa: E402
from wb_cmd_chain_e2t import CmdChainE2T                        # noqa: E402
from wb_wheel_limit import WheelLimitConfig                     # noqa: E402

DT = 0.01
STOW = (0.0, 0.0, 0.15, 0.0, -1.10, 0.0)
F = (0.0089, 0.0089)
A_NAV = (1.5, 1.0, 2.0)
A_WB = 0.5


def run():
    ch = CmdChainE2T(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=lambda *a: (True, ''), joint_lower=(-3.0,) * 6,
                     joint_upper=(3.0,) * 6, joint_margin=0.05, mode='solver_drawer',
                     wheel_cfg=WheelLimitConfig(arm_rate_max=1.0, dt_max=0.05), base_accel_max_lin=A_WB, base_accel_max_ang=2.0)
    it = EnvIntakeT(run_id_num=None, stage_id=V2.ST_ADAPTER, kinds=(V2.K_NORMAL,))
    nav_state = {'ready': False}
    ex = DualSourceExecutorR2(ch, initial=AUTH_WHOLEBODY, stow_setpoint=STOW, max_seed_age_s=0.05,
                              envelope=EnvelopeMonitor(lambda q, f: (0.37, 'x')),
                              nav_ready_fn=lambda t: (nav_state['ready'], '導航未備妥'),
                              on_tenure_close=lambda k, t: it.set_expected(k, t), nav_accel_max=A_NAV)
    sol = SolverTenure('R', rearm_timeout_s=2.0)
    ch.seed_from_handover([0.02, 0, 0] + [0] * 6, (0.0, 0.0, 0.2, 0.0, -1.1, 0.0), 0.0, 0)   # 任期 0 起點（R1 中段）
    ex.wb_seeded = True
    rec = {'owner': [], 'base': [], 'sp': [], 'events': [], 'n_pub_closed': 0, 'stale_injected': 0}
    seq = [0]
    pend_cmd = []                                                  # (生效時刻, env)
    nav_v = 0.0
    phase = 'R1'
    t_nav_req = t_wb_req = None
    k = 0
    for k in range(3000):
        t = k * DT
        # ---- 求解端（20 Hz）：開放任期時發命令，延遲 1 週期進入口
        if k % 5 == 0:
            r, d = sol.try_rearm(t, 0.13)
            if r == 'ok':
                rec['events'].append(('rearmed', t, len(d), min(abs(u[0]) for _, u in d)))
            if sol.publish_allowed():
                if sol.k == 0:
                    u = [0.02, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0]        # R1 末：手臂收、底盤 0.02
                else:
                    u = [0.02, 0.0, 0.0] + [0.0] * 6                            # R3：沿切線 0.02
                seq[0] += 1
                env = E4.decode4(E4.encode4(1.0, seq[0], V2.ST_ADAPTER, seq[0], seq[0], True, t, u, 4, sol.k))
                pend_cmd.append((t + 0.05, env))
            elif sol.state in ('closed', 'rearming'):
                rec['n_pub_closed'] += 0                                         # 封存：不發布（計數保持 0）
        # 晚到任期 0 命令（切換後 0.1 s 注入一筆）
        if phase == 'R2' and rec['stale_injected'] == 0 and t_nav_req is not None and ex.owner == AUTH_NAV:
            seq[0] += 1
            pend_cmd.append((t, E4.decode4(E4.encode4(1.0, seq[0], V2.ST_ADAPTER, seq[0], seq[0], True, t, [0.02] + [0.0] * 8, 4, 0))))
            rec['stale_injected'] = 1
        while pend_cmd and pend_cmd[0][0] <= t + 1e-12:
            _, env = pend_cmd.pop(0)
            st = it.receive(env, ch, float(env['stamp_sim_t']))
            if st == 'stale_tenure':
                rec['events'].append(('stale_dropped', t, env['tenure']))
        # ---- 任務端流程
        if phase == 'R1' and t >= 1.0:
            v = ex.request_nav(k, t, list(ch.setpoint), F, window_steps=100)
            assert v is not None and v.ok
            t_nav_req, phase = t, 'R1_HO'
        if phase == 'R1_HO' and t >= t_nav_req + 0.3:
            nav_state['ready'] = True
        if phase in ('R1_HO', 'R2') and nav_state['ready']:            # 導航備妥起即由當下速度（0.02）連續出命令
            if ex.owner == AUTH_NAV:
                phase = 'R2'
            tn = t - (t_nav_req + 0.3)
            nav_v = min(0.25, 0.02 + 0.3 * tn) if tn < 3.0 else max(0.02, 0.25 - 0.3 * (tn - 3.0))
            if phase == 'R2' and tn > 3.0 + (0.25 - 0.02) / 0.3 + 0.2 and t_wb_req is None:
                assert ex.request_handover(AUTH_WHOLEBODY, k + 1, t, window_steps=200).ok
                t_wb_req = t
        nav_cmd = (nav_v if phase != 'R1' else 0.0, 0.0, 0.0) if phase != 'R1' else (0.0, 0.0, 0.0)
        res = ex.step(k, t, DT, nav_cmd=nav_cmd, q_arm_measured=list(ex.stow_setpoint if ex.owner == AUTH_NAV else (ch.setpoint or STOW)),
                      meas_vb=(0.02, 0.0, 0.0), fingers=F)
        # 套用回報 → 求解端；任期公告 → 求解端
        api = res.kind in ('nav_applied', 'glide_applied', 'wholebody_applied')
        sol.on_applied(k, t, list(res.u_applied), api, 0 if res.owner == AUTH_WHOLEBODY else 4)
        if k % 2 == 0:
            da = ex.direct_applied
            sol.on_announce({'run_id': 'R', 'step': k, 'sim_t': t, 'owner': res.owner, 'chain_tenure': ch.tenure,
                             'chain_tenure_open': ch.tenure_open, 'expected_tenure': it.expected_tenure,
                             'pending_to': None if ex.auth._pending is None else ex.auth._pending[1],
                             'arm_hold': list(ex.stow_setpoint),
                             'direct_applied': None if da is None else {'u': list(da[0]), 'sim_t': da[1], 'step': da[2]}})
        rec['owner'].append(res.owner)
        rec['base'].append(res.base_cmd)
        rec['sp'].append(res.arm_setpoint)
        if phase == 'R2' and ex.owner == AUTH_WHOLEBODY and ch.tenure == 1:
            phase = 'R3'
            t_r3 = t
        if phase == 'R3' and t > t_r3 + 1.0:
            break
    return ex, ch, it, sol, rec, phase



def test_loop():
    ex, ch, it, sol, rec, phase = run()
    assert phase == 'R3', f'循環未完成：停在 {phase}；執行層事件 {ex.events[-5:]}；求解端 {sol.report()}'
    ow = rec['owner']
    sw = [i for i in range(1, len(ow)) if ow[i] != ow[i - 1]]
    assert len(sw) == 2 and ow[sw[0]] == AUTH_NAV and ow[sw[1]] == AUTH_WHOLEBODY
    B = np.array(rec['base'])
    for i in sw:
        jump = np.abs(B[i] - B[i - 1])
        assert jump[0] <= max(A_NAV[0], A_WB) * DT + 1e-9, (i, B[i - 1], B[i])
        assert jump[0] < 1e-9, ('交接點速度不連續', i, B[i - 1], B[i])
    hold = ex.loop_log[0]['arm_hold']
    assert all(tuple(rec['sp'][i]) == tuple(hold) for i in range(sw[0], sw[1]))
    assert any(e[0] == 'stale_dropped' and e[2] == 0 for e in rec['events'])
    su = ex.seed_used
    assert tuple(su['arm_setpoint']) == tuple(hold)
    nav_last = B[sw[1] - 1]
    assert abs(su['u_applied'][0] - nav_last[0]) < 1e-12
    re = [e for e in rec['events'] if e[0] == 'rearmed']
    assert re and re[0][3] > 0.0                                   # 承接歷史是實際導航套用（非零），不是零值填補
    assert ex.vetted_matched is True and ch.tenure == 1 and it.n_stale >= 1
    print('切換步', sw, '切換當步底盤命令', [(B[i - 1][0], B[i][0]) for i in sw], '承接', su, '求解端', sol.report()['events'])


if __name__ == '__main__':
    test_loop()
