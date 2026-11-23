#!/usr/bin/env python3
"""b2（Codex 20261031_162054）：交棒准入框與求解容量框分離的非物理測試。
  rt_44 B 段放行時本體實測速度 (0.06748, 0.01774, −0.35)：落在求解框 ×2（0.07051／0.3998）之內、舊准入框（0.035255／0.1999）之外。
  A. 編排端 NavHandoverConfig（drawer_mission_node2 --v-box-* 恢復舊值）：此速度 ⇒ 拒絕；框內且在動 ⇒ 通過。
  B. 執行端 DualSourceExecutor 切換當步預核（ev9r9a --v-box-* 恢復舊值）：此速度在**提交當步**出現 ⇒ 取消切換、控制權留導航；
     框內 ⇒ 正常切換，種子取導航實際套用回報（套用歷史承接）。
  C. 跑法 roundtrip4 生效值：執行端准入＝編排端准入（兩處 drawer_mission_node2 都同值）、准入 ⊆ 求解可行框、容量框（權威／入口）＝求解框。
FakeChain／FakeSnap 取自 test_dual_source_executor.py 原始碼（只取類別定義，不執行該測試）。
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from control_authority import AUTH_NAV, AUTH_WHOLEBODY                 # noqa: E402
from dual_source_executor import DualSourceExecutor                    # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import WGMPCConfig                  # noqa: E402
from ammr_wholebody_mpc.wgmpc_handover import (NH_OK, NavHandoverConfig, NavHandoverState,  # noqa: E402
                                               check_nav_handover)

_src = open(os.path.join(HERE, 'test_dual_source_executor.py'), encoding='utf-8').read()
_tree = ast.parse(_src)
_g = {'np': __import__('numpy')}
for node in _tree.body:
    if isinstance(node, ast.ClassDef) and node.name in ('FakeSnap', 'FakeChain'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'test_dual_source_executor.py', 'exec'), _g)
FakeChain = _g['FakeChain']

OLD = (0.035255, 0.1999)
V_RT44 = (0.06748, 0.01774, -0.35)
STOW = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LO = (-6.283185, -2.617994, -0.061087, -6.283185, -2.164208, -6.283185)
HI = (6.283185, 2.617994, 2.935644, 6.283185, 2.164208, 6.283185)
BAD, N = [], 0


def chk(name, cond, info=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name} {info}')


# ---- A 編排端
cfg = NavHandoverConfig(joint_lower=LO, joint_upper=HI, joint_margin=0.05, max_state_age_s=0.1,
                        v_box_lin_mps=OLD[0], v_box_ang_rps=OLD[1])


def st(vx, vy, wz):
    return NavHandoverState(t_now=50.0, nav_in_control=True, wgmpc_armed=False, nav_u_applied=(vx, vy, wz, 0, 0, 0, 0, 0, 0),
                            nav_applied_t=49.98, nav_applied_step=4998, base_vx_body=vx, base_vy_body=vy, base_yaw_rate_rps=wz,
                            base_pose_t=49.98, park_dist_m=0.21, reserved_clearance_m=0.28, q_meas=STOW, q_meas_t=49.98, stow_q=STOW,
                            exec_mode=4, n_recv=0, n_rejected=0, exec_report_t=49.98, setpoint=None)


va = check_nav_handover(st(*V_RT44), cfg)
chk('A1 rt_44 放行速度在舊准入框外 ⇒ 編排端拒絕', not va.ok, f'code={va.code}')
vb = check_nav_handover(st(0.030, -0.004, 0.05), cfg)
chk('A2 框內且在動 ⇒ 通過', vb.ok and vb.code == NH_OK, f'code={vb.code}')
vc = check_nav_handover(st(0.030, -0.004, -0.25), cfg)
chk('A3 只有偏航率超舊框（−0.25，仍在 ×2 求解框內）⇒ 拒絕', not vc.ok, f'code={vc.code}')


# ---- B 執行端：提交當步重新核對
def run(vb_at_switch, switch_at=8, steps=16):
    e = DualSourceExecutor(FakeChain(ready_from=0.0), stow_setpoint=STOW, v_box_lin=OLD[0], v_box_ang=OLD[1], v_min_lin=0.010)
    for k in range(steps):
        e.chain.tick(k * 0.01)
        vb_ = vb_at_switch if k >= switch_at - 1 else (0.020, 0.0, 0.0)
        e.step(k, k * 0.01, 0.01, nav_cmd=(0.020, 0.0, 0.0), q_arm_measured=list(STOW), meas_vb=vb_)
        if k == 2:
            e.request_handover(AUTH_WHOLEBODY, at_step=switch_at, sim_t=k * 0.01)
    return e


e1 = run(V_RT44)
chk('B1 提交當步實測＝rt_44 放行速度 ⇒ 取消切換', e1.auth.switch_step is None, str(e1.auth.switch_step))
chk('B2 控制權留在導航', e1.owner == AUTH_NAV, str(e1.owner))
chk('B3 取消理由指向速度框', any('超出速度框' in str(x) or '超框' in str(x) for x in e1.events), str(e1.events[-1:]))
e2 = run((0.020, 0.010, 0.05))
chk('B4 框內且在動 ⇒ 第 8 步切換', e2.auth.switch_step == 8, str(e2.auth.switch_step))
chk('B5 切換種子取導航實際套用回報（套用歷史承接）', e2.chain.seeded is not None, str(e2.chain.seeded))
e3 = run((0.020, 0.010, -0.25))
chk('B6 只有偏航率在舊框外 ⇒ 取消', e3.auth.switch_step is None, str(e3.auth.switch_step))

# ---- C 跑法 roundtrip4 生效值
R4 = open(os.path.join(HERE, 'run_mission_roundtrip4.sh'), encoding='utf-8').read()
sim = [ln for ln in R4.splitlines() if 'spawn sim' in ln and 'isaac_door_k2_sim_ev9r9a.py' in ln][0]
dm = [ln for ln in R4.splitlines() if 'spawn mission_' in ln and 'drawer_mission_node2.py' in ln]


def opt(ln, o):
    m = re.search(re.escape(o) + r' ([0-9.]+)', ln)
    return None if m is None else float(m.group(1))


ex_adm = (opt(sim, '--v-box-lin'), opt(sim, '--v-box-ang'))
dm_adm = {(opt(ln, '--v-box-lin'), opt(ln, '--v-box-ang')) for ln in dm}
cap = (opt(sim, '--wb-box-lin'), opt(sim, '--wb-box-ang'))
k = {float(m) for m in re.findall(r'--base-box-scale ([0-9.]+)', R4)}
c = WGMPCConfig()
c.base_box_scale = k.pop() if len(k) == 1 else float('nan')
vb_sol = c.vbox()
chk('C1 執行端准入恢復舊框', ex_adm == OLD, str(ex_adm))
chk('C2 編排端兩處准入同值且＝執行端', len(dm) == 2 and dm_adm == {OLD}, str(dm_adm))
chk('C3 准入 ⊆ 求解可行框', ex_adm[0] <= vb_sol[0] and ex_adm[1] <= vb_sol[2], f'{ex_adm} vs {vb_sol[:3]}')
chk('C4 容量框（權威／入口）＝求解框', abs(cap[0] - vb_sol[0]) < 1e-9 and abs(cap[1] - vb_sol[2]) < 1e-9, f'{cap} vs {vb_sol[:3]}')
chk('C5 執行端合速度界 ≥ √2×求解框、角速度界 ≥ 求解框', opt(sim, '--base-lin-max') >= 2 ** 0.5 * vb_sol[0] and opt(sim, '--base-ang-max') >= vb_sol[2])
print(f'b2 准入框測試 {N} 項：' + ('全過' if not BAD else '失敗 ' + '；'.join(BAD)))
sys.exit(0 if not BAD else 1)
