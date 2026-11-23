#!/usr/bin/env python3
"""wgmpc_core_sp_fs（候選首筆承接約束）驗證。構型取自內側把手串接 v1 static 各階段列；目標＝該列 TCP 平移 2 cm、轉 0.05 rad。
  F1 回歸：first_step=None 時 solve_sp_fs 與原 wgmpc_core_sp.solve_sp 的 ok、reason、u0、U、max_residual 逐位元相同（另確認核心檔未改：與 git HEAD 內容一致）；
  F2 啟用：u_applied＝原解 u0 加小偏差（框內可承接）⇒ ok、|u0 − u_applied| ≤ a_max×0.01＋r_tol、residual_by_block 含 first_step_accept；
  F3 不可承接：u_applied 手臂 2.0 rad/s（速度框 1.0）⇒ 不得 ok（不裁切）；
  F4 只約束首筆：啟用時 |u1 − u0| 仍可 > a_max×0.01（後續步以 cfg.dt 0.05 的框）——找到至少一例；
  F5 參數：u_applied 非 9 維／含 NaN、dt_phys ≤ 0 ⇒ ValueError。

    python3 evaluation/test_wgmpc_core_sp_fs.py
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp_fs as FS             # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

fails = []


def ok(c, m):
    if not c:
        fails.append(m)


core = os.path.join(WS, 'src', 'ammr_wholebody_mpc', 'ammr_wholebody_mpc', 'wgmpc_core_sp.py')
head = subprocess.run(['git', '-C', WS, 'show', 'HEAD:src/ammr_wholebody_mpc/ammr_wholebody_mpc/wgmpc_core_sp.py'], capture_output=True).stdout
ok(head == open(core, 'rb').read(), 'F1 核心檔與 HEAD 不同')
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
rows = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_inner_v1_static.json')))['rows']
am = S.ArmSetpointModel(alpha=ident['alpha'], bias=ident['bias_rad'], phys_dt=ident['phys_dt_measured_s'])
kw = dict(N=5, dt=0.05, arm_model=am, row_scaling=True, w_s=0.001, w_a=0.05, w_s_arm=0.05)
cfg0 = S.WGMPCConfigSP(**kw)
alpha, bias = np.asarray(ident['alpha']), np.asarray(ident['bias_rad'])
amax = cfg0.amax()
f4 = False
for r in rows[::25]:
    q = np.array(r['x'], float)
    s = q[3:] - bias / alpha
    z = S.make_z(q, s)
    T = K.fk(q, 'link_tcp').copy()
    T[:3, 3] += [0.02, -0.01, 0.01]
    c, sn = np.cos(0.05), np.sin(0.05)
    T[:3, :3] = T[:3, :3] @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    up = np.zeros(9)
    a = S.solve_sp(K, z, up, T, cfg0)
    b = FS.solve_sp_fs(K, z, up, T, FS.WGMPCConfigSPFS(**kw))
    same = (a.ok == b.ok and a.reason == b.reason and a.max_residual == b.max_residual
            and ((a.u0 is None and b.u0 is None) or np.array_equal(a.u0, b.u0)) and ((a.U is None and b.U is None) or np.array_equal(a.U, b.U)))
    ok(same, f"F1 {r['stage']} 不同：{a.ok}/{b.ok} {a.reason}/{b.reason}")
    if not a.ok:
        continue
    ua = np.asarray(a.u0) + np.r_[0.002, -0.001, 0.003, 0.05, -0.04, 0.03, 0.0, 0.02, -0.01]
    ua = np.clip(ua, -cfg0.vmax() * 0.9, cfg0.vmax() * 0.9)
    c2 = FS.WGMPCConfigSPFS(**kw, first_step=(ua, 0.01))
    r2 = FS.solve_sp_fs(K, z, up, T, c2)
    ok(r2.ok and 'first_step_accept' in r2.residual_by_block, f"F2 {r['stage']} ok={r2.ok} {r2.reason}")
    if r2.ok:
        ok(np.all(np.abs(np.asarray(r2.u0) - ua) <= amax * 0.01 + 1e-6), f"F2 {r['stage']} 首筆超框")
        if np.any(np.abs(np.asarray(r2.U[1]) - np.asarray(r2.U[0])) > amax * 0.01 + 1e-6):
            f4 = True
    ub = np.zeros(9)
    ub[3] = 2.0
    r3 = FS.solve_sp_fs(K, z, up, T, FS.WGMPCConfigSPFS(**kw, first_step=(ub, 0.01)))
    ok(not r3.ok, f"F3 {r['stage']} 不可承接卻 ok")
ok(f4, 'F4 未見後續步超過首筆框（後續應以 cfg.dt 為框）')
for bad in (((np.zeros(8), 0.01)), ((np.r_[np.nan, np.zeros(8)], 0.01)), ((np.zeros(9), 0.0))):
    try:
        FS.solve_sp_fs(K, S.make_z(np.array(rows[0]['x']), np.array(rows[0]['x'][3:]) - bias / alpha), np.zeros(9), np.eye(4),
                       FS.WGMPCConfigSPFS(**kw, first_step=bad))
        ok(False, f'F5 未拒 {bad}')
    except ValueError:
        pass
print('全部通過' if not fails else f'失敗 {len(fails)}', f'（構型 {len(rows[::25])}）')
for f in fails:
    print('  ', f)
sys.exit(1 if fails else 0)
