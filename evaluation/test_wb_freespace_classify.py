"""自由空間狀態分類的離線測試：不開模擬器、不含 ROS。

要驗的是**分類**，不是門檻大小：

  A 資料未知（未確認）→ NODATA 仍套 nodata_speed_cap
  B 已確認自由空間     → NODATA 不套 cap，其餘限制全留
  C 過期資料           → 即使宣告自由空間，STALE 的上限仍生效
  D 上下游輸入一致     → 求解端（空點集合）與下游（11 列 NODATA ＋ 已確認）
                        得到相同的有效速度框
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, 'install/ammr_wholebody_mpc/lib/python3.12/site-packages')
sys.path.insert(0, 'evaluation')
from ammr_wholebody_mpc.wholebody_safety_filter import (             # noqa: E402
    STATUS_NODATA, STATUS_STALE, DetectionPoint, SafetyConfig,
    _box_rows, _rows_from_points, filter_velocity)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa
from ammr_wholebody_mpc.arm_link_geometry import arm_link_names      # noqa: E402
from wb_qp_lowspeed import lowspeed_cfg                              # noqa: E402

URDF = 'evaluation/models/omni_bot_wholebody_expanded.urdf'
K = WholeBodyKinematics.from_urdf_file(URDF)
LINKS = arm_link_names(open(URDF).read())
DT = 0.049608

fails = []
def expect(n, c, d=''):
    print(f"  [{'ok' if c else '**FAIL**'}] {n}{(' — ' + d) if d else ''}")
    if not c:
        fails.append(n)


def pts_of(status):
    return [DetectionPoint(frame=n, p=np.zeros(3), n=np.array([1., 0, 0]),
                           d=0.0, status=int(status), age=0.0, occluded=False,
                           offset=np.zeros(3), rho=0.015) for n in LINKS]


def cfg_of(freespace):
    c = lowspeed_cfg(SafetyConfig(alpha=2.0, d0=0.05, tau=0.15, a_brake=1.0,
                                  eps=0.03, dt=DT, fix_base=False))
    c.freespace_confirmed = freespace
    return c


# 取真實趟次的代表性狀態
d = json.load(open('evaluation/runs/wb_solver_iso_094526/solver_out.json'))
lg = d['log']
k = len(lg) // 2
r, rp, rp2 = lg[k], lg[k - 1], lg[k - 2]
q9 = np.array(list(r['base']) + list(r['q']))
v_in = np.array(r['cmd_in'])
obs = np.array(r['cmd_out'])
v_prev = np.array(rp['cmd_out'])
a_prev = (v_prev - np.array(rp2['cmd_out'])) / DT

print('A 資料未知（未確認自由空間）：NODATA 仍套上限')
_, _, cap_a, _ = _rows_from_points(K, q9, pts_of(STATUS_NODATA), cfg_of(False), v_in)
expect('A1 cap = nodata_speed_cap', abs(cap_a - 0.05) < 1e-12, f'cap={cap_a}')
va = np.asarray(filter_velocity(K, q9, v_in, pts_of(STATUS_NODATA),
                                cfg=cfg_of(False), v_prev=v_prev,
                                a_prev=a_prev, dt=DT).v)
expect('A2 重現實際趟次的手臂受抑制', np.abs(va - obs).max() < 1e-6,
       f'與觀測差 {np.abs(va - obs).max():.2e}')

print('B 已確認自由空間：NODATA 不套 cap，其餘限制全留')
_, _, cap_b, _ = _rows_from_points(K, q9, pts_of(STATUS_NODATA), cfg_of(True), v_in)
expect('B1 cap = inf', not np.isfinite(cap_b), f'cap={cap_b}')
vb = np.asarray(filter_velocity(K, q9, v_in, pts_of(STATUS_NODATA),
                                cfg=cfg_of(True), v_prev=v_prev,
                                a_prev=a_prev, dt=DT).v)
expect('B2 手臂命令通過', np.abs(vb[3:]).max() > 0.8,
       f'手臂 |max| {np.abs(vb[3:]).max():.6f}（v_in {np.abs(v_in[3:]).max():.6f}）')
n = len(K.dof_names)
Ax_b, bx_b = _box_rows(cfg_of(True), n, cap_b, v_prev, DT)
expect('B3 速度／加速度框仍然存在', len(Ax_b) == 18, f'{len(Ax_b)} 列')
expect('B4 手臂速度框仍是低速框 0.9999',
       abs(max(bx_b[6:12]) - 0.9999) < 1e-6 or
       abs(cfg_of(True).vmax[3] - 0.9999) < 1e-9,
       f'vmax_arm {cfg_of(True).vmax[3]:.6f}')

print('C 過期資料：宣告自由空間也不得放行')
_, _, cap_c, _ = _rows_from_points(K, q9, pts_of(STATUS_STALE), cfg_of(True), v_in)
expect('C1 STALE 的上限仍生效', np.isfinite(cap_c),
       f'cap={cap_c}（stale_speed_cap）')
vc = np.asarray(filter_velocity(K, q9, v_in, pts_of(STATUS_STALE),
                                cfg=cfg_of(True), v_prev=v_prev,
                                a_prev=a_prev, dt=DT).v)
expect('C2 手臂仍被抑制', np.abs(vc[3:]).max() < 0.5,
       f'手臂 |max| {np.abs(vc[3:]).max():.6f}')
mix = pts_of(STATUS_NODATA)
mix[0] = DetectionPoint(frame=LINKS[0], p=np.zeros(3), n=np.array([1., 0, 0]),
                        d=0.0, status=int(STATUS_STALE), age=1.0,
                        occluded=False, offset=np.zeros(3), rho=0.015)
_, _, cap_m, _ = _rows_from_points(K, q9, mix, cfg_of(True), v_in)
expect('C3 只要有一列 STALE，上限就生效', np.isfinite(cap_m), f'cap={cap_m}')

print('D 上下游輸入一致')
_, _, cap_up, _ = _rows_from_points(K, q9, [], cfg_of(True), v_in)
expect('D1 求解端（空點集合）與下游（NODATA＋已確認）cap 相同',
       (not np.isfinite(cap_up)) and (not np.isfinite(cap_b)),
       f'求解端 {cap_up}、下游 {cap_b}')
Ax_up, bx_up = _box_rows(cfg_of(True), n, cap_up, v_prev, DT)
expect('D2 兩者的速度／加速度框逐列相同',
       np.allclose(np.array(Ax_up), np.array(Ax_b))
       and np.allclose(np.array(bx_up), np.array(bx_b)))
vu = np.asarray(filter_velocity(K, q9, v_in, [], cfg=cfg_of(True),
                                v_prev=v_prev, a_prev=a_prev, dt=DT).v)
expect('D3 兩者濾波輸出相同', np.abs(vu - vb).max() < 1e-12,
       f'最大差 {np.abs(vu - vb).max():.2e}')

print('E 預設行為不變')
_, _, cap_e, _ = _rows_from_points(K, q9, pts_of(STATUS_NODATA),
                                   SafetyConfig(dt=DT), v_in)
expect('E1 未帶旗標的 SafetyConfig 仍套 0.05', abs(cap_e - 0.05) < 1e-12,
       f'cap={cap_e}')

print(f"\n{'全部通過' if not fails else '**未通過：' + ', '.join(fails) + '**'}")
sys.exit(1 if fails else 0)
