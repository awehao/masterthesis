"""低速框下的實際鏈路核對：**QP → 安全濾波 → adapter 座標轉換 → E2**。

用 main4 的 APPROACH 狀態（底盤 park、手臂≈零、開度 0），確認低速框
（L1：逐軸 0.035255 / 0.199900 / 0.999900）下：

  1. QP 的**約束集合裡真的有**低速框的列（不是對輸出裁切）
  2. QP 求解狀態與輸出
  3. 安全濾波輸出
  4. adapter 純旋轉後的平面範數
  5. **E2 是否接受並產生可套用命令**（`receive` ＋ `step`，`fail` 保持 None）

**呼叫的是實際程式碼**：`WholeBody.solve` / `_solve_qp`、`PullSolver._constraints`、
`filter_velocity`、`arm_vel_adapter.world_to_body9`、`CmdChainE2` 都是實跑用的
同一批函式物件，只是以替身物件承載屬性以避開 ROS 初始化。

**距離列**是用距離節點的同一組幾何函式在該狀態重新產生的，
**不是** main4 實際發布的雲（main4 沒有封存雲列）。

本核對只證明**已測狀態**的相容性，**不證明整段任務必然成功**。
規格：evaluation/results/specs/coman_low_speed_box_l1.yaml
"""
from __future__ import annotations
import argparse
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                   # noqa
from ammr_wholebody_mpc.arm_link_distance import (                     # noqa
    STATUS_OK as D_OK, VOBS_STATIC, expand_pair_rows, forced_pair_rows,
    parse_obstacles)
from ammr_wholebody_mpc.arm_link_geometry import (                     # noqa
    arm_link_names, obstacle_distance_matrix, sample_links_certified)
from ammr_wholebody_mpc.arm_pregrasp import ARM_JOINTS                 # noqa
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa
from ammr_wholebody_mpc.wholebody_safety_filter import (               # noqa
    SafetyConfig, detection_point_from_row, filter_velocity)
from wb_cmd_chain_e2 import CmdChainE2                                 # noqa
from wb_wheel_limit import WheelLimitConfig                            # noqa
from coman_low_speed_box import (e2_compat_violations, tighten_vmax,   # noqa
                                 worst_plane_norm)

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
PAIR_ROWS = ['link4:*', 'link5:*', 'link6:*', 'uflite_finger1:*',
             'uflite_finger2:*', 'uflite_gripper_link:*']
EXEMPT = ['uflite_finger1:handle_bar', 'uflite_finger2:handle_bar']
PAIR_GAP = ('uflite_gripper_link:handle_bar:0.003,'
            'uflite_finger1:drawer_front_panel:0.010,'
            'uflite_finger2:drawer_front_panel:0.010,link6:handle_bar:0.045,'
            'uflite_finger1:handle_post_l:0.045,uflite_finger1:handle_post_r:0.045,'
            'uflite_finger2:handle_post_l:0.045,uflite_finger2:handle_post_r:0.045,'
            'link5:handle_bar:0.065,link4:handle_bar:0.070,'
            'uflite_gripper_link:drawer_front_panel:0.030,'
            'uflite_gripper_link:handle_post_l:0.035,'
            'uflite_gripper_link:handle_post_r:0.035,'
            'link6:handle_post_l:0.060,link6:handle_post_r:0.060')

# L1 低速框（求解端＋安全層，**逐軸**）
L1_LIN, L1_ANG, L1_ARM = 0.035255, 0.199900, 0.999900
# E2 的整筆拒收門檻（執行端 argparse 預設，**不動**）
E2_LIN, E2_ANG, E2_ARM = 0.05, 0.2, 1.0
# main4 的 APPROACH 狀態
BASE = np.array([10.5, 8.0897, math.pi / 2])
Q_ARM = np.array([-0.000288, 0.000726, -0.000102, 0.000158, 0.0, -0.0])
UNIT_X, DRAWER_Y0, OPENING = 10.5, 9.0, 0.0
# main4 封存的把手世界位置（sim 11.94）與設計抓取偏移
HANDLE_P = np.array([10.5, 8.715, 0.55])
TCP_OFF_Z = 0.0147


def load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def adapter_rotation():
    """取 adapter 的 world_to_body9，**不載入其 ROS 依賴**。"""
    src = open(os.path.join(HERE, 'arm_vel_adapter.py'), encoding='utf-8').read()
    i, j = src.index('def world_to_body9'), src.index('class ')
    ns = {'math': math}
    exec(src[i:j], ns)
    return ns['world_to_body9']


def build_rows(K, q, obs, names, S, links):
    """用距離節點的同一組幾何函式產生 22 欄距離列。"""
    pairs, _ = expand_pair_rows(PAIR_ROWS, EXEMPT, names)
    rows, seen = [], set()
    for li, nm in enumerate(links):
        Tw = K.fk(q, nm)
        Sx = S[nm]
        W = (Sx.points @ Tw[:3, :3].T) + Tw[:3, 3]
        D, V = obstacle_distance_matrix(W, obs)
        jm = np.argmin(D, axis=1)
        ixr = np.arange(len(W))
        dd, vv = D[ixr, jm], V[ixr, jm]
        dmin = float(dd.min())
        sel = np.nonzero(dd <= dmin + Sx.rho)[0]
        sel = sel[np.argsort(dd[sel])]
        for k in sel:
            nh = vv[k] / max(abs(float(dd[k])), 1e-9)
            rows.append(list(W[k]) + list(nh)
                        + [float(dd[k]), float(D_OK), 0.0, 0.0, float(li)]
                        + list(Sx.points[k])
                        + [float(Sx.rho), float(names.index(names[jm[k]])),
                           0.0, 0.0, 0.0, float(VOBS_STATIC), 0.0, 0.0])
            seen.add((li, int(jm[k]), *(round(float(x), 12) for x in Sx.points[k])))
        for obn in pairs.get(nm, []):
            oi = names.index(obn)
            ex = forced_pair_rows(W, Sx.points, obs[oi], oi, li, Sx.rho, D_OK,
                                  0.0, lambda p, v: 0.0, 3.0, 0,
                                  vobs_batch=None, d_col=D[:, oi], v_col=V[:, oi])
            for r in ex:
                key = (int(r[10]), int(r[15]),
                       *(round(float(r[c]), 12) for c in (11, 12, 13)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
    return np.asarray(rows, float)


def make_solver_stand_in(K, cfg, rows, links, names, a):
    """承載屬性的替身；方法一律取**實跑用的同一批函式物件**。"""
    M = load_module('wbp_lowspeed', os.path.join(HERE, 'wholebody_pregrasp.py'))
    pull = load_module('cpsn_lowspeed',
                       os.path.join(HERE, 'coman_pull_solver_node.py'))
    cl = argparse.Namespace(contact_pairs='', pair_d0='', pair_gap=PAIR_GAP,
                            stroke_m=0.020, pull_duration_s=4.0, retreat_m=0.040,
                            tcp_offset_z=0.0147, slide_axis='[0.0, -1.0, 0.0]',
                            rows_wait_s=15.0, retreat_clear_m=0.0233,
                            grasp_rot='[[-1,0,0],[0,0,1],[0,1,0]]', out='/dev/null')
    PullSolver = pull.build(M, cl)

    class Stand:
        pass
    Stand.solve = M.WholeBody.solve                 # 實際的 solve
    Stand._solve_qp = M.WholeBody._solve_qp         # 實際的 OSQP 設定
    Stand.q9 = M.WholeBody.q9
    Stand.tool = M.WholeBody.tool
    Stand._constraints = PullSolver._constraints    # 實跑用的覆寫版本
    o = Stand()
    o.K, o.cfg, o.rows, o.link_names, o.obs_names, o.a = K, cfg, rows, links, names, a
    o.n = len(K.dof_names)
    o.idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    o.base, o.q_arm = BASE.copy(), Q_ARM.copy()
    o.q_pref = np.array(a.posture, float)
    o.v_prev = np.zeros(o.n)
    o.n_qp_fail, o.qp_iter_max, o.qp_last_status = 0, 0, ''
    o.qp_status, o.t_qp = {}, 0.0
    return o, M


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:58s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    # ---------- 狀態、障礙物、距離列 ----------
    K = WholeBodyKinematics.from_urdf_file(URDF)
    xml = open(URDF, encoding='utf-8').read()
    n = len(K.dof_names)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    q = np.zeros(n)
    q[:3] = BASE
    q[idx] = Q_ARM
    sp = subprocess.run([sys.executable, os.path.join(HERE,
                         'coman_obstacle_specs.py'), str(UNIT_X),
                         str(DRAWER_Y0)],
                        capture_output=True, text=True, check=True)
    obs = parse_obstacles([x for x in sp.stdout.split() if x.strip()])
    names = [o.name for o in obs]
    # 掛在 model 上的抽屜部件，其 T_world_link 由執行端發布位姿；離線要自己給。
    # 抽屜本體世界 y = drawer_y0 - opening（開度為 0 ⇒ 關閉位置）。
    T_db = np.eye(4)
    T_db[0, 3], T_db[1, 3] = UNIT_X, DRAWER_Y0 - OPENING
    for _o in obs:
        if _o.model:
            _o.T_world_link = T_db.copy()
    S = sample_links_certified(xml, rho_target=0.015, tol=0.001)
    links = arm_link_names(xml)
    rows = build_rows(K, q, obs, names, S, links)
    print(f'狀態：底盤 ({BASE[0]:.4f}, {BASE[1]:.4f}) yaw {math.degrees(BASE[2]):.1f}°、'
          f'手臂≈零、開度 {OPENING*1e3:.1f} mm；距離列 {len(rows)}')
    print()

    # ---------- 低速框套進 cfg（與求解端／安全層**同一段算術**）----------
    print('A 低速框進入約束組裝（不是對輸出裁切）')
    cfg = SafetyConfig(dt=1.0 / 20.0)
    cfg.pair_gap = dict(
        (k.rsplit(':', 1)[0].replace(':', '|'), float(k.rsplit(':', 1)[1]))
        for k in PAIR_GAP.split(','))
    hw = np.asarray(cfg.vmax, float).copy()
    # 用**共用的** tighten_vmax（求解端啟動時呼叫的同一個函式）
    vm, ov = tighten_vmax(cfg.vmax, L1_LIN, L1_ANG, L1_ARM, np)
    cfg.vmax = vm
    check('三項覆寫都生效', sorted(ov) == ['vmax_arm', 'vmax_base_ang',
                                     'vmax_base_lin'], f'  {sorted(ov)}')
    check('硬體框確實被收緊（只縮不放）',
          bool((vm <= hw + 1e-12).all() and vm[0] < hw[0]),
          f'  {hw[0]:.4f}→{vm[0]:.6f}')
    check('逐軸框兩軸同時到頂仍在 E2 範數門檻內',
          worst_plane_norm(vm[0]) <= E2_LIN,
          f'  √2×{vm[0]:.6f}={worst_plane_norm(vm[0]):.6f} <= {E2_LIN}')
    check('共用守門判 L1 為相容',
          not e2_compat_violations(vm[0], vm[2], max(vm[3:]),
                                   E2_LIN, E2_ANG, E2_ARM))

    a = argparse.Namespace(
        tcp='link_tcp', v_task=0.10, w_task=0.5, kp_p=1.0, kp_r=1.0,
        base_weight=3.0, limit_margin=0.35, k_limit=1.5, mu_post=1.0,
        kp_post=1.2, damping=0.06, solver='qp', rate=20.0,
        posture=list(np.zeros(6)))
    o, M = make_solver_stand_in(K, cfg, rows, links, names, a)

    # 約束集合裡要**找得到**低速框的列：|e_i| = 1 的單位列，界為 L1 值
    A, b, nb = o._constraints(q, np.zeros(n))
    # `_constraints` 依序組 屏障(nb) + 關節限位 + 速度／加速度框，
    # 而 `_box_rows` 每個自由度發**兩列** ⇒ 框列就是最後 2n 列。
    # （先前用 A[nb:] 會連關節限位列一起取到，那些界是位置餘量／dt，數量級不同。）
    box, bb = A[-2 * n:], b[-2 * n:]
    unit = [i for i in range(len(box))
            if abs(np.abs(box[i]).sum() - 1.0) < 1e-12
            and (np.abs(box[i]) > 0.5).sum() == 1]
    check('框列全為單位列且共 2n 列', len(unit) == 2 * n, f'  {len(unit)}/{2*n}')
    found_lin = [bb[i] for i in unit
                 if int(np.argmax(np.abs(box[i]))) in (0, 1)]
    found_arm = [bb[i] for i in unit
                 if int(np.argmax(np.abs(box[i]))) in idx]
    check('約束集合含底盤平移的速度框列', len(found_lin) == 4,
          f'  {len(found_lin)} 列')
    check('底盤框列的界 == L1 逐軸值',
          bool(found_lin) and max(abs(abs(x) - L1_LIN) for x in found_lin) < 1e-9,
          f'  max |b| = {max(abs(x) for x in found_lin):.6f}')
    check('手臂框列的界 <= L1 關節值',
          bool(found_arm) and max(abs(x) for x in found_arm) <= L1_ARM + 1e-9,
          f'  max |b| = {max(abs(x) for x in found_arm):.6f}')
    check('屏障列仍在（避碰未被關掉）', nb > 0, f'  {nb} 列')
    print()

    # ---------- QP ----------
    print('B QP 求解（低速框在約束裡）')
    # **main4 的實際接近目標**，用封存的把手世界位姿依 approach_target 反推：
    #   T_des.p = p_handle - R_des @ grasp_offset,  grasp_offset = (0, 0, -tcp_offset_z)
    R_des = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    T_des = np.eye(4)
    T_des[:3, :3] = R_des
    T_des[:3, 3] = HANDLE_P - R_des @ np.array([0.0, 0.0, -TCP_OFF_Z])
    try:
        v_qp, _T, ep, er = o.solve(T_des)
        ok_qp, why = True, ''
    except RuntimeError as exc:
        v_qp, ok_qp, why = None, False, str(exc)
    check('QP 可行（已測狀態）', ok_qp, f'  {why}')
    if ok_qp:
        check('接近誤差與 main4 相符（±5 mm）', abs(ep - 0.4679) < 0.005,
              f'  ep {ep:.4f} vs main4 0.4679')
    if not ok_qp:
        print('\n**QP 不可行** —— 依裁定回報狀態與衝突證據，不關屏障、不放寬界限、'
              '不退回無約束命令。')
        print(f'  OSQP 狀態計數 {o.qp_status}、最後狀態 {o.qp_last_status!r}、'
              f'迭代上限 {o.qp_iter_max}')
        return 1
    lin_w = math.hypot(v_qp[0], v_qp[1])
    print(f'  OSQP 狀態 {o.qp_status}、迭代上限 {o.qp_iter_max}、'
          f'ep {ep:.4f} m、er {er:.4f} rad')
    print(f'  v_qp 底盤 ({v_qp[0]:+.6f}, {v_qp[1]:+.6f}, {v_qp[2]:+.6f})  '
          f'平面範數 {lin_w:.6f}')
    print(f'  v_qp 手臂 {np.round(v_qp[idx], 6).tolist()}')
    check('QP 輸出逐軸在低速框內',
          bool((np.abs(v_qp[:2]) <= L1_LIN + 1e-6).all()
               and abs(v_qp[2]) <= L1_ANG + 1e-6
               and (np.abs(v_qp[idx]) <= L1_ARM + 1e-6).all()))
    check('QP 輸出平面範數 < E2 門檻', lin_w < E2_LIN,
          f'  {lin_w:.6f} < {E2_LIN}')
    print()

    # ---------- 安全濾波（實際 filter_velocity）----------
    print('C 安全濾波（同一份低速框）')
    pts = [p for p in (detection_point_from_row(r, links, names) for r in rows
                       if r[7] == D_OK) if p is not None]
    res = filter_velocity(K, q, v_qp, pts, cfg, v_prev=np.zeros(n), dt=cfg.dt)
    v_sf = np.asarray(res.v, float)
    lin_sf = math.hypot(v_sf[0], v_sf[1])
    print(f'  列 {res.n_rows}、活躍 {res.n_active}、fallback {res.fallback}、'
          f'unresolved {res.unresolved}')
    print(f'  v_sf 底盤 ({v_sf[0]:+.6f}, {v_sf[1]:+.6f}, {v_sf[2]:+.6f})  '
          f'平面範數 {lin_sf:.6f}')
    check('濾波未 fallback、未 unresolved',
          (not res.fallback) and (not res.unresolved))
    check('濾波輸出逐軸在低速框內',
          bool((np.abs(v_sf[:2]) <= L1_LIN + 1e-6).all()
               and abs(v_sf[2]) <= L1_ANG + 1e-6
               and (np.abs(v_sf[idx]) <= L1_ARM + 1e-6).all()))
    check('濾波輸出平面範數 < E2 門檻', lin_sf < E2_LIN,
          f'  {lin_sf:.6f} < {E2_LIN}')
    print()

    # ---------- adapter ＋ E2 ----------
    print('D adapter 純旋轉 → E2 新鮮命令執行')
    w2b = adapter_rotation()
    v9 = [float(v_sf[0]), float(v_sf[1]), float(v_sf[2])] + \
         [float(x) for x in v_sf[idx]]
    body = w2b(v9, float(BASE[2]))
    check('旋轉保範數', abs(math.hypot(body[0], body[1]) - lin_sf) < 1e-12,
          f'  {math.hypot(body[0], body[1]):.9f}')

    def e2_try(cmd, tag, expect_ok):
        ch = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=E2_ARM,
                        wheel_ok=lambda vx, vy, wz: (
                            (False, f'線速度 {math.hypot(vx,vy):.4f} > {E2_LIN} m/s')
                            if math.hypot(vx, vy) > E2_LIN else
                            (False, f'角速度 {abs(wz):.4f} > {E2_ANG} rad/s')
                            if abs(wz) > E2_ANG else (True, None)),
                        joint_lower=tuple(LITE6_SAFE.lower),
                        joint_upper=tuple(LITE6_SAFE.upper),
                        mode='sync',
                        wheel_cfg=WheelLimitConfig(arm_rate_max=E2_ARM,
                                                   dt_max=0.05))
        got = ch.receive(list(cmd), 1.0)
        out = ch.step(1.0, 0.01, list(Q_ARM)) if got else None
        good = bool(got) and out is not None and ch.fail is None
        check(f'{tag}', good == expect_ok,
              f'  receive={got} step={"有" if out else "None"} fail={ch.fail}')
        return out, ch

    out, ch = e2_try(body, 'E2 接受並產生可套用命令（濾波輸出）', True)
    if out is not None:
        print(f'  E2 套用：底盤本體 {tuple(round(x,6) for x in out[0])}、'
              f'手臂設定點 {tuple(round(x,6) for x in out[1])}')
    print()

    # ---------- 兩軸同時接近上限的反例 ----------
    print('E 兩軸同時接近上限（不能只測單軸）')
    both = [L1_LIN, L1_LIN, L1_ANG] + [L1_ARM] * 6
    print(f'  逐軸到頂：平面範數 {math.hypot(both[0], both[1]):.6f}')
    e2_try(w2b(both, float(BASE[2])), 'E2 接受兩軸同時到頂的命令', True)
    e2_try(w2b(both, 0.7854), 'E2 接受同一命令旋轉 45° 後（範數不變）', True)
    # 負向對照：門檻本身必須仍會擋
    over = [E2_LIN / math.sqrt(2) + 1e-4, E2_LIN / math.sqrt(2) + 1e-4, 0.0] \
        + [0.0] * 6
    print(f'  略超：平面範數 {math.hypot(over[0], over[1]):.6f} > {E2_LIN}')
    e2_try(over, 'E2 仍擋下略超範數的命令（門檻未被繞過）', False)
    e2_try([0.0, 0.0, 0.0] + [E2_ARM + 1e-3] + [0.0] * 5,
           'E2 仍擋下略超關節速率的命令', False)
    # main4 的實際命令：回歸證據
    m4 = [-0.0000, -0.2775, 0.0185, 0.0396, 1.1376, 0.9558,
          -0.0703, -1.0051, -0.0701]
    print(f'  main4 實際命令：平面範數 {math.hypot(m4[0], m4[1]):.4f}、'
          f'手臂最大 {max(abs(x) for x in m4[3:]):.4f}')
    e2_try(w2b(m4, float(BASE[2])), 'E2 仍會擋下 main4 的越界命令', False)

    print()
    print('低速框鏈路核對：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    print('本核對只證明**已測狀態**的相容性，不證明整段任務必然成功。')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
