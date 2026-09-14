"""完整鏈路的**離線閉迴路**核對：求解器 → 安全濾波器 → adapter → E2。

不開模擬器、不啟動 ROS。**重用既有函式**，不另寫近似濾波器：

| 段 | 實際函式 |
|---|---|
| 求解器 | `wholebody_pregrasp.Node.solve`（未綁定呼叫，`--solver dls`） |
| 安全濾波器 | `ammr_wholebody_mpc.wholebody_safety_filter.filter_velocity` |
| adapter | report frame → 本體座標的旋轉（與 `arm_vel_adapter` 相同的式子） |
| 執行端 | `wb_cmd_chain_e2.CmdChainE2` ＋ E2 的 `low_speed_bound` |
| guard | `wholebody_pregrasp.Node.guard` 的關節限位判準（lo+0.05 / hi−0.05） |

**後續位形依最後可執行命令更新**，不是繼續積分原始 DLS 輸出。

安全層的速度框預設是 `[0.2775, 0.2775, 1.1327]`（線速度 m/s、角速度 rad/s），
**不是** E2 的 `lin ≤ 0.05`／`ang ≤ 0.2`。兩者是不同層的不同界限。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'install', 'ammr_wholebody_mpc',
                                'lib', 'python3.12', 'site-packages'))

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa
from ammr_wholebody_mpc.wholebody_safety_filter import (             # noqa: E402
    STATUS_NODATA, DetectionPoint, SafetyConfig, filter_velocity)
from wb_cmd_chain_e2 import CmdChainE2                               # noqa: E402
from wb_qp_lowspeed import (SceneFacts, check_e2_bounds,             # noqa: E402
                            constraints_lowspeed, lowspeed_cfg)
from wb_wheel_limit import WheelLimitConfig                          # noqa: E402

ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]


def load_solver_module():
    """載入 wholebody_pregrasp。該檔有 `if __name__ == '__main__'` 保護，
    直接匯入不會執行 main()；rclpy 只是被 import，不會 init。"""
    path = os.path.join(HERE, 'wholebody_pregrasp.py')
    spec = importlib.util.spec_from_file_location('wbp_offline', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['wbp_offline'] = mod
    spec.loader.exec_module(mod)
    return mod


class Shim:
    """帶著 solve() 需要的狀態，讓我們能呼叫**真正的** solve。"""

    def __init__(self, K, a, q_arm, base, q_pref):
        self.K, self.a = K, a
        self.n = len(K.dof_names)
        self.idx = [K.dof_names.index(j) for j in ARM_JOINTS]
        self.q_arm = np.asarray(q_arm, float)
        self.base = np.asarray(base, float)
        self.q_pref = np.asarray(q_pref, float)
        # **自由空間下距離節點實際產生的列**：每個連桿一列（不是零列），
        # 狀態 STATUS_NODATA(3.0)、位置全零。與「該連桿缺 TF」編碼相同 ——
        # 因此空場景要靠設定確認，不靠列推論。
        self.rows = None               # 由 main 指派
        self.q_arm_t = self.base_t = None      # 由 main 以 time.monotonic() 設定
        self.min_d = float('inf')   # 自由空間：無障礙物，模型間距不受限
        self.base0 = np.asarray(base, float).copy()   # 由 main 覆寫為起始位姿
        self.n_qp_fail = 0
        self.qp_iter_max = 4000        # 與節點預設一致
        self.qp_last_status = ''
        self.qp_status = {}
        self.n_bar_rows = 0
        self.cfg = None                # 由 main 指派為 SafetyConfig
        self.link_names = []           # 由 main 以 arm_link_names 指派


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', nargs=3, type=float, default=[0.30, 0.0, 0.55])
    ap.add_argument('--urdf', default=os.path.join(
        HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--physics-dt', type=float, default=0.01)
    ap.add_argument('--timeout-s', type=float, default=60.0)
    ap.add_argument('--tol-p', type=float, default=0.005)
    ap.add_argument('--tol-r', type=float, default=0.02)
    ap.add_argument('--settle-s', type=float, default=2.0)
    ap.add_argument('--base-lin-max', type=float, default=0.05)
    ap.add_argument('--base-ang-max', type=float, default=0.2)
    # **另立配置**用：把安全層的底盤速度框收到能滿足 E2 的範數界限。
    # 速度框是**逐軸**的 |vx|,|vy| ≤ b，範數上界是 b·√2，
    # 所以要滿足 hypot ≤ 0.05 必須設 b ≤ 0.05/√2 = 0.035355。
    ap.add_argument('--base-vmax', type=float, default=None,
                    help='安全層底盤逐軸速度框（m/s）；不給則用預設 0.2775')
    ap.add_argument('--base-wmax', type=float, default=None,
                    help='安全層底盤角速度框（rad/s）；不給則用預設 1.1327')
    ap.add_argument('--arm-vmax', type=float, default=None,
                    help='安全層手臂速度框（rad/s）；預設 LITE6_SAFE 的 3.14，'
                         '比 E2 的 arm_rate_max 1.0 寬')
    ap.add_argument('--lowspeed-qp', action='store_true',
                    help='另立配置：把 E2 執行界限放進 QP 約束集'
                         '（wb_qp_lowspeed）；**不冒稱 B 基線**')
    ap.add_argument('--solver', default='dls', choices=['dls', 'qp'])
    ap.add_argument('--label', default='')
    ap.add_argument('--out', required=True)
    cl = ap.parse_args()

    M = load_solver_module()
    # **重用真正的方法**，不自己重寫：q9/tool/guard 都直接綁到 shim 上
    Shim.q9 = M.WholeBody.q9
    Shim.tool = M.WholeBody.tool
    Shim.guard = M.WholeBody.guard
    Shim._constraints = M.WholeBody._constraints      # QP 路徑用
    Shim._solve_qp = M.WholeBody._solve_qp
    K = WholeBodyKinematics.from_urdf_file(cl.urdf)
    from ammr_wholebody_mpc.arm_link_geometry import arm_link_names
    link_names = arm_link_names(open(cl.urdf).read())
    # 自由空間：每連桿一列、STATUS_NODATA、位置全零（見 arm_link_distance._rows_links）
    STATUS_NODATA = 3.0
    free_rows = np.array([[0.0] * 6 + [0.0, STATUS_NODATA, -1.0, 1.0,
                                       float(li), 0.0, 0.0, 0.0, 0.015]
                          for li in range(len(link_names))], dtype=float)
    facts = SceneFacts(obstacles_configured=0, tf_ok_links=len(link_names),
                       n_links=len(link_names), rows_total=len(free_rows),
                       rows_status_ok=0)
    # **與線上相同的濾波器輸入**：距離節點在自由空間下每個連桿一列 NODATA。
    # 先前離線餵空點集合，與線上不等價（空集合 cap=inf、NODATA cap=0.05），
    # 那正是 wb_solver_iso_094526 離線／線上結果相反的原因。
    pts_online = [DetectionPoint(frame=n, p=np.zeros(3),
                                 n=np.array([1.0, 0.0, 0.0]), d=0.0,
                                 status=int(STATUS_NODATA), age=0.0,
                                 occluded=False, offset=np.zeros(3), rho=0.015)
                  for n in link_names]

    # 求解器參數：**沿用既有預設**，逐項寫出以便查核
    a = types.SimpleNamespace(
        tcp='link_tcp', target=cl.target, kp_p=1.0, kp_r=1.0,
        v_task=0.10, w_task=0.5, base_weight=3.0, limit_margin=0.35,
        k_limit=1.5, solver=cl.solver, mu_post=1.0, kp_post=1.2, damping=0.06,
        tol_p=cl.tol_p, tol_r=cl.tol_r, settle_s=cl.settle_s,
        timeout_s=cl.timeout_s, max_data_age=0.50,
        abort_d=0.02, max_base_travel=1.2)     # guard 的另外兩項門檻
    q_pref = np.array([0.0, 0.3641, 0.5263, 0.0, -1.4086, 0.0])   # pregrasp_reference

    zc = np.array([1.0, 0.0, 0.0])
    xc = np.array([0.0, 0.0, 1.0])
    xc = xc - float(xc @ zc) * zc
    xc /= np.linalg.norm(xc)
    T_des = np.eye(4)
    T_des[:3, :3] = np.column_stack([xc, np.cross(zc, xc), zc])
    T_des[:3, 3] = np.array(cl.target, float)

    q_arm = np.zeros(6)          # test_start
    base = np.zeros(3)           # (x, y, yaw) in odom
    LO, HI = np.array(LITE6_SAFE.lower), np.array(LITE6_SAFE.upper)
    GUARD = 0.05                 # 完整節點 guard 的距限位餘裕

    scfg = SafetyConfig(alpha=2.0, d0=0.05, tau=0.15, a_brake=1.0, eps=0.03,
                        dt=1.0 / cl.rate, fix_base=False)
    if cl.lowspeed_qp:
        scfg = lowspeed_cfg(scfg)
        # **狀態分類**：空場景已由 facts 確認，NODATA 讀成「範圍內沒有東西」。
        # 未確認時這裡維持 False，退化上限照舊生效。
        scfg.freespace_confirmed = bool(facts.empty_scene_confirmed)
        print(f'[cfg] freespace_confirmed={scfg.freespace_confirmed}'
              f'（NODATA 的語意：'
              f'{"範圍內沒有東西" if scfg.freespace_confirmed else "資料未知"}）')
        print(f'[cfg] **低速 QP 配置**：速度框 → |vx|,|vy| ≤ '
              f'{scfg.vmax[0]:.6f}, |wz| ≤ {scfg.vmax[2]}, 手臂 ≤ {scfg.vmax[3]}')
        print(f'[cfg] 空場景確認：障礙物設定 {facts.obstacles_configured} 個、'
              f'TF {facts.tf_ok_links}/{facts.n_links} —— '
              f'{"已確認" if facts.empty_scene_confirmed else facts.why_not()}')
        # **只換約束集的組裝方式**，目標函式與權重沿用既有 solve
        def _c(self, q, v_lin):
            A, b, nb, info = constraints_lowspeed(
                self.K, q, v_lin, [], self.cfg, facts,
                v_prev=self.v_prev, dt=self.cfg.dt)
            self.last_con_info = info
            return A, b, nb
        Shim._constraints = _c
    if (cl.base_vmax is not None or cl.base_wmax is not None
            or cl.arm_vmax is not None):
        vm = scfg.vmax.copy()
        if cl.base_vmax is not None:
            vm[0] = vm[1] = cl.base_vmax
        if cl.base_wmax is not None:
            vm[2] = cl.base_wmax
        if cl.arm_vmax is not None:
            vm[3:] = cl.arm_vmax
        scfg.vmax = vm
        print(f'[cfg] **另立配置**：安全層底盤速度框 → '
              f'|vx|,|vy| ≤ {vm[0]}, |wz| ≤ {vm[2]}, 手臂 ≤ {vm[3]}'
              f'（預設 0.2775 / 1.1327 / 3.1416）')
    # QP 需要距離約束列；自由空間下 rows 為空，仍走同一條 _constraints 路徑
    wcfg = WheelLimitConfig(dt_max=max(5.0 * cl.physics_dt, 0.05))
    chain = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
                       wheel_ok=lambda vx, vy, wz: (
                           (False, f'線速度 {math.hypot(vx, vy):.4f} > {cl.base_lin_max}')
                           if math.hypot(vx, vy) > cl.base_lin_max else
                           (False, f'角速度 {abs(wz):.4f} > {cl.base_ang_max}')
                           if abs(wz) > cl.base_ang_max else (True, None)),
                       joint_lower=tuple(LITE6_SAFE.lower),
                       joint_upper=tuple(LITE6_SAFE.upper),
                       mode='solver_freespace', wheel_cfg=wcfg,
                       keep_limit_rows=100000)

    rows, v_prev, a_prev = [], None, None
    sim_t, seq = 0.0, 0
    period = 1.0 / cl.rate
    n_inner = max(1, int(round(period / cl.physics_dt)))
    stop_reason, reached_since = None, None
    for cyc in range(int(cl.timeout_s * cl.rate) + 1):
        q9 = np.zeros(9)
        q9[:3] = base
        q9[3:] = q_arm
        T = K.fk(q9, a.tcp)
        e_p = float(np.linalg.norm(T_des[:3, 3] - T[:3, 3]))
        e_r = float(np.linalg.norm(M.rot_error(T[:3, :3], T_des[:3, :3])))

        marg_lo = float(np.min(q_arm - LO))
        marg_hi = float(np.min(HI - q_arm))

        # ---- guard：**完整節點既有的 guard**，不是我重寫的判準 ----
        sh = Shim(K, a, q_arm, base, q_pref)
        sh.q_arm_t = sh.base_t = time.monotonic()
        sh.base0 = np.zeros(3)                 # 起始底盤位姿（原點）
        sh.cfg = scfg                          # QP 的約束集用同一份安全設定
        sh.link_names = link_names
        sh.rows = free_rows
        sh.v_prev, sh.a_prev, sh.dt = v_prev, a_prev, period
        why = sh.guard()
        if why:
            j = int(np.argmin(np.minimum(q_arm - LO, HI - q_arm))) + 1
            stop_reason = (f'guard：{why}（最小餘裕 '
                           f'{min(marg_lo, marg_hi):+.5f} rad，joint{j}）')
            break

        # ---- 求解器（真正的 solve）----
        v_raw, _T, _ep, _er = M.WholeBody.solve(sh, T_des)   # **真正的 solve**

        # ---- 安全濾波器（真正的 filter_velocity）----
        sr = filter_velocity(K, q9, v_raw, pts_online, cfg=scfg,
                             v_prev=v_prev, a_prev=a_prev, dt=period)
        v_saf = np.asarray(sr.v, float)

        # ---- adapter：report frame → 本體座標 ----
        cy, sy = math.cos(base[2]), math.sin(base[2])
        vbx = v_saf[0] * cy + v_saf[1] * sy
        vby = -v_saf[0] * sy + v_saf[1] * cy
        v_body = np.concatenate([[vbx, vby, v_saf[2]], v_saf[3:]])

        # ---- E2：接收 + 每個物理步套用 ----
        accepted = chain.receive(list(v_body), sim_t)
        applied = []
        for k in range(n_inner):
            out = chain.step(sim_t, cl.physics_dt, q_arm)
            if chain.fail is not None:
                break
            if out is None:
                sim_t += cl.physics_dt
                continue
            b_body, sp = out
            # **依最後可執行命令更新位形**
            cyk, syk = math.cos(base[2]), math.sin(base[2])
            base = base + np.array([b_body[0] * cyk - b_body[1] * syk,
                                    b_body[0] * syk + b_body[1] * cyk,
                                    b_body[2]]) * cl.physics_dt
            q_arm = np.array(sp, float)
            applied.append(list(b_body))
            sim_t += cl.physics_dt
        rows.append({
            'cyc': cyc, 'sim_t': round(sim_t, 4),
            'e_p': round(e_p, 6), 'e_r': round(e_r, 6),
            'solver_raw': [round(float(x), 6) for x in v_raw],
            'safety_out': [round(float(x), 6) for x in v_saf],
            'adapter_body': [round(float(x), 6) for x in v_body],
            'e2_accepted': bool(accepted),
            'e2_reject': chain.summary().get('last_reject'),
            'e2_fail': chain.fail,
            'lam': (chain.last_limit or {}).get('lam'),
            'n_applied_steps': len(applied),
            'base': [round(float(x), 6) for x in base],
            'joint_margin_lo': round(marg_lo, 6),
            'joint_margin_hi': round(marg_hi, 6),
            'solver_lin': round(float(math.hypot(v_raw[0], v_raw[1])), 6),
            'safety_lin': round(float(math.hypot(v_saf[0], v_saf[1])), 6),
            'body_lin': round(float(math.hypot(vbx, vby)), 6),
            'e2_bounds_ok': check_e2_bounds(v_body)[0],
            'e2_bounds_why': check_e2_bounds(v_body)[1],
            'con': getattr(sh, 'last_con_info', None),
        })
        # a_prev 必須用**前兩筆**輸出；先前寫成先更新 v_prev 再算，恆為 0
        a_prev = ((v_saf - v_prev) / period) if v_prev is not None else None
        v_prev = v_saf.copy()
        if chain.fail is not None:
            stop_reason = f'E2 整體失效：{chain.fail}'
            break
        if e_p <= cl.tol_p and e_r <= cl.tol_r:
            reached_since = reached_since if reached_since is not None else sim_t
            if sim_t - reached_since >= cl.settle_s:
                stop_reason = '到達並保持'
                break
        else:
            reached_since = None
    else:
        stop_reason = f'任務上限 {cl.timeout_s}s'

    sl = [r['solver_lin'] for r in rows]
    sf = [r['safety_lin'] for r in rows]
    bl = [r['body_lin'] for r in rows]
    ang = [abs(r['adapter_body'][2]) for r in rows]
    n_rej = sum(1 for r in rows if not r['e2_accepted'])
    out = {
        'schema': 'wb_solver_offline_chain/1',
        'reused_functions': {
            'solver': 'wholebody_pregrasp.Node.solve（--solver dls）',
            'safety': 'wholebody_safety_filter.filter_velocity',
            'adapter': 'report frame → 本體座標旋轉',
            'endpoint': 'wb_cmd_chain_e2.CmdChainE2 ＋ E2 low_speed_bound',
            'guard': 'wholebody_pregrasp 的關節限位判準 lo+0.05 / hi−0.05'},
        'note': ('後續位形依**最後可執行命令**更新，非積分原始 DLS 輸出。'
                 '安全層速度框 [0.2775, 0.2775, 1.1327]，'
                 '與 E2 的 lin 0.05／ang 0.2 是不同層的不同界限。'),
        'label': cl.label, 'solver': cl.solver,
        'safety_vmax_base': [float(scfg.vmax[0]), float(scfg.vmax[1]),
                             float(scfg.vmax[2])],
        'target': cl.target, 'cycles': len(rows), 'sim_t': round(sim_t, 3),
        'stop_reason': stop_reason,
        'final_e_p': rows[-1]['e_p'] if rows else None,
        'final_e_r': rows[-1]['e_r'] if rows else None,
        'peak': {'solver_lin': max(sl) if sl else None,
                 'safety_lin': max(sf) if sf else None,
                 'body_lin': max(bl) if bl else None,
                 'body_ang': max(ang) if ang else None},
        'e2_bounds': {'lin': cl.base_lin_max, 'ang': cl.base_ang_max},
        'e2_rejected_cycles': n_rej,
        'e2_fail': chain.fail,
        'joint_margin_min': min((r['joint_margin_lo'] for r in rows),
                                default=None),
        'wheel_limit': chain.summary()['wheel_limit'],
        'rows': rows}
    os.makedirs(os.path.dirname(cl.out) or '.', exist_ok=True)
    json.dump(out, open(cl.out, 'w'), ensure_ascii=False, indent=1)

    print(f'週期 {len(rows)}，模擬時間 {sim_t:.2f}s，結束原因：{stop_reason}')
    if rows:
        print(f"  末誤差 位置 {rows[-1]['e_p']*1000:.2f} mm、"
              f"姿態 {rows[-1]['e_r']:.4f} rad")
        print(f'  線速度峰值：求解器 {max(sl):.4f} → 安全層 {max(sf):.4f} → '
              f'本體 {max(bl):.4f}（E2 界限 {cl.base_lin_max}）')
        print(f'  角速度峰值（本體）{max(ang):.4f}（E2 界限 {cl.base_ang_max}）')
        print(f'  E2 拒收週期 {n_rej} / {len(rows)}；整體失效 {chain.fail}')
        print(f"  關節距下限最小餘裕 {out['joint_margin_min']:+.5f} rad"
              f'（guard 門檻 {GUARD}）')
        wl = out['wheel_limit']
        print(f"  輪級限制：modified {wl['modified_steps']} 步、"
              f"逾時減速 {wl['timeout_decel_steps']} 步")
    print(f'  -> {cl.out}')
    return 0


sys.exit(main())
