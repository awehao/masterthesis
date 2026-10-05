#!/usr/bin/env python3
"""DL3 任務節點的 vision 模式掛鉤（純函式，可離線測試；規格 results/vision/DL3_vision_pregrasp_spec.md draft-2）。

* 鎖定前：**不產生、不發布任何目標**（draft-2 審查必修 1：求解端 --require-topic-target ⇒ 第一輪必用鎖定後目標）。
* ALIGN 進入時：dl3_latch.latch() → GC2（估計物體）→ 固定底盤 IK 前檢；全部通過才可啟動求解。
* 鎖定後：ALIGN 目標、斜坡終點、到位誤差（含 T_ref）全部由**同一個**鎖定把手位姿產生。

能力邊界（照規格寫）：
* GC2 對「由估計把手反推的估計物體」核對：工具、把手、櫃體一起剛體變換時局部距離不變 ⇒ 通過只表示指定抓取轉換與
  估計物體模型在目標處相容，**不是觀測準確性閘門，也不驗證運動路徑**。
* IK 前檢＝固定底盤、種子＝當下手臂角；失敗稱「固定底盤 IK 前檢未通過」，不稱全身不可達。
"""
from __future__ import annotations

import math

import numpy as np

import dl3_latch as LT
import gripper_shell_check as G2
import object_target_geometry as OT

JOINT_MARGIN = 0.05        # 控制器有效限位 = LITE6_SAFE ± joint_margin（wgmpc_core.WGMPCConfig.joint_margin）
S_PRE = 0.03


def grasp_params(R_grasp, grasp_depth):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_grasp, float)
    T[:3, 3] = [0.0, float(grasp_depth), 0.0]
    return T, np.array([0.0, -1.0, 0.0])


def map_prior_H(xyz):
    """地圖先驗把手位姿（資產約定旋轉 I：櫃體 yaw 0 擺放）。"""
    v = [float(x) for x in xyz]
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError('map_handle_pose 須為 3 個有限數值')
    H = np.eye(4)
    H[:3, 3] = v
    return H


def target_from_H(T_WH, standoff, T_HG, a_H):
    r = OT.object_target(T_WH, T_HG, a_H, float(standoff))
    if not r['ok']:
        raise ValueError('target:' + r['why'])
    return r['T_WE']


def ik_precheck(K, base_pose, q_arm, T_des):
    """固定底盤 IK：種子＝當下手臂角；解後核對 LITE6_SAFE ± JOINT_MARGIN 與 FK 殘差。"""
    from ammr_wholebody_mpc.arm_limits import LITE6_SAFE
    from ammr_wholebody_mpc.arm_pregrasp import solve_ik
    seed = np.array(list(base_pose[:3]) + list(q_arm), float)
    r = solve_ik(K, seed, np.asarray(T_des, float))
    q = np.asarray(r.q, float)[3:9]
    lo, hi = LITE6_SAFE.lower + JOINT_MARGIN, LITE6_SAFE.upper - JOINT_MARGIN
    margin = np.minimum(q - lo, hi - q)
    out = {'ok': bool(r.ok) and bool(np.all(margin >= 0.0)), 'converged': bool(r.ok), 'q_arm': q.tolist(),
           'fk_pos_residual_m': float(r.pos_err), 'fk_rot_residual_rad': float(r.rot_err),
           'min_effective_margin_rad': float(margin.min()), 'argmin_joint': int(np.argmin(margin)) + 1,
           'base_state': [float(x) for x in base_pose[:3]], 'q_seed': [float(x) for x in q_arm],
           'label': '固定底盤 IK 前檢（非全身可達判定）'}
    if not out['ok']:
        out['why'] = 'fixed_base_ik_precheck_failed:' + ('not_converged' if not r.ok else f'margin_j{out["argmin_joint"]}')
    return out


def latch_and_check(ests, t_latch, R_grasp, grasp_depth, C2, K, base_pose, q_arm):
    """回傳 dict：ok、why、latch、gc2、ik、lock（T_WH）。任何失敗 ⇒ ok False（由呼叫端走 A0 受控停止）。"""
    T_HG, a_H = grasp_params(R_grasp, grasp_depth)
    rep = {'ok': False, 'stage': 'latch'}
    L = LT.latch(ests, t_latch, R_grasp, T_HG, a_H, s=S_PRE)
    rep['latch'] = {k: (np.asarray(v).tolist() if k in ('T_WH', 'T_WE_s') else v) for k, v in L.items()}
    if not L['ok']:
        rep['why'] = 'vision_latch_failed:' + str(L['why'])
        return rep
    rep['stage'] = 'gc2'
    T_WO_est = L['T_WH'] @ np.linalg.inv(OT.trans(C2['gc1']['bar_c']))      # T_WO_est = T_WH_est · T_OH(0)⁻¹
    g = G2.check2(C2, T_WO_est, 0.0, T_HG, a_H, S_PRE)
    rep['gc2'] = {'ok': g['ok'], 'why': g.get('why'), 'object': 'estimated', 'T_WO_est': T_WO_est.tolist(),
                  'meaning': '指定抓取轉換與估計物體模型在目標處相容；不是觀測準確性閘門、不驗證路徑'}
    if not g['ok']:
        rep['why'] = 'vision_precheck_failed:gc2:' + str(g.get('why'))
        return rep
    rep['stage'] = 'ik'
    ik = ik_precheck(K, base_pose, q_arm, L['T_WE_s'])
    rep['ik'] = ik
    if not ik['ok']:
        rep['why'] = 'vision_precheck_failed:' + ik['why']
        return rep
    rep.update({'ok': True, 'stage': 'done', 'lock_T_WH': L['T_WH'].tolist(),
                'geometry_contract': 'checked_estimated_object_static'})
    return rep


def load_contracts():
    """vision 模式啟動時載入 GC1／GC2 契約（含 GC1 凍結清單逐檔核對）；失敗 ⇒ 拒絕啟動。"""
    C2 = G2.load_contract2()
    return C2


