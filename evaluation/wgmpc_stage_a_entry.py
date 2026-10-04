"""階段 A 的啟動核對與目標計算。**判定規則與執行期讀回共用同一份函式。**

為什麼要鎖這些
--------------
階段 A 的每一個結論都依賴一組前提。任何一項被默默改掉，趟次仍會跑完、仍會
產出數字，但那些數字衡量的是另一件事：
  * 真值模式沒開 ⇒ 量到的是 blind_approach_cap 的節流行為（152/209 週期被改命令）
  * `freespace_confirmed` 被設真 ⇒ 宣稱場景裡沒有東西，而櫃體與抽屜就在那裡
  * 具名幾何不齊 ⇒ 屏障少了列，而少列不會有錯誤訊息
  * 模式不是 solver_drawer ⇒ 底盤或手臂分量被整筆拒收
  * 求解器與執行端的 joint_margin 不一致 ⇒ 被約束的值與被保護的值不是同一個

所以這些是**啟動中止條件**，不是警告。
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

import drawer_asset as DA                                      # noqa: E402
from ammr_wholebody_mpc.arm_link_distance import (              # noqa: E402
    parse_obstacles, parse_scene_truth, validate_scene_truth)

# 正式預抓取姿態：工具 z 指世界 +y、手指閉合方向世界 +z、工具 x 指世界 −x
R_DES = np.array([[-1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0]])
STAGE_A_BACKOFF_M = 0.120
STAGE_A_JOINT_MARGIN = 0.05
STAGE_A_MODE = 'solver_drawer'


class EntryError(RuntimeError):
    """啟動前提不成立。呼叫端必須中止，不得降級繼續。"""


def _as_bool(v):
    """ROS 參數讀回的布林可能是字串。**不認得的值不當成假。**"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('true', '1', 'yes'):
        return True
    if s in ('false', '0', 'no'):
        return False
    raise EntryError(f'無法判讀布林值 {v!r}：不以「看起來不像真」當成假')


# ------------------------------------------------------------------ 目標
def compute_pre_contact_pause(spec, drawer_pose_xy, q_d, backoff_m,
                              stamp_sim_t=None, source='unspecified'):
    """由抽屜的**當步位姿**算出六維接觸前暫停位姿。

    回傳 `(p_world, R_world, meta)`。`meta` 記下目標的來源與時間 —— 事後要能
    分辨這個目標是由哪一筆抽屜位姿、在哪個模擬時刻算出來的。把手隨抽屜移動，
    所以 `q_d` 一變目標就跟著變；**這也是為什麼抽屜被推動必須另外標記**，
    否則目標跟著跑、紀錄裡看不出有人碰過它。
    """
    if backoff_m <= 0.0 or not math.isfinite(backoff_m):
        raise EntryError(f'退讓 {backoff_m!r} 不合法')
    g = np.asarray(DA.grasp_tcp_world(spec, drawer_pose_xy, q_d), float)
    p = g - float(backoff_m) * np.array([0.0, 1.0, 0.0])   # 沿工具 z 後退
    meta = {
        'target_source': source,
        'stamp_sim_t': (None if stamp_sim_t is None else float(stamp_sim_t)),
        'wall_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'drawer_pose_xy': [float(drawer_pose_xy[0]), float(drawer_pose_xy[1])],
        'drawer_opening_m': float(q_d),
        'approach_backoff_m': float(backoff_m),
        'grasp_tcp_world': g.tolist(),
        'tcp_world': p.tolist(),
        'R_world': R_DES.tolist(),
        'asset_schema': spec.get('schema'),
        'asset_name': spec.get('name'),
        'note': '把手隨抽屜移動 ⇒ 目標由當步位姿算出；抽屜若被推動，目標會跟著變',
    }
    return p, R_DES.copy(), meta


def check_target(p_used, R_used, spec, drawer_pose_xy, q_d, backoff_m,
                 tol_m=1e-9, tol_rot=1e-9):
    """核對趟次實際使用的目標，是否就是由當步位姿算出的那一個。"""
    p, R, _ = compute_pre_contact_pause(spec, drawer_pose_xy, q_d, backoff_m)
    dp = float(np.max(np.abs(np.asarray(p_used, float) - p)))
    dr = float(np.max(np.abs(np.asarray(R_used, float) - R)))
    if dp > tol_m:
        raise EntryError(f'目標位置與當步位姿推得的不符：最大差 {dp:.6e} m')
    if dr > tol_rot:
        raise EntryError(f'目標姿態與 R_DES 不符：最大差 {dr:.6e}')
    if len(np.asarray(p_used, float)) != 3 or np.asarray(R_used).shape != (3, 3):
        raise EntryError('目標必須是六維（位置三 ＋ 姿態三自由度）')
    return dp, dr


# -------------------------------------------------------- 距離節點與安全層
def expected_drawer_params(spec, drawer_pose_xy, q_d, model='drawer_unit'):
    """由資產算出距離節點該有的 obstacles／scene_truth_approved。

    **幾何只有一份來源**。這裡與 gen_drawer_obstacle_params.py 走同一條路徑，
    所以核對的是「節點拿到的」對「資產算出的」，不是對一份手抄的名單。
    """
    obstacles, approved = [], []
    for item in DA.shapes_world(spec, drawer_pose_xy, q_d):
        nm = item[0]
        md = '' if nm.startswith('cabinet/') else model
        if item[1] == 'box':
            c, s = item[2], item[3]
            obstacles.append(f'{nm}:{md}:box:{s[0]},{s[1]},{s[2]}:'
                             f'{c[0]},{c[1]},{c[2]}:')
        else:
            c, r, L = item[2], item[3], item[4]
            obstacles.append(f'{nm}:{md}:cylinder:{r},{L}:'
                             f'{c[0]},{c[1]},{c[2]}:0,{np.pi / 2},0')
        approved.append(f'{nm}:{md}')
    return obstacles, approved


def check_dist_params(params, spec, drawer_pose_xy, q_d, model='drawer_unit'):
    """距離節點：真值模式、核准清單、具名幾何、geometry。"""
    if not _as_bool(params.get('scene_truth', False)):
        raise EntryError(
            'scene_truth 不是真：這趟會改用 blind_approach_cap 的退化上限，'
            '量到的不是 W-GMPC 的接近行為')
    if str(params.get('geometry', '')) != 'links':
        raise EntryError(f'geometry={params.get("geometry")!r} 不是 links：'
                         f'真值模式只接在這條列建構路徑上')
    got_obs = [s for s in (params.get('obstacles') or []) if str(s).strip()]
    got_ap = [s for s in (params.get('scene_truth_approved') or [])
              if str(s).strip()]
    exp_obs, exp_ap = expected_drawer_params(spec, drawer_pose_xy, q_d, model)
    if sorted(got_obs) != sorted(exp_obs):
        only_got = sorted(set(got_obs) - set(exp_obs))
        only_exp = sorted(set(exp_obs) - set(got_obs))
        raise EntryError(
            f'距離節點的 obstacles 與資產算出的不符：'
            f'節點多了 {only_got[:3]}；資產有而節點沒有 {only_exp[:3]}')
    if sorted(got_ap) != sorted(exp_ap):
        raise EntryError('scene_truth_approved 與資產算出的具名幾何不符')
    # 啟動前提：名單涵蓋、身分一致、geometry 正確（與節點同一份函式）
    obs = parse_obstacles(got_obs)
    for o in obs:
        if o.model:
            o.T_world_link = np.eye(4)     # 核對名單用；實跑由位姿話題填
    validate_scene_truth(obs, parse_scene_truth(got_ap), True, 'links')
    return len(obs)


def check_safety_params(params):
    """安全層：`freespace_confirmed` 必須為假，保護參數不得放寬。"""
    if _as_bool(params.get('freespace_confirmed', False)):
        raise EntryError(
            'freespace_confirmed 為真：場景裡有櫃體與抽屜，這個宣稱不成立；'
            '它關掉的是 nodata_speed_cap，與遮蔽是不同的事。'
            '**不要從自由空間趟次複製這一行。**')
    for k, want in (('d0', 0.05), ('eps', 0.03), ('tau', 0.15), ('alpha', 2.0)):
        if k in params and abs(float(params[k]) - want) > 1e-12:
            raise EntryError(f'安全層 {k}={params[k]} 不是原值 {want}：'
                             f'不靠放寬保護讓原解過關')
    return True


def check_margin_consistency(solver_jm, endpoint_jm,
                             want=STAGE_A_JOINT_MARGIN):
    """求解器與執行端的 joint_margin 必須一致且等於約定值。"""
    s, e = float(solver_jm), float(endpoint_jm)
    if abs(s - want) > 1e-12 or abs(e - want) > 1e-12:
        raise EntryError(f'joint_margin 必須是 {want}：'
                         f'求解器 {s}、執行端 {e}（維持原值，不放寬）')
    if abs(s - e) > 1e-12:
        raise EntryError(f'求解器 joint_margin {s} 與執行端 {e} 不一致：'
                         f'被約束的值與被保護的值不是同一個')
    return want


def check_mode(mode, want=STAGE_A_MODE):
    if str(mode) != want:
        raise EntryError(f'模式 {mode!r} 不是 {want!r}：'
                         f'其他模式會把底盤或手臂分量整筆拒收')
    return want


# ------------------------------------------------------------ 執行期讀回
def ros_param(node, name, timeout=10.0):
    """`ros2 param get` 讀回一個參數。讀不到就拋出，**不以預設值代替**。"""
    r = subprocess.run(['ros2', 'param', 'get', node, name],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise EntryError(f'讀不到 {node} 的參數 {name}：{r.stderr.strip()}')
    out = r.stdout.strip()
    # 形如 "Boolean value is: True" / "String values are: ['a', 'b']"
    if ' is: ' in out:
        return out.split(' is: ', 1)[1]
    if ' are: ' in out:
        import ast
        return ast.literal_eval(out.split(' are: ', 1)[1])
    raise EntryError(f'無法判讀 {node}/{name} 的讀回內容：{out!r}')


def main() -> int:
    import argparse, json
    ap = argparse.ArgumentParser(description='階段 A 啟動核對（執行期讀回）')
    ap.add_argument('--asset', default=os.path.join(
        WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
    ap.add_argument('--pose', default='0.0,1.45')
    ap.add_argument('--opening', type=float, default=0.0)
    ap.add_argument('--model', default='drawer_unit')
    ap.add_argument('--dist-node', default='/arm_link_distance')
    ap.add_argument('--safety-node', default='/wholebody_safety')
    ap.add_argument('--solver-joint-margin', type=float, required=True)
    ap.add_argument('--endpoint-joint-margin', type=float, required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    spec = DA.load(a.asset)
    pose = tuple(float(v) for v in a.pose.split(','))
    rep = {}
    try:
        dp = {k: ros_param(a.dist_node, k) for k in
              ('scene_truth', 'geometry', 'obstacles', 'scene_truth_approved')}
        rep['n_named_geometry'] = check_dist_params(dp, spec, pose, a.opening,
                                                    a.model)
        sp = {k: ros_param(a.safety_node, k) for k in ('freespace_confirmed',)}
        check_safety_params(sp)
        rep['freespace_confirmed'] = False
        rep['joint_margin'] = check_margin_consistency(
            a.solver_joint_margin, a.endpoint_joint_margin)
        rep['mode'] = check_mode(a.mode)
        p, R, meta = compute_pre_contact_pause(
            spec, pose, a.opening, STAGE_A_BACKOFF_M,
            source=f'{a.dist_node} 的 obstacles 所用的抽屜位姿（啟動核對時刻）')
        rep['pre_contact_pause'] = meta
        rep['verdict'] = 'pass'
    except EntryError as e:
        rep['verdict'] = 'fail'
        rep['error'] = str(e)
        print(f'[階段A] **啟動核對未通過**：{e}', flush=True)
        if a.out:
            open(a.out, 'w').write(json.dumps(rep, ensure_ascii=False, indent=1))
        return 70
    print(f'[階段A] 啟動核對通過：具名幾何 {rep["n_named_geometry"]} 個、'
          f'scene_truth=true、freespace_confirmed=false、'
          f'mode={rep["mode"]}、joint_margin={rep["joint_margin"]}', flush=True)
    print(f'[階段A] 接觸前暫停位姿 TCP {np.round(p, 4).tolist()}'
          f'（退讓 {STAGE_A_BACKOFF_M} m，來源：{rep["pre_contact_pause"]["target_source"]}，'
          f'牆鐘 {rep["pre_contact_pause"]["wall_utc"]}）', flush=True)
    if a.out:
        open(a.out, 'w').write(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
