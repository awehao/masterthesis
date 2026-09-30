"""G8：底盤回授的座標系與 TF 樹慣例（離線，不開 Isaac）。

被修的缺口：協同執行端不發 /odom 與 odom→base_footprint TF，
且 runner 不起 robot_state_publisher ⇒ 求解端 base=False、距離節點整片 NODATA。

本檔固定四個**容易安靜出錯**的慣例，並直接對原始碼核對接線。
"""
from __future__ import annotations
import math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.join(HERE, 'isaac_coman_drawer_sim.py')
RUNNER = os.path.join(HERE, 'run_coman_drawer20.sh')


def rotz(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def twist_to_body(R_fp, v_world, w_world):
    """**與執行端同一式**：世界 → child_frame（本體）。"""
    return R_fp.T @ np.asarray(v_world, float), R_fp.T @ np.asarray(w_world, float)


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:56s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    ep = open(EP, encoding='utf-8').read()
    rn = open(RUNNER, encoding='utf-8').read()

    # ---- 1 twist 必須是本體座標 ----
    R = rotz(math.pi / 2.0)          # 底盤朝 +y（本趟停放 yaw = 90°）
    v_w = np.array([0.0, 0.30, 0.0])  # 世界 +y 前進
    v_b, _ = twist_to_body(R, v_w, np.zeros(3))
    check('1 yaw=90° 時世界 +y 的速度在本體是 +x',
          abs(v_b[0] - 0.30) < 1e-12 and abs(v_b[1]) < 1e-12,
          f'  本體 {np.round(v_b, 4).tolist()}')
    check('1 **不可**直接填世界座標（yaw≠0 時會錯）',
          not np.allclose(v_b, v_w), f'  世界 {v_w.tolist()}')
    R0 = np.eye(3)
    check('1 yaw=0 時兩者碰巧相同 —— 這不代表定義正確',
          np.allclose(twist_to_body(R0, v_w, np.zeros(3))[0], v_w))
    w_w = np.array([0.0, 0.0, 0.5])
    _, w_b = twist_to_body(R, np.zeros(3), w_w)
    check('1 繞 z 的角速度在本體不變（旋轉軸與 z 同向）',
          abs(w_b[2] - 0.5) < 1e-12)

    # ---- 2 TF 只能發 odom → base_footprint ----
    childs = re.findall(r"tr\.child_frame_id = '([^']+)'", ep)
    check('2 TF 的 child 是 base_footprint', childs == ['base_footprint'],
          f'  {childs}')
    check('2 **沒有**發 odom → base_link（否則 base_link 兩個父節點）',
          "child_frame_id = 'base_link'" not in ep)
    parents = re.findall(r"tr\.header\.frame_id = '([^']+)'", ep)
    check('2 TF 的 parent 是 odom', parents == ['odom'], f'  {parents}')

    # ---- 3 /odom 與 TF 同一時間戳、同一模擬時間 ----
    check('3 TF 明確沿用 /odom 的 stamp',
          re.search(r'tr\.header\.stamp = stamp', ep) is not None)
    check('3 stamp 由**模擬時間 t** 組出（非牆鐘）',
          re.search(r'stamp\.sec = int\(t\)', ep) is not None
          and re.search(r'stamp\.nanosec = min\(int\(round\(\(t - int\(t\)\)',
                        ep) is not None)
    check('3 odom 的 frame/child 與 TF 一致',
          "od.header.frame_id = 'odom'" in ep
          and "od.child_frame_id = 'base_footprint'" in ep)

    # ---- 4 位姿取自 prim 的完整變換，不是 yaw 重建 ----
    check('4 位姿來自 base_footprint prim 的世界變換',
          'GetLocalToWorldTransform(FP_PRIM)' in ep)
    check('4 取完整四元數（未用 yaw 重建）',
          'ExtractRotationQuat()' in ep and 'q_yaw(' not in
          ep[ep.index('def publish_base_feedback'):
             ep.index('def publish_base_feedback') + 2500])
    check('4 USD 列向量慣例已轉置',
          re.search(r'_R_fp = \(_R3 / _sc\[:, None\]\)\.T', ep) is not None)
    check('4 找不到 prim 即中止（不靜默跳過）',
          'FP_PRIM is None' in ep and 'return 12' in ep)

    # ---- 5 runner 起 robot_state_publisher，且用對 URDF ----
    check('5 runner 起 robot_state_publisher',
          'robot_state_publisher robot_state_publisher "$URDF_TF"' in rn)
    check('5 rsp 用 **manip** URDF（不是 wholebody）',
          re.search(r'URDF_TF="\$WS/evaluation/models/omni_bot_manip\.urdf"',
                    rn) is not None)
    check('5 距離／安全節點仍用 wholebody URDF 當 FK 參數',
          rn.count('-p wholebody_urdf:="$URDF_WB"') == 2)
    check('5 rsp 用模擬時間', re.search(
        r'robot_state_publisher[^\n]*\n\s*--ros-args -p use_sim_time:=true',
        rn) is not None)
    check('5 兩份 URDF 都存在',
          os.path.exists(os.path.join(HERE, 'models/omni_bot_manip.urdf'))
          and os.path.exists(os.path.join(
              HERE, 'models/omni_bot_wholebody_expanded.urdf')))

    # ---- 6 每週期都發（與 joint_states 同一處）----
    i_js = ep.index('node.js_pub.publish(js)')
    check('6 底盤回授緊接 /joint_states 發布（同一週期）',
          0 < ep.index('node.publish_base_feedback(') - i_js < 1200)

    print('底盤回授座標系測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
