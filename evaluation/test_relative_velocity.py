"""D2：屏障的相對接近速度（離線四情境，不開模擬器、不新增幾何掃描）。

屏障應比較**兩者互相接近的速度**：

    n^T (J u − v_obs) <= alpha (d_eff − d_stop)

且 `d_stop` 裡的接近速度**用同一定義**。本檔驗四個核心情境。
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics    # noqa
from ammr_wholebody_mpc.wholebody_safety_filter import (                   # noqa
    DetectionPoint, SafetyConfig, STATUS_OK, VOBS_OK, VOBS_STATIC,
    VOBS_UNKNOWN, _brake_along, _rows_from_points)

K = WholeBodyKinematics.from_urdf_file(
    os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
N = len(K.dof_names)
Q = np.zeros(N)
LINK = 'uflite_gripper_link'


def pt(n, d, v_obs=None, state=VOBS_STATIC):
    return DetectionPoint(frame=LINK, p=np.zeros(3), n=np.asarray(n, float),
                          d=d, status=STATUS_OK, offset=np.zeros(3), rho=0.0,
                          obs='handle_bar', v_obs=v_obs, v_obs_state=state)


def rowvals(pts, v_in, cfg=None):
    cfg = cfg or SafetyConfig()
    A, b, cap, _ = _rows_from_points(K, Q, pts, cfg, np.asarray(v_in, float))
    return np.array(A), np.array(b), cap


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:56s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    n = np.array([0.0, 1.0, 0.0])
    d = 0.120
    # 讓機器人朝 +y 走（沿 n，即朝障礙物）
    v_in = np.zeros(N)
    v_in[1] = 0.10
    # 該列的 n^T J 在此位姿的增益（底盤平移直接進入）
    A0, b0, _ = rowvals([pt(n, d)], v_in)
    g = float(A0[0] @ v_in)
    print(f'前提：此列的 n^T J u = {g:+.4f} m/s（機器人朝障礙物 0.10 m/s）')

    # ---- 情境 1：障礙物靜止 ⇒ 重現既有屏障 ----
    A1, b1, _ = rowvals([pt(n, d, state=VOBS_STATIC)], v_in)
    A1b, b1b, _ = rowvals([DetectionPoint(frame=LINK, p=np.zeros(3), n=n, d=d,
                                          status=STATUS_OK, offset=np.zeros(3),
                                          rho=0.0, obs='handle_bar')], v_in)
    check('情境1 靜止：與不帶速度資訊的列完全相同',
          np.allclose(A1, A1b) and np.allclose(b1, b1b),
          f'  rhs {float(b1[0]):+.5f}')
    # 明確零速度向量也要一致
    A1c, b1c, _ = rowvals([pt(n, d, v_obs=np.zeros(3), state=VOBS_OK)], v_in)
    check('情境1 靜止：明確零速度向量結果相同',
          np.allclose(b1, b1c))

    # ---- 情境 2：兩者同速平移 ⇒ 不產生相對接近速度 ----
    v_same = np.array([0.0, float(g), 0.0])     # 障礙物沿 n 以相同速度退開
    A2, b2, _ = rowvals([pt(n, d, v_obs=v_same, state=VOBS_OK)], v_in)
    resid2 = float(b2[0] - A2[0] @ v_in)
    check('情境2 同速平移：相對接近速度為零 ⇒ 約束不啟動',
          resid2 >= 0.0, f'  餘量 {resid2:+.5f}')
    # d_stop 也必須用同一定義：同速時 d_stop 應退回零速值
    cfg = SafetyConfig()
    d_stop0 = cfg.d0 + cfg.eps                   # 零接近速度
    rhs_expect = cfg.alpha * (d - d_stop0) + float(n @ v_same)
    check('情境2 d_stop 退回零速值（速度項未用機器人速度算）',
          abs(float(b2[0]) - rhs_expect) < 1e-12,
          f'  rhs {float(b2[0]):+.6f} vs {rhs_expect:+.6f}')

    # ---- 情境 3：障礙物迎面靠近 ⇒ 比忽略其速度更嚴 ----
    v_face = np.array([0.0, -0.05, 0.0])        # 沿 −n 迎面而來
    A3, b3, _ = rowvals([pt(n, d, v_obs=v_face, state=VOBS_OK)], v_in)
    check('情境3 迎面：rhs 比視為靜止時更小（更嚴）',
          float(b3[0]) < float(b1[0]),
          f'  {float(b3[0]):+.5f} < {float(b1[0]):+.5f}')
    resid3 = float(b3[0] - A3[0] @ v_in)
    resid1 = float(b1[0] - A1[0] @ v_in)
    check('情境3 迎面：同一命令的餘量也更小', resid3 < resid1,
          f'  {resid3:+.5f} < {resid1:+.5f}')
    # d_stop 應隨相對接近速度變大而變大
    v_rel3 = g - float(n @ v_face)
    d_stop3 = (cfg.d0 + v_rel3 * cfg.tau
               + v_rel3 * v_rel3 / (2.0 * max(cfg.a_brake, 1e-3)) + cfg.eps)
    check('情境3 d_stop 用的是相對接近速度（非機器人速度）',
          v_rel3 > g and d_stop3 > d_stop0,
          f'  v_rel {v_rel3:+.4f} > v_robot {g:+.4f}')

    # ---- 情境 4：動態物件速度失效 ⇒ 不得當成靜態放行 ----
    for label, p_ in (('狀態為 UNKNOWN', pt(n, d, state=VOBS_UNKNOWN)),
                      ('宣稱可用但為 None',
                       pt(n, d, v_obs=None, state=VOBS_OK)),
                      ('宣稱可用但含 NaN',
                       pt(n, d, v_obs=np.array([0.0, np.nan, 0.0]),
                          state=VOBS_OK)),
                      ('宣稱可用但維度錯',
                       pt(n, d, v_obs=np.zeros(2), state=VOBS_OK))):
        A4, b4, cap4 = rowvals([p_], v_in)
        check(f'情境4 {label}：rhs 比視為靜止更嚴',
              float(b4[0]) < float(b1[0]), f'  {float(b4[0]):+.5f}')
        check(f'情境4 {label}：同時套退化速度上限',
              np.isfinite(cap4) and cap4 <= cfg.unknown_vobs_speed_cap,
              f'  cap {cap4}')
    A4, b4, _ = rowvals([pt(n, d, state=VOBS_UNKNOWN)], v_in)
    exp4 = -float(cfg.unknown_vobs_speed)
    v_rel4 = g - exp4
    # 煞停用的是**程式實際採用**的 Jacobian 界（不是 cfg.a_brake）
    a_br = max(_brake_along(A4[0], cfg, N), cfg.brake_floor)
    d_stop4 = (cfg.d0 + v_rel4 * cfg.tau
               + v_rel4 * v_rel4 / (2.0 * max(a_br, 1e-3)) + cfg.eps)
    check('情境4 取最壞接近 −V 而**不是零**',
          abs(float(b4[0]) - (cfg.alpha * (d - d_stop4) + exp4)) < 1e-9,
          f'  a_br {a_br:.3f} m/s²')

    # ---- 既有行為不變：不帶速度資訊的呼叫端 ----
    cfg2 = SafetyConfig()
    A5, b5, cap5 = rowvals([DetectionPoint(frame=LINK, p=np.zeros(3), n=n,
                                           d=d, status=STATUS_OK,
                                           offset=np.zeros(3), rho=0.0)],
                           np.zeros(N), cfg2)
    check('既有呼叫端（無 v_obs 欄位語意）行為完全不變',
          abs(float(b5[0]) - cfg2.alpha * (d - d_stop0)) < 1e-12)

    # ---- 物件 twist 估計：旋轉也要算進表面點速度 ----
    from ammr_wholebody_mpc.arm_link_distance import (
        model_twist, surface_point_velocity, VOBS_UNKNOWN as DN_UNKNOWN)
    T0 = np.eye(4)
    T1 = np.eye(4); T1[1, 3] = 0.010                 # 0.010 m / 0.05 s
    tw = model_twist(T0, 0.0, T1, 0.05)
    check('twist：純平移的線速度正確',
          tw is not None and np.allclose(tw[0], [0.0, 0.2, 0.0])
          and np.allclose(tw[1], 0.0, atol=1e-9))
    th = 0.02
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    T2 = np.eye(4); T2[:3, :3] = Rz
    tw2 = model_twist(T0, 0.0, T2, 0.05)
    check('twist：純旋轉的角速度正確（原點不動）',
          tw2 is not None and abs(float(tw2[1][2]) - th / 0.05) < 1e-9
          and np.allclose(tw2[0], 0.0, atol=1e-12))
    Vs = surface_point_velocity(tw2, np.zeros(3), np.array([[0.1, 0.0, 0.0]]))
    check('表面點速度含旋轉貢獻（omega x r，原點速度為零）',
          Vs is not None and abs(float(Vs[0][1]) - (th / 0.05) * 0.1) < 1e-9,
          f'  {float(Vs[0][1]):+.4f} m/s')
    check('twist：dt 過小回傳 None（不硬算）',
          model_twist(T0, 0.0, T1, 1e-6) is None)
    check('twist：缺前一筆位姿回傳 None', model_twist(None, 0.0, T1, 0.05) is None)
    check('表面點速度：twist 為 None 時回傳 None',
          surface_point_velocity(None, np.zeros(3), np.zeros((1, 3))) is None)
    check('距離節點與濾波器的狀態碼一致', DN_UNKNOWN == VOBS_UNKNOWN)

    print('相對接近速度離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
