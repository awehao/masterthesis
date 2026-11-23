"""求解端與安全層必須由**同一份 22 欄訊息**建出**相同的屏障**（離線）。

先前求解端只取到 `obs`，漏了 `v_obs`／`v_obs_state`／`d_lb`，於是
**安全層用新模型、求解器把障礙物當靜態並用舊的 d − rho**。
十五組間距相同不代表兩端約束相同 —— 約束由整列欄位決定。
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
    VOBS_UNKNOWN, _rows_from_points, detection_point_from_row)

K = WholeBodyKinematics.from_urdf_file(
    os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
N = len(K.dof_names)
Q = np.zeros(N)
LINKS = ['uflite_gripper_link', 'uflite_finger1', 'link6']
OBS = ['handle_bar', 'drawer_front_panel', 'handle_post_l']


def row(li, oi, d, rho, vo, vst, dlb, dlbv):
    """22 欄距離列（欄位順序與 arm_link_distance.FIELDS 相同）。"""
    return ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, d, float(STATUS_OK), 0.0, 0.0,
             float(li), 0.0, 0.0, 0.0, rho, float(oi)]
            + list(vo) + [float(vst), dlb, dlbv])


def old_parse(r, links, obs_names):
    """**修補前**的求解端解析：只帶 obs。作為反例對照。"""
    li = int(r[10])
    oi = int(r[15])
    return DetectionPoint(frame=links[li], p=np.asarray(r[0:3], float),
                          n=np.asarray(r[3:6], float), d=float(r[6]),
                          status=int(r[7]), age=float(r[8]),
                          occluded=bool(r[9] >= 0.5),
                          offset=np.asarray(r[11:14], float),
                          rho=float(r[14]),
                          obs=(obs_names[oi] if 0 <= oi < len(obs_names)
                               else None))


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:56s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    cfg = SafetyConfig(g_by_pair={'uflite_gripper_link|handle_bar': 0.003,
                                  'uflite_finger1|drawer_front_panel': 0.010,
                                  'link6|handle_post_l': 0.060})
    cfg.phase = 'pull'
    v_in = np.zeros(N)
    v_in[1] = 0.04
    v_in[3] = 0.06

    rows = [
        # 殼—橫桿：帶 G2 下界與物件速度
        row(0, 0, d=0.00869, rho=0.01335, vo=(0.0, -0.018, 0.0),
            vst=VOBS_OK, dlb=0.0072377, dlbv=1.0),
        # 指—面板：有速度、無下界
        row(1, 1, d=0.02807, rho=0.01273, vo=(0.0, -0.018, 0.0),
            vst=VOBS_OK, dlb=0.0, dlbv=0.0),
        # link6—支柱：速度不可用
        row(2, 2, d=0.06668, rho=0.01330, vo=(0.0, 0.0, 0.0),
            vst=VOBS_UNKNOWN, dlb=0.0, dlbv=0.0),
    ]

    # ---- 兩端用共用解析 ⇒ 屏障必須完全相同 ----
    pts_safety = [detection_point_from_row(r, LINKS, OBS) for r in rows]
    pts_solver = [detection_point_from_row(r, LINKS, OBS) for r in rows]
    As, bs, caps, _ = _rows_from_points(K, Q, pts_safety, cfg, v_in)
    Ao, bo, capo, _ = _rows_from_points(K, Q, pts_solver, cfg, v_in)
    check('兩端 A 逐位元相同', np.array_equal(np.array(As), np.array(Ao)))
    check('兩端 b 逐位元相同', np.array_equal(np.array(bs), np.array(bo)))
    check('兩端速度上限相同', caps == capo)

    # ---- **漏洞重現**：舊解析會給出不同的屏障 ----
    pts_old = [old_parse(r, LINKS, OBS) for r in rows]
    Aold, bold, capold, _ = _rows_from_points(K, Q, pts_old, cfg, v_in)
    check('舊解析的 b 與安全層**不同**（漏洞重現）',
          not np.allclose(np.array(bold), np.array(bs)))
    # 逐項說明差異來源
    i_shell = 0
    check('殼—橫桿：舊解析用 d − rho（負值），新解析用 G2 下界',
          float(bold[i_shell]) < float(bs[i_shell]),
          f'  {float(bold[i_shell]):+.5f} → {float(bs[i_shell]):+.5f}')
    # 隔離**距離項**：用同一列但障礙物宣告為靜止（v_obs = 0），
    # 否則零命令下相對接近速度仍非零（D2 的正確行為），會混進速度項。
    z = np.zeros(N)
    r_static = row(0, 0, d=0.00869, rho=0.01335, vo=(0.0, 0.0, 0.0),
                   vst=VOBS_STATIC, dlb=0.0072377, dlbv=1.0)
    pts_safety0 = [detection_point_from_row(r_static, LINKS, OBS)]
    pts_old0 = [old_parse(r_static, LINKS, OBS)]
    _, bs0, _, _ = _rows_from_points(K, Q, pts_safety0, cfg, z)
    _, bo0, _, _ = _rows_from_points(K, Q, pts_old0, cfg, z)
    exp_new = cfg.alpha * (0.0072377 - 0.003)
    exp_old = cfg.alpha * ((0.00869 - 0.01335) - 0.003)
    check('靜止障礙物、零命令：新解析 rhs = alpha(d_lb − g_pair)',
          abs(float(bs0[i_shell]) - exp_new) < 1e-12,
          f'  {float(bs0[i_shell]):+.6f}')
    check('同上：舊解析 rhs = alpha(d − rho − g_pair)，為**負**',
          abs(float(bo0[i_shell]) - exp_old) < 1e-12 and exp_old < 0.0,
          f'  {float(bo0[i_shell]):+.6f}')
    check('差距即 rho 扣除造成的 (d_lb − (d − rho))',
          abs((float(bs0[i_shell]) - float(bo0[i_shell]))
              - cfg.alpha * (0.0072377 - (0.00869 - 0.01335))) < 1e-12,
          f'  {(float(bs0[i_shell])-float(bo0[i_shell]))/cfg.alpha*1000:+.2f} mm')
    check('速度不可用的列：舊解析沒有套退化上限',
          not np.isfinite(capold) and np.isfinite(caps),
          f'  舊 {capold} → 新 {caps}')

    # ---- 缺欄位的舊寬度訊息：兩端行為也要一致（fail closed）----
    narrow = [r[:16] for r in rows]
    pn = [detection_point_from_row(r, LINKS, OBS) for r in narrow]
    check('舊寬度訊息：速度視為未知（不當成零速）',
          all(p.v_obs_state == VOBS_UNKNOWN for p in pn))
    check('舊寬度訊息：下界為 None（不把 0.0 當下界）',
          all(p.d_lb is None for p in pn))
    An, bn, capn, _ = _rows_from_points(K, Q, pn, cfg, v_in)
    check('舊寬度訊息：套退化上限', np.isfinite(capn))

    # ---- 連桿索引越界 ⇒ 兩端都回 None ----
    bad_row = row(99, 0, 0.01, 0.01, (0.0, 0.0, 0.0), VOBS_STATIC, 0.0, 0.0)
    check('連桿索引越界回傳 None（不建立無效列）',
          detection_point_from_row(bad_row, LINKS, OBS) is None)

    print('兩端屏障一致性離線測試：'
          + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
