"""配對層級例外的離線驗證：**只有指定配對在指定相位**被放寬。

驗證對象是 wholebody_safety_filter 的屏障列產生器，不開模擬器。
"""
from __future__ import annotations
import os, sys
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa
from ammr_wholebody_mpc.wholebody_safety_filter import (                  # noqa
    DetectionPoint, SafetyConfig, STATUS_OK, _rows_from_points)

K = WholeBodyKinematics.from_urdf_file(
    os.path.join(WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
Q = np.zeros(len(K.dof_names))


def pt(link, obs, d=0.030):
    return DetectionPoint(frame=link, p=np.zeros(3), n=np.array([0.0, 1.0, 0.0]),
                          d=d, status=STATUS_OK, offset=np.zeros(3), rho=0.0,
                          obs=obs)


def rhs_for(cfg, points):
    A, b, cap, owner = _rows_from_points(K, Q, points, cfg, np.zeros(9))
    return len(b), (b[0] if b else None)


def main() -> int:
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f'  {name:56s} {"ok" if cond else "**錯**"}')
        bad += not cond

    base = SafetyConfig()
    cfg = SafetyConfig(
        d0_by_pair={'uflite_finger1|drawer_front_panel': 0.010,
                    'uflite_finger2|drawer_front_panel': 0.010},
        contact_pairs={'uflite_finger1|handle_bar': ['engage', 'pull', 'hold'],
                       'uflite_finger2|handle_bar': ['engage', 'pull', 'hold']})

    # 1 例外配對：只改 d0，**速度項仍在**
    n0, b0 = rhs_for(base, [pt('uflite_finger1', 'drawer_front_panel')])
    n1, b1 = rhs_for(cfg, [pt('uflite_finger1', 'drawer_front_panel')])
    check('手指—面板：仍產生屏障列（沒有被刪除）', n1 == 1)
    check('手指—面板：放寬後 rhs 變大（d0 由 50→10 mm）', b1 > b0)
    cfg_v = SafetyConfig(d0_by_pair=cfg.d0_by_pair)
    _, b_static = rhs_for(cfg_v, [pt('uflite_finger1', 'drawer_front_panel')])
    v_in = np.zeros(9); v_in[3] = 0.5      # 讓該點有接近速度
    A, b_dyn, _, _ = _rows_from_points(K, Q, [pt('uflite_finger1', 'drawer_front_panel')],
                                       cfg_v, v_in)
    check('速度相關停止距離**未取消**（有接近速度時 rhs 更小）',
          b_dyn[0] < b_static)

    # 2 非例外配對：完全不受影響
    for link, obs in (('uflite_finger1', 'cab_side_left'),
                      ('link4', 'drawer_front_panel'),
                      ('link4', 'handle_bar'),
                      ('base_link', 'handle_bar')):
        na, ba = rhs_for(base, [pt(link, obs)])
        nb, bb = rhs_for(cfg, [pt(link, obs)])
        check(f'{link}↔{obs}：規則未改變', na == nb and abs(ba - bb) < 1e-15)

    # 3 接觸例外：只在允許相位、且只對指定連桿
    cfg.phase = 'pull'
    n_bar_f1, _ = rhs_for(cfg, [pt('uflite_finger1', 'handle_bar')])
    check('手指—橫桿：在 pull 相位不產生屏障列（允許接觸）', n_bar_f1 == 0)
    n_bar_l4, _ = rhs_for(cfg, [pt('link4', 'handle_bar')])
    check('**其他連桿—橫桿：仍產生屏障列**（橫桿未從避碰刪除）', n_bar_l4 == 1)
    n_bar_base, _ = rhs_for(cfg, [pt('base_link', 'handle_bar')])
    check('**底盤—橫桿：仍產生屏障列**', n_bar_base == 1)
    cfg.phase = 'approach'
    n_bar_app, _ = rhs_for(cfg, [pt('uflite_finger1', 'handle_bar')])
    check('手指—橫桿：非允許相位仍受一般規則', n_bar_app == 1)
    cfg.phase = None
    n_bar_unk, _ = rhs_for(cfg, [pt('uflite_finger1', 'handle_bar')])
    check('相位未知 ⇒ 無任何例外（fail closed）', n_bar_unk == 1)
    cfg.phase = 'pull'
    n_panel_pull, _ = rhs_for(cfg, [pt('uflite_finger1', 'drawer_front_panel')])
    check('接觸例外不外溢到同連桿的其他物件（手指—面板仍有列）',
          n_panel_pull == 1)

    # 4 預設設定完全不變
    cfg_def = SafetyConfig()
    check('預設 d0_by_pair／contact_pairs 皆空、phase 為 None',
          cfg_def.d0_by_pair == {} and cfg_def.contact_pairs == {}
          and cfg_def.phase is None)
    n_d, b_d = rhs_for(cfg_def, [pt('uflite_finger1', 'handle_bar')])
    check('預設下所有配對一律一般規則', n_d == 1 and abs(b_d - b0) < 1e-15)

    print('配對例外離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
