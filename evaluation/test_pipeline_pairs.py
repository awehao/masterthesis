"""兩個接線漏洞的反例測試（離線，不開模擬器）。

1. **相位名稱對應**：求解端發布的相位必須真的能觸發 S1 的接觸例外。
2. **必要配對列**：最近障礙物是例外對象時，其他物件仍要有自己的距離列。
"""
from __future__ import annotations
import os, sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_link_distance import forced_pair_rows            # noqa
from ammr_wholebody_mpc.arm_link_geometry import Obstacle, obstacle_distances  # noqa
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics      # noqa
from ammr_wholebody_mpc.wholebody_safety_filter import (                     # noqa
    DetectionPoint, SafetyConfig, STATUS_OK, _rows_from_points)
from coman_pull_policy import PHASES as POLICY_PHASES                        # noqa
from coman_pull_solver_node import PHASE_MAP                                 # noqa

K = WholeBodyKinematics.from_urdf_file(
    os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
Q = np.zeros(len(K.dof_names))
S1 = yaml.safe_load(open(os.path.join(
    HERE, 'results/specs/wb_coman_drawer20_supplement_s1.yaml'), encoding='utf-8'))


def box(name, center, size):
    o = Obstacle(name=name, model='', kind='box')
    o.size = np.array(size, float)
    T = np.eye(4); T[:3, 3] = np.array(center, float)
    o.T_world_link, o.T_link_collision = T, np.eye(4)
    return o


def pt(link, obs, d=0.030):
    return DetectionPoint(frame=link, p=np.zeros(3), n=np.array([0.0, 1.0, 0.0]),
                          d=d, status=STATUS_OK, offset=np.zeros(3), rho=0.0,
                          obs=obs)


def main() -> int:
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f'  {name:58s} {"ok" if cond else "**錯**"}')
        bad += not cond

    # ---------- 1 相位名稱對應 ----------
    check('每個任務相位都有明確對應（不是靠大小寫轉換）',
          all(p in PHASE_MAP for p in POLICY_PHASES))
    check('未列出的相位對應為 unknown',
          PHASE_MAP.get('NO_SUCH_PHASE', 'unknown') == 'unknown')
    spec_phases = set()
    for v in S1['pair_avoidance']['contact_pairs'].values():
        spec_phases.update(v)
    produced = set(PHASE_MAP.values())
    orphan = spec_phases - produced
    check(f'S1 的接觸相位都有產生端（孤兒：{sorted(orphan) or "無"}）', not orphan)

    cfg = SafetyConfig(
        contact_pairs={k: list(v) for k, v in
                       S1['pair_avoidance']['contact_pairs'].items()},
        d0_by_pair={k: float(v) for k, v in
                    S1['pair_avoidance']['d0_by_pair'].items()})
    for policy_phase in ('ENGAGE_WAIT', 'PULL', 'HOLD', 'RELEASE_WAIT'):
        cfg.phase = PHASE_MAP[policy_phase]
        A, b, _, _ = _rows_from_points(K, Q, [pt('uflite_finger1', 'handle_bar')],
                                       cfg, np.zeros(9))
        check(f'發布 {policy_phase}→{cfg.phase} 時接觸例外**確實生效**', len(b) == 0)
    cfg.phase = 'PULL'                      # 未經對應直接送任務相位名
    A, b, _, _ = _rows_from_points(K, Q, [pt('uflite_finger1', 'handle_bar')],
                                   cfg, np.zeros(9))
    check('直接送未對應的名稱（PULL）→ 例外不生效（fail closed）', len(b) == 1)
    for policy_phase in ('APPROACH', 'RETREAT'):
        cfg.phase = PHASE_MAP[policy_phase]
        A, b, _, _ = _rows_from_points(K, Q, [pt('uflite_finger1', 'handle_bar')],
                                       cfg, np.zeros(9))
        check(f'{policy_phase}→{cfg.phase}：不得有接觸例外', len(b) == 1)

    # ---------- 2 必要配對列 ----------
    # 手指取樣點附近：橫桿較近、面板較遠
    W = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]])
    loc = W.copy()
    bar = box('handle_bar', [0.0, 0.020, 0.0], [0.20, 0.010, 0.010])
    panel = box('drawer_front_panel', [0.0, 0.060, 0.0], [0.575, 0.018, 0.26])
    d_all, _, which = obstacle_distances(W, [bar, panel])
    check('前提：最近障礙物是橫桿', all(w == 'handle_bar' for w in which))
    normal_rows = [pt('uflite_finger1', 'handle_bar', d=float(d_all[0]))]
    cfg.phase = 'pull'
    A, b, _, _ = _rows_from_points(K, Q, normal_rows, cfg, np.zeros(9))
    check('**只有最近列時**：進入接觸相位後完全沒有屏障列（漏洞重現）', len(b) == 0)

    extra = forced_pair_rows(W, loc, panel, 1, li=0, rho=0.0, status=STATUS_OK,
                             age=0.0, occ_of=lambda p, v: 0.0, max_range=1.0)
    check('必要配對列產生了手指—面板的列', len(extra) > 0)
    check('補列帶的障礙物索引指向面板',
          all(abs(r[15] - 1.0) < 1e-9 for r in extra))
    d_panel = min(r[6] for r in extra)
    check('補列的距離是**對面板**量的（大於對橫桿的距離）',
          d_panel > float(d_all.min()))
    with_extra = normal_rows + [pt('uflite_finger1', 'drawer_front_panel',
                                   d=float(d_panel))]
    A, b, _, _ = _rows_from_points(K, Q, with_extra, cfg, np.zeros(9))
    check('加入必要配對列後：接觸相位仍保有**面板**的屏障列', len(b) == 1)

    # 例外對象之外的連桿不受影響
    A, b, _, _ = _rows_from_points(K, Q, [pt('link4', 'handle_bar')], cfg,
                                   np.zeros(9))
    check('link4—橫桿在同一相位仍有屏障列', len(b) == 1)

    print('管線配對接線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
