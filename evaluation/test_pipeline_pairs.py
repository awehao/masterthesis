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
from ammr_wholebody_mpc.arm_link_distance import (                           # noqa
    expand_pair_rows, forced_pair_rows)
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

    # ---------- 3 漏列不限於面板：任一非例外物件都要有列 ----------
    # 假設幾何（非本趟量測）：橫桿最近且允許接觸、面板 28 mm 合格、
    # 支柱 40 mm 應受一般 50 mm 規則限制。
    post = box('handle_post_l', [0.0, 0.040, 0.0], [0.02, 0.02, 0.06])
    obs_all = [bar, panel, post]
    names = [o.name for o in obs_all]
    d3, _, which3 = obstacle_distances(W, obs_all)
    check('前提：三個物件中最近者仍是橫桿',
          all(w == 'handle_bar' for w in which3))
    only_panel = [pt('uflite_finger1', 'handle_bar', d=float(d3.min())),
                  pt('uflite_finger1', 'drawer_front_panel', d=0.028)]
    A, b, _, _ = _rows_from_points(K, Q, only_panel, cfg, np.zeros(9))
    names_in = {p_.obs for p_ in only_panel} - {'handle_bar'}
    check('**只補面板時**：支柱完全沒有列（面板測試會通過但規則不成立）',
          len(b) == 1 and 'handle_post_l' not in names_in)

    pairs, exempt = expand_pair_rows(
        ['uflite_finger1:*', 'uflite_finger2:*'],
        ['uflite_finger1:handle_bar', 'uflite_finger2:handle_bar'], names)
    check('link:* 展開到兩指',
          set(pairs) == {'uflite_finger1', 'uflite_finger2'})
    check('展開涵蓋所有非例外物件（面板與支柱都在）',
          all(set(pairs[lk]) == {'drawer_front_panel', 'handle_post_l'}
              for lk in pairs))
    check('例外對象（橫桿）不在展開結果內',
          all('handle_bar' not in v for v in pairs.values()))
    check('免列清單記錄了橫桿',
          all(exempt[lk] == {'handle_bar'} for lk in pairs))

    rows_all = [pt('uflite_finger1', 'handle_bar', d=float(d3.min()))]
    for obn in pairs['uflite_finger1']:
        ob = {o.name: o for o in obs_all}[obn]
        ex = forced_pair_rows(W, loc, ob, names.index(obn), li=0, rho=0.0,
                              status=STATUS_OK, age=0.0,
                              occ_of=lambda p, v: 0.0, max_range=1.0)
        check(f'{obn} 產生了必要配對列', len(ex) > 0)
        rows_all.append(pt('uflite_finger1', obn, d=min(r[6] for r in ex)))
    A, b, _, _ = _rows_from_points(K, Q, rows_all, cfg, np.zeros(9))
    kept = {p_.obs for p_ in rows_all if p_.obs != 'handle_bar'}
    check('橫桿最近＋面板合格＋支柱違規：**支柱仍有限制列**',
          len(b) == 2 and 'handle_post_l' in kept)
    # 支柱 40 mm < 一般 d0 50 mm ⇒ 該列的 rhs 必須是**負的**（要求遠離）
    A1, b1, _, _ = _rows_from_points(
        K, Q, [pt('uflite_finger1', 'handle_post_l', d=0.040)], cfg,
        np.zeros(9))
    check('支柱那一列確實在限制方向（rhs < 0，要求遠離）', float(b1[0]) < 0.0)
    # **S1 的 10 mm 並沒有打通面板那一對。** 零接近速度下
    #   d_stop = d0 + eps = 10 + 30 = 40 mm > 設計間距 27.9 mm
    # eps（幾何＋量測餘裕）預設 30 mm，本身就大於 27.9 mm ⇒
    # **任何 d0 ≥ 0 都放行不了**。這是規格缺口，不在此處調門檻。
    A2, b2, _, _ = _rows_from_points(
        K, Q, [pt('uflite_finger1', 'drawer_front_panel', d=0.0279)], cfg,
        np.zeros(9))
    check('面板在設計間距 27.9 mm 下**仍是限制方向**（d0+eps = 40 mm）',
          float(b2[0]) < 0.0)
    cfg0 = SafetyConfig(contact_pairs=dict(cfg.contact_pairs),
                        d0_by_pair={'uflite_finger1|drawer_front_panel': 0.0})
    cfg0.phase = 'pull'
    A3, b3, _, _ = _rows_from_points(
        K, Q, [pt('uflite_finger1', 'drawer_front_panel', d=0.0279)], cfg0,
        np.zeros(9))
    check('連 d0 = 0 都仍是限制方向（eps 單獨 30 mm > 27.9 mm）',
          float(b3[0]) < 0.0)
    check('支柱的限制比面板嚴（一般 d0 50 mm vs 配對 10 mm）',
          float(b1[0]) < float(b2[0]))
    check('支柱沿用一般門檻，未被 10 mm 設定波及',
          'uflite_finger1|handle_post_l' not in cfg.d0_by_pair)

    # 例外對象之外的連桿不受影響
    A, b, _, _ = _rows_from_points(K, Q, [pt('link4', 'handle_bar')], cfg,
                                   np.zeros(9))
    check('link4—橫桿在同一相位仍有屏障列', len(b) == 1)

    print('管線配對接線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
