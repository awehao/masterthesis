"""R1：十組配對的一次性定版提案表（離線產生，不重跑既有套件、不開模擬器）。

每組的「相容性」欄位是**用完整屏障算出的零相對速度餘量**（rhs / alpha），
不是 `d − g` 的名目減法。距離項一律取**實際生效**的那一個：
G2 配對用當前位姿的認證下界，其餘用 `d − rho`。

候選值的選法（明寫，不是「低於下界所以安全」）：
  * 原則是**盡量保留一般規則的 80 mm** —— 每組取其幾何允許的最大值，
    而不是取一個剛好能過的小值。
  * 一般配對：小於等於「有效距離項 − 5 mm」的最大 5 mm 整數倍。
    5 mm 是留給**未核對的位姿**與**模型差異**的工程預留，不是證明。
  * 兩指—面板沿用已裁定的 10 mm。
  * 夾爪殼—橫桿改用 1 mm 粒度（有效距離項只有約 7.2 mm），
    留 4 mm 以上。**填值是提案，不代表已認證。**
"""
from __future__ import annotations
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_link_distance import (                        # noqa
    pair_distance_bound, parse_obstacles)
from ammr_wholebody_mpc.arm_link_geometry import (                        # noqa
    link_collision_tris, obstacle_distances, sample_links_certified)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics    # noqa
from ammr_wholebody_mpc.wholebody_safety_filter import (                   # noqa
    DetectionPoint, SafetyConfig, STATUS_OK, VOBS_STATIC, _rows_from_points)
from coman_zero_cmd_feasible import hold_state                             # noqa

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
ALL_PHASES = ['approach', 'engage', 'pull', 'hold', 'release', 'retreat']
G2_PAIRS = {('uflite_gripper_link', 'handle_bar')}
# 已裁定者直接沿用；其餘由規則產生
# **R1 已核准的十組固定不動**（不重新計算，避免動到核准歷史）
import yaml as _yaml, os as _os
_R1 = _yaml.safe_load(open(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    'results/specs/coman_r1_pair_gaps_proposal.yaml'), encoding='utf-8'))
FIXED = {tuple(p['pair'].split('|')): float(p['g_pair_m']) for p in _R1['pairs']}
# R1.1 增補的五組（O3）：**不是**因為遇到失敗才降門檻，而是補上
# 「先前根本沒有列、因此從未被檢查過」的配對。選值用**同一條規則**。
EXTRA = [('uflite_gripper_link', 'drawer_front_panel'),
         ('uflite_gripper_link', 'handle_post_l'),
         ('uflite_gripper_link', 'handle_post_r'),
         ('link6', 'handle_post_l'),
         ('link6', 'handle_post_r')]
PAIRS = [('uflite_gripper_link', 'handle_bar'),
         ('uflite_finger1', 'drawer_front_panel'),
         ('uflite_finger2', 'drawer_front_panel'),
         ('link6', 'handle_bar'),
         ('uflite_finger1', 'handle_post_l'),
         ('uflite_finger1', 'handle_post_r'),
         ('uflite_finger2', 'handle_post_l'),
         ('uflite_finger2', 'handle_post_r'),
         ('link5', 'handle_bar'),
         ('link4', 'handle_bar')] + EXTRA
# 同類配對取同一值：四組指—支柱與 link6—橫桿共用最緊者決定的值
UNIFORM = {('link6', 'handle_bar'), ('uflite_finger1', 'handle_post_l'),
           ('uflite_finger1', 'handle_post_r'),
           ('uflite_finger2', 'handle_post_l'),
           ('uflite_finger2', 'handle_post_r')}


def candidate(d_eff, granularity, reserve):
    """小於等於 (d_eff − reserve) 的最大 granularity 整數倍，下限 0。"""
    v = np.floor(max(0.0, d_eff - reserve) / granularity) * granularity
    return float(round(v, 4))


def main() -> int:
    st = hold_state()
    K = WholeBodyKinematics.from_urdf_file(URDF)
    q = np.zeros(len(K.dof_names))
    q[0], q[1], q[2] = st['park']
    q[3:9] = st['q_arm']
    xml = open(URDF, encoding='utf-8').read()
    sp = subprocess.run([sys.executable, os.path.join(HERE,
                         'coman_obstacle_specs.py'), str(st['unit'][0]),
                         str(st['unit'][1])], capture_output=True, text=True,
                        check=True)
    obs = parse_obstacles([x for x in sp.stdout.split() if x.strip()])
    T_db = np.eye(4)
    T_db[0, 3] = st['unit'][0]
    T_db[1, 3] = st['drawer_y0'] - st['opening']
    for o in obs:
        if o.model == 'drawer_body':
            o.T_world_link = T_db
    names = [o.name for o in obs]
    byname = {o.name: o for o in obs}
    S = sample_links_certified(xml, rho_target=0.015, tol=0.001)
    tris_cache = link_collision_tris(xml, links=[lk for lk, _ in G2_PAIRS])

    rows = {}
    for link, ob in PAIRS:
        Tw = K.fk(q, link)
        W = (S[link].points @ Tw[:3, :3].T) + Tw[:3, 3]
        d_all, v_all, _ = obstacle_distances(W, [byname[ob]])
        k = int(np.argmin(d_all))
        n_hat = v_all[k] / max(abs(float(d_all[k])), 1e-9)
        rec = dict(link=link, obs=ob, d_sample=float(d_all[k]),
                   rho=float(S[link].rho), n=n_hat, off=S[link].points[k],
                   p=W[k])
        if (link, ob) in G2_PAIRS:
            tw = (tris_cache[link].reshape(-1, 3) @ Tw[:3, :3].T
                  + Tw[:3, 3]).reshape(-1, 3, 3)
            b = pair_distance_bound(tw, byname[ob], tol=5.0e-4,
                                    max_tris=60000, budget_s=0.020)
            rec.update(method='G2', d_lb=float(b['lb']), ub=float(b['ub']),
                       elapsed_s=float(b['elapsed_s']))
            rec['d_eff'] = float(b['lb'])
        else:
            rec.update(method='G1', d_lb=None)
            rec['d_eff'] = rec['d_sample'] - rec['rho']
        rows[(link, ob)] = rec

    # ---- 候選值 ----
    uni = min(rows[p]['d_eff'] for p in UNIFORM)
    for p_, rec in rows.items():
        if p_ in FIXED:
            rec['g'] = FIXED[p_]
            rec['basis'] = 'R1 已核准值，**原樣沿用，不重新計算**'
            rec['from_r1'] = True
        elif p_ in G2_PAIRS:   # 已被 FIXED 覆蓋，保留分支以防設定變動
            rec['g'] = candidate(rec['d_eff'], 0.001, 0.004)
            rec['basis'] = ('1 mm 粒度、留 4 mm 以上；有效距離項只有約 '
                            f'{rec["d_eff"]*1000:.2f} mm，是本表最緊的一組')
        elif p_ in UNIFORM:
            rec['g'] = candidate(uni, 0.005, 0.005)
            rec['basis'] = ('五組同類配對（手臂／手指對橫桿與支柱）取**同一值**，'
                            f'由最緊者 {uni*1000:.2f} mm 決定，留 5 mm 以上')
        else:
            rec['g'] = candidate(rec['d_eff'], 0.005, 0.005)
            rec['basis'] = ('**R1.1 增補**：同一條規則（≤ 有效距離項 − 5 mm 的最大 '
                            '5 mm 整數倍）；5 mm 為參考位姿的可行性預留。'
                            '盡量保留一般規則的 80 mm，不是取剛好能過的小值')

    # ---- 用完整屏障算零相對速度餘量 ----
    cfg = SafetyConfig(g_by_pair={f'{lk}|{ob}': rows[(lk, ob)]['g']
                                 for lk, ob in PAIRS})
    cfg.phase = None            # 不啟用接觸例外，十組都要有列
    for (link, ob), rec in rows.items():
        pt = DetectionPoint(link, rec['p'], rec['n'], rec['d_sample'],
                            STATUS_OK, 0.0, False, offset=rec['off'],
                            rho=rec['rho'], obs=ob, v_obs=np.zeros(3),
                            v_obs_state=VOBS_STATIC, d_lb=rec['d_lb'])
        A, b, _, _ = _rows_from_points(K, q, [pt], cfg, np.zeros(len(q)))
        rec['rhs'] = float(b[0])
        rec['margin_mm'] = float(b[0]) / cfg.alpha * 1000.0

    print(f'{"配對":42s}{"法":>4s}{"有效距離":>10s}{"候選g":>9s}{"餘量":>9s}')
    print(f'{"":42s}{"":>4s}{"(mm)":>10s}{"(mm)":>9s}{"(mm)":>9s}')
    nbad = 0
    for lk, ob in PAIRS:
        r = rows[(lk, ob)]
        nbad += r['margin_mm'] < 0
        print(f'{lk+"|"+ob:42s}{r["method"]:>4s}{r["d_eff"]*1000:10.2f}'
              f'{r["g"]*1000:9.1f}{r["margin_mm"]:+9.2f}'
              f'{"  **負**" if r["margin_mm"] < 0 else ""}')
    print(f'\n零相對速度下餘量為負的配對：{nbad} 組')
    json.dump({f'{lk}|{ob}': {k: (v if not isinstance(v, np.ndarray)
                                  else [float(x) for x in v])
                              for k, v in rows[(lk, ob)].items()}
               for lk, ob in PAIRS},
              open(os.path.join(HERE, 'results', 'coman_r1_table.json'), 'w'),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
