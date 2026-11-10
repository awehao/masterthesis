#!/usr/bin/env python3
"""(d) 量測後的站位（車道橫向距離）選擇（Codex 20261111_042529／050821／053311 (d)；Howard 11/11：Y_REACH 0.4275 是固定模板）。純函式；計畫工作者 kind＝'standoff' 與離線共用。
時機：導引候選（只有 xy；底盤相機看不到支撐）時決定車道 ⇒ 支撐高度未知，候選須在區域桌高範圍 Z_LIST 每一個都可行（穩健）；
  之後手腕量到支撐，抓取計畫（plan_grasp）以實際高度沿選定車道再核一次（不可行照基線受控停止）。
候選：y_reach ∈ Y_LIST（車道 y＝y_obj − y_reach；物件在左側——右側車道／抓取姿態待 (d) 右側接線，未實作）；窗相對起點 WIN_LO（沿用；−0.25 等較晚窗在離線評估多不可行）。
每候選、每 z：plan_worker.window_seeded（預抓／抓取／抬升三位姿沿窗延拓；種子依序試、第一個可行分支即採用並優先用於下一個 z；某 z 不可行即停該候選）⇒ 可行與最小關節餘裕。
另算底盤圓盤（R_BASE 0.39）沿窗段對已知桌腳方盒（區域先驗桌）的最小間隙；車道須在放行側向管 y_tube（bottle_s0b_ctrl.Y_TUBE，已驗證範圍）內。
選擇：全 z 可行者中，最小關節餘裕（取各 z 最小）最大 → 底盤對桌腳間隙最大；到抓取時間由底盤剖面決定、各候選相同（standoff_select_check：皆 9.38 s），不列入。
"""
import numpy as np

Y_LIST = (0.36, 0.40, 0.4275, 0.46, 0.50)
Z_LIST = (0.55, 0.60, 0.65)
WIN_LO, WIN_W = -0.35, 0.50
R_BASE = 0.39


def evaluate(inp):
    """inp：xy（區域）、band、pre_m、lift_m、q_seeds、（可選）y_list、z_list、legs（[(中心, 尺寸)]）、y_tube。回 (ok, why, res)。"""
    import bottle_plan_worker as PW
    import bottle_ref as BRF
    import bottle_s0_ctrl as C0
    x, y = float(inp['xy'][0]), float(inp['xy'][1])
    ylist = tuple(inp.get('y_list') or Y_LIST)
    zlist = tuple(inp.get('z_list') or Z_LIST)
    seeds = [np.asarray(s, float) for s in inp['q_seeds']]
    legs = [(np.asarray(c, float), np.asarray(sz, float)) for c, sz in (inp.get('legs') or [])]
    rows = []
    tube = inp.get('y_tube')
    for yr in ylist:
        yl = y - yr
        if tube is not None and not (float(tube[0]) <= yl <= float(tube[1])):
            rows.append(dict(y_reach=yr, y_lane=round(yl, 4), ok=False, why='outside_release_tube', margin_min=None, margin_by_z=None, base_leg_clear=None))
            continue                                                         # 放行 INSERT 的側向管（bottle_s0b_ctrl.Y_TUBE，已驗證範圍）外不選
        mins, why = [], None
        for z in zlist:
            T_g = C0.grasp_target(x, y, z + float(inp.get('band', 0.10)))
            T_p = T_g.copy()
            T_p[:3, 3] -= float(inp.get('pre_m', 0.06)) * T_g[:3, 2]
            T_l = BRF.lift_pose(T_g, float(inp.get('lift_m', 0.05)))
            best = None
            for sd in seeds:                                                 # 種子依序（當下手臂、方位、亂數）：第一個可行分支即採用（同 plan_grasp 的分支族）
                b, w = PW.window_seeded(T_p, T_g, T_l, x + WIN_LO, x + WIN_LO + WIN_W, sd, y=yl)
                if b is not None and np.isfinite(b['margin']) and b['margin'] > 0.0:
                    best = b['margin']
                    seeds = [sd] + [q for q in seeds if q is not sd]            # 下一個 z 先試同一分支
                    break
                if why is None:
                    why = w
            if best is None:
                mins = None
                why = why or 'no_branch'
                break
            mins.append(best)
        clr = None
        if legs:
            xs = np.linspace(x + WIN_LO, x + WIN_LO + WIN_W, 11)
            d = [float(np.linalg.norm(np.maximum(np.abs(np.array([xx, yl]) - c[:2]) - sz[:2] / 2, 0.0))) for xx in xs for c, sz in legs]
            clr = round(min(d) - R_BASE, 4)
        ok = mins is not None and (clr is None or clr >= 0.02)
        rows.append(dict(y_reach=yr, y_lane=round(yl, 4), ok=bool(ok), why=None if ok else (why or 'base_leg'), margin_min=None if mins is None else round(min(mins), 4),
                         margin_by_z=None if mins is None else [round(m, 4) for m in mins], base_leg_clear=clr))
    feas = [r for r in rows if r['ok']]
    if not feas:
        return False, 'no_feasible_standoff', dict(rows=rows)
    best = max(feas, key=lambda r: (r['margin_min'], r['base_leg_clear'] if r['base_leg_clear'] is not None else 0.0))
    return True, None, dict(rows=rows, chosen=best, rule='全 z 可行 → 最小關節餘裕最大 → 底盤對桌腳間隙最大', z_list=list(zlist), win=[WIN_LO, WIN_LO + WIN_W])
