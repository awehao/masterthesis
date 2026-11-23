#!/usr/bin/env python3
"""(d) 量測後站位選擇的離線評估（Codex 20261111_042529／050821 (d)；Howard 11/11：Y_REACH 0.4275／RUNUP 0.55 是量測平移的固定模板）。純離線。
同一物件（tpa_nom_3 量測瓶位）、同一承接狀態（實錄 8.80 s 計畫承接：U 中、手臂 0.93 rad/s；底盤 0.35 m/s），比較站位候選：
  車道橫向距離 y_reach ∈ {0.36, 0.40, 0.4275, 0.46, 0.50} m（車道 y＝y_obj − y_reach）× 窗相對起點 win_lo ∈ {−0.35, −0.25} m。
每候選：bottle_plan_worker.plan_grasp（窗內三位姿 IK 分支連續、最小關節餘裕）→ bottle_w1_tpa_ctrl.plan_tpa（從當下狀態的整段軌跡時長）→
  沿軌跡（車道直線）核手臂對桌、自碰撞、全部連桿對瓶（含開爪手指）、底盤對桌腳；報可行與否、T、到抓取時間、各餘裕。
選擇規則（提案，未接控制器）：可行者中取「到抓取時間最短」（0.1 s 內視為同），再取窗內關節餘裕最大、底盤對桌腳最遠；固定模板（0.4275／−0.35）列為對照。
    python3 evaluation/standoff_select_check.py OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_plan_worker as PW  # noqa: E402
import bottle_r0_approach as R0  # noqa: E402
import bottle_ref as BR  # noqa: E402
import bottle_s0_ctrl as C0  # noqa: E402
import bottle_w1_tpa_ctrl as TPA  # noqa: E402
import tpa_offline_check as TO  # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L  # noqa: E402

LAT = C0.D_STEPS * C0.PHYS


def main():
    out = sys.argv[1]
    r, c, O, fr, tl = TO.load()
    xb, yb = fr['xy']
    zs = fr['z_support']
    T_g = C0.grasp_target(xb, yb, zs + 0.10)
    T_pre = BR.offset_along_tool(T_g, -0.06)
    q0, qd0, base0 = TO.rec_state(c, O, 8.80)
    boxes = R0.table_boxes(zs)
    rows = []
    for y_reach in (0.36, 0.40, 0.4275, 0.46, 0.50):
        for win_lo in (-0.35, -0.25):
            yl = yb - y_reach
            x_pre = xb + win_lo + 0.05
            x_slow = x_pre - 0.6
            row = dict(y_reach=y_reach, win_lo=win_lo, template=bool(abs(y_reach - 0.4275) < 1e-9 and abs(win_lo + 0.35) < 1e-9))
            inp = dict(xy=[xb, yb], z_support=zs, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[win_lo, win_lo + 0.5], x_nom=0.0, y_lane=yl,
                       q_seeds=[s.tolist() for s in [q0] + list(R0.RNG_SEEDS)], q_now=q0.tolist(), q_seed=None)
            ok, why, res = PW.plan_grasp(inp)
            if not ok:
                rows.append(dict(row, ok=False, why='plan_' + str(why)))
                print(rows[-1], flush=True)
                continue
            sig_e = float(base0[0]) + 0.35 * LAT
            Rr, why = TPA.plan_tpa(R0.K, q0, qd0, np.array([x_pre, yl, 0.0]), T_pre, np.asarray(res['q_pre_lo'], float), sig_e, x_pre, x_slow, 0.35, 0.10, 0.35, R0.IDX, L.lower, L.upper)
            if Rr is None:
                rows.append(dict(row, ok=False, why='tpa_' + str(why), plan_margin=round(res['margin'], 4)))
                print(rows[-1], flush=True)
                continue
            P, T = Rr['P'], Rr['P']['T']
            prof = []
            x, v = sig_e, 0.35
            for k in range(int(T / 0.05) + 1):
                prof.append(x)
                vt = 0.35 if x < x_slow else 0.10
                v = min(vt, v + 0.4 * 0.05) if v < vt else max(vt, v - 0.4 * 0.05)
                x = min(x + v * 0.05, x_pre)
            W = dict(env=np.inf, self=np.inf, base=np.inf, bottle=np.inf)
            for k in range(0, len(prof), 3):
                q9 = np.r_[prof[k], yl, 0.0, TPA.traj_eval(P, k * 0.05)]
                e, sc, bc, bt = TO.collide(q9, boxes, xb, yb, zs)
                W['env'], W['self'], W['base'], W['bottle'] = min(W['env'], e[0]), min(W['self'], sc[0]), min(W['base'], bc), min(W['bottle'], bt[0])
            okc = W['env'] >= R0.ENV_M and W['self'] >= R0.SELF_M and W['base'] >= R0.BASE_M and W['bottle'] >= 0.005
            row.update(ok=bool(okc), plan_margin=round(res['margin'], 4), T=round(T, 2), t_to_grasp=round(T + LAT + 0.05 + 1.0, 2), v_ratio=Rr['info']['v_ratio'],
                       env_min=round(W['env'], 4), self_min=round(W['self'], 4), base_leg_min=round(W['base'], 4), bottle_min=round(W['bottle'], 4),
                       q1_end=round(float(Rr['q1'][0]), 3), why=None if okc else 'clearance')
            rows.append(row)
            print({k: row[k] for k in ('y_reach', 'win_lo', 'template', 'ok', 'plan_margin', 'T', 't_to_grasp', 'env_min', 'self_min', 'base_leg_min', 'bottle_min')}, flush=True)
    feas = [r_ for r_ in rows if r_.get('ok')]
    best = min(feas, key=lambda r_: (round(r_['t_to_grasp'], 1), -r_['plan_margin'], -r_['base_leg_min'])) if feas else None   # 時間最短（0.1 s 內視為同）→ 窗內關節餘裕最大 → 底盤對桌腳最遠
    tmpl = next((r_ for r_ in rows if r_['template']), None)
    json.dump(dict(run=TO.RUN, object=dict(x=xb, y=yb, z=zs), rows=rows, chosen=best, template=tmpl,
                   note='同一物件與承接狀態；底盤沿各候選車道直線、名目剖面；選擇規則為提案，未接控制器'),
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print('選擇', None if best is None else {k: best[k] for k in ('y_reach', 'win_lo', 't_to_grasp', 'plan_margin')}, '模板', None if tmpl is None else {k: tmpl.get(k) for k in ('ok', 't_to_grasp', 'plan_margin')})


if __name__ == '__main__':
    main()
