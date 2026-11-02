#!/usr/bin/env python3
"""持瓶路徑承接（carry_join_lateral）偏差定位（Codex 20261102_083312；只讀既有紀錄、離線計算，不實跑、不改門檻）。
對每趟列：請求起點 → Smac 原始路徑首點（與起點格的格角／格心距離）→ 重取樣＋平滑後最近點 → 承接時在途前推位置的橫向偏差（同 bottle_place_ctrl.build_carry 的算法）。
地圖：原點 (−5.0, −3.0)、解析度 0.05（runs/*/map.yaml）。costmap 慣例格心＝原點＋(i＋0.5)·res；紀錄中原始路徑首點／末點恰為起點格／目標格的格角（原點＋i·res）。
另列「若把原始路徑換成格心慣例（x、y 各＋res/2）」時同一算法的承接偏差，作為最小修正設計的離線證據（不是已採用的修正）。
承接時的在途前推位置未逐拍記錄：以承接時刻實測底盤 xy 加在途時間（D_STEPS×PHYS）×實測車速沿 +x 估計，另報與紀錄承接偏差的差。
理論：起點在格內均勻分布時，格角慣例的首點距離最大 res·√2＝70.7 mm、超過 50 mm 的機率 1−π/4＝21.5%；格心慣例最大 res·√2/2＝35.4 mm。
    python3 evaluation/bottle_carry_join_offset.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_n2_path as NP                             # noqa: E402
import bottle_place_ctrl as PC                          # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402

RUNS = ['bottle_online3_u3_1', 'bottle_online_u3_2', 'bottle_online4_u3_1']
OUT = os.path.join(HERE, 'results', 'bottle', 'carry_join_offset.json')


def lateral(raw, p_pred, reg):
    raw = PC.smooth_path(NP.resample(np.asarray(raw, float)), reg['smooth_iters'])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    d = np.linalg.norm(raw - p_pred, axis=1)
    k = int(np.argmin(d[s <= 1.0])) if (s <= 1.0).any() else int(np.argmin(d))
    return float(d[k]), raw[k].tolist(), raw[0].tolist()


def main():
    res = []
    for run in RUNS:
        D = os.path.join(HERE, 'runs', run)
        import yaml
        my = yaml.safe_load(open(os.path.join(D, 'map.yaml')))
        ox, oy, rr = float(my['origin'][0]), float(my['origin'][1]), float(my['resolution'])
        r = json.load(open(os.path.join(D, 's1_report.json')))
        p = r['place']
        cr, pl = p['carry_req'], p['carry']['plan']
        raw = np.array(pl['raw'], float)
        st = np.array(cr['start'][:2])
        i, j = math.floor((st[0] - ox) / rr), math.floor((st[1] - oy) / rr)
        corner = np.array([ox + i * rr, oy + j * rr])
        center = corner + rr / 2
        reg = dict(PC.REG)
        cn = [e for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'CARRY_NAV']
        t_join = cn[0]['t'] if cn else (r['stop'] or {}).get('t')
        lat_log = cn[0]['lateral'] if cn else (r['stop'] or {}).get('lateral')
        Z = np.load(os.path.join(D, 's1_steps.npz'))
        c = {str(k): Z['rows'][:, ii] for ii, k in enumerate(Z['cols'])}
        t = c['t']
        k = int(np.argmin(np.abs(t - t_join)))
        v = float(np.hypot(c['base_x'][k] - c['base_x'][k - 5], c['base_y'][k] - c['base_y'][k - 5]) / (t[k] - t[k - 5]))
        p_meas = np.array([c['base_x'][k], c['base_y'][k]])
        p_est = p_meas + np.array([v * C0.D_STEPS * C0.PHYS, 0.0])
        lat_c, nr_c, s0_c = lateral(raw, p_est, reg)
        lat_m, nr_m, s0_m = lateral(raw + rr / 2, p_est, reg)
        o = dict(run=run, map=dict(origin=[ox, oy], res=rr), request_start=np.round(st, 4).tolist(), start_cell=[i, j], raw_first=raw[0].tolist(), raw_last=raw[-1].tolist(), goal=np.round(cr['goal'], 4).tolist(),
                 start_to_raw_first_mm=round(float(np.linalg.norm(raw[0] - st)) * 1e3, 1), start_to_cell_corner_mm=round(float(np.linalg.norm(corner - st)) * 1e3, 1),
                 start_to_cell_center_mm=round(float(np.linalg.norm(center - st)) * 1e3, 1), raw_first_is_corner=bool(np.allclose(raw[0], corner, atol=1e-6)),
                 goal_to_raw_last_mm=round(float(np.linalg.norm(raw[-1] - np.array(cr['goal']))) * 1e3, 1),
                 t_join=t_join, v_meas_join=round(v, 4), p_meas_join=np.round(p_meas, 4).tolist(), p_pred_est=np.round(p_est, 4).tolist(),
                 smoothed_first=np.round(s0_c, 4).tolist(), lateral_logged_mm=None if lat_log is None else round(lat_log * 1e3, 1), lateral_recomputed_mm=round(lat_c * 1e3, 1),
                 recompute_minus_logged_mm=None if lat_log is None else round((lat_c - lat_log) * 1e3, 1),
                 lateral_if_center_convention_mm=round(lat_m * 1e3, 1), join_lat_mm=reg['join_lat'] * 1e3)
        res.append(o)
        print(json.dumps(o, ensure_ascii=False))
    # 理論分布（起點在格內均勻）
    g = np.linspace(0, 0.05, 501)
    X, Y = np.meshgrid(g, g)
    dc = np.hypot(X, Y)
    dm = np.hypot(X - 0.025, Y - 0.025)
    theory = dict(corner=dict(max_mm=round(float(dc.max()) * 1e3, 1), p_gt_50mm=round(float((dc > 0.05).mean()), 3)), center=dict(max_mm=round(float(dm.max()) * 1e3, 1), p_gt_50mm=round(float((dm > 0.05).mean()), 3)))
    json.dump(dict(runs=res, theory_uniform_start_in_cell=theory, note='重算以實測位置＋在途時間×實測車速估在途前推位置；與紀錄承接偏差的差照列'), open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('理論', theory)
    return 0


if __name__ == '__main__':
    sys.exit(main())
