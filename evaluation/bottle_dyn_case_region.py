#!/usr/bin/env python3
"""R-B1 區域任務的迎面錯身移動物案例（Codex 20261102_200625；生成方式同 bottle_w1_dyn_case.CASE〔w1_dyn_pass1〕，事前固定規則，不依成敗調整）。
全部用世界座標：障礙真值只供 Isaac 場景驅動、評估與安全核對，不送進控制器（控制器只經掃描→追蹤器看到它；追蹤器輸出由 dynsr 的 obs_to_region 轉區域）。
生成規則（只讀名目趟 bottle_rb1_gaze_1 的量測軌跡；carrygov 基線在 R-B1 不觸發持守，持瓶段與名目相同——假世界 48.4 s 相同）：
  交會時刻 t_meet＝持瓶導航段（place_phase CARRY_NAV → PLACE_APPROACH）量測底盤軌跡弧長分率 f 處的時刻；P_m＝當時底盤位置，d＝前後 0.5 s 位移方向（單位向量）。
  f 的選法（事前宣告、只依下列幾何前提，不依閉迴路成敗）：候選 f＝0.5、0.4、0.6、0.3、0.7、0.2、0.8、0.1、0.9（離中點近者優先），取第一個四項前提全過者。
  （第一版只用中點 f＝0.5：R-B1 持瓶路徑先下行再轉東，中點的直線錯身線離第一桌桌面 0.031 m、離規劃路徑 0.563 m，前提 1、2 不過；0.4–0.6 同樣不過，0.7 起全過。）
  障礙：尺寸、速度沿用 arena 交通 dyn_obs_0（0.30 m/s、半徑 0.25 m、高 1.0 m）；在行進方向右側法向 0.80 m 處沿 −d 反向等速（迎面錯身，不橫越規劃路徑）；
  起點 S＝P_m＋2.2 d＋0.80 n_右、終點 E＝P_m − 2.5 d＋0.80 n_右；t_start＝t_meet − 2.2／0.30（任務時刻；之前靜止在 S，到 E 後靜止）。
幾何前提（同 test_bottle_w1_dyn.geometry，換成 R-B1 世界位置）：
  1 障礙軌跡離靜態物（第一桌桌面與桌腳〔區域平移後〕、第二桌、原障礙方塊）≥ 半徑＋0.05；
  2 障礙中心離持瓶規劃路徑（名目趟 place.carry_path，區域→世界）≥ 0.74 m（不進代價圖內切格 ⇒ 不觸發 path_blocked）；
  3 最接近時刻落在名目趟 CARRY_NAV 區間，底盤中心到障礙表面 < 0.69（進入動態 CBF 影響範圍）；
  4 不反應時連桿點雲與持瓶到障礙表面 > 0。
    python3 evaluation/bottle_dyn_case_region.py      （印出案例與前提結果，寫 experiments/20261102_rb1_dyn_case.json）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL = 'bottle_rb1_gaze_1'
OUT = os.path.join(HERE, 'experiments', '20261102_rb1_dyn_case.json')
SPEED, RADIUS, HEIGHT, LATERAL, AHEAD, BEHIND = 0.30, 0.25, 1.0, 0.80, 2.2, 2.5


F_ORDER = (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9)


def generate(frac=0.5):
    D = os.path.join(HERE, 'runs', NOMINAL)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t, x, y = c['t'], c['base_x'], c['base_y']
    lo = next(e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'CARRY_NAV')
    hi = next(e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'PLACE_APPROACH')
    m = (t >= lo) & (t <= hi)
    idx = np.flatnonzero(m)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x[idx]), np.diff(y[idx])))]
    k = idx[int(np.argmin(np.abs(s - s[-1] * frac)))]
    t_meet = float(t[k])
    a, b = int(np.argmin(np.abs(t - (t_meet - 0.5)))), int(np.argmin(np.abs(t - (t_meet + 0.5))))
    d = np.array([x[b] - x[a], y[b] - y[a]])
    d /= np.linalg.norm(d)
    n_r = np.array([d[1], -d[0]])
    P = np.array([x[k], y[k]])
    S, E = P + AHEAD * d + LATERAL * n_r, P - BEHIND * d + LATERAL * n_r
    return dict(name='rb1_dyn_pass1', source='arena_traffic dyn_obs_0（同 w1_dyn_pass1 生成法）', nominal=NOMINAL, speed=SPEED, radius=RADIUS, height=HEIGHT,
                start=tuple(round(float(v), 4) for v in S), end=tuple(round(float(v), 4) for v in E), t_start=round(t_meet - AHEAD / SPEED, 3), t_meet=round(t_meet, 3),
                lateral=LATERAL, meet_robot=tuple(round(float(v), 4) for v in P), robot_dir=tuple(round(float(v), 4) for v in d), carry_nav=[lo, hi], frame='world',
                frac=frac, rule='t_meet＝持瓶導航段量測軌跡弧長分率 frac 處（候選依離中點順序，取第一個幾何前提全過者）；右側法向 0.80 m 反向等速；起點前 2.2 m、終點後 2.5 m')


def cases():
    """已產生並提交的案例（experiments/20261102_rb1_dyn_case.json）：{名稱: 案例 dict（世界座標）}，供 Isaac 腳本登錄到 bottle_w1_dyn_case.CASES。"""
    j = json.load(open(OUT))
    ca = dict(j['case'])
    for k in ('start', 'end', 'meet_robot', 'robot_dir'):
        ca[k] = tuple(ca[k])
    return {ca['name']: ca}


def geometry(CA):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
    import bottle_s1_geom as G
    import bottle_w1_dyn_case as DC
    D = os.path.join(HERE, 'runs', NOMINAL)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    O = np.array(r['region']['offset'], float)
    z1 = float(r['table1_z'])
    t1 = [('t1:' + n, c_ + np.r_[O, 0.0], s_) for n, c_, s_ in G.table_boxes(dict(G.S1_TABLE, top_z=z1))]
    t2 = [('t2:' + n, c_ + np.array([2.4, -1.0, 0.0]), s_) for n, c_, s_ in G.table_boxes(dict(G.S1_TABLE))]
    box = [('box', np.array([-2.0, 0.0, 0.25]), np.array([0.40, 0.40, 0.50]))]
    S, E = np.array(CA['start']), np.array(CA['end'])
    line = S + np.linspace(0, 1, 2000)[:, None] * (E - S)

    def seg_box(cc, sz):
        dx = np.maximum(np.abs(line[:, 0] - cc[0]) - sz[0] / 2, 0)
        dy = np.maximum(np.abs(line[:, 1] - cc[1]) - sz[1] / 2, 0)
        return float(np.hypot(dx, dy).min())
    dmin = {n: round(seg_box(cc, sz) - CA['radius'], 4) for n, cc, sz in t1 + t2 + box}
    worst = min(dmin, key=dmin.get)
    cp = np.array(r['place']['carry_path'], float)[:, :2] + O
    dpath = float(np.min(np.linalg.norm(cp[:, None, :] - line[None, ::4, :], axis=2)))
    thr = CA['radius'] + 0.39 + 0.05 + 0.05
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']
    lo, hi = CA['carry_nav']
    ob = np.array([DC.state(tt, CA)[0] for tt in t])
    db = np.hypot(c['base_x'] - ob[:, 0], c['base_y'] - ob[:, 1]) - CA['radius']
    i = int(np.argmin(db))
    BP = G.bottle_points()
    best = (9.9, None)
    for k in range(0, len(t), 5):
        if abs(t[k] - t[i]) > 4.0:
            continue
        q = np.array([c[n_][k] for n_ in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        pts = list(G.link_world(q, float(c['finger1_q'][k])).values())
        w, xq, yq, zq = (c[n_][k] for n_ in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
        Rb = np.array([[1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)], [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)],
                       [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)]])
        pts.append((Rb @ BP.T).T + np.array([c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]))
        P = np.vstack(pts)
        P = P[P[:, 2] <= CA['height']]
        dd = float((np.hypot(P[:, 0] - ob[k, 0], P[:, 1] - ob[k, 1]) - CA['radius']).min())
        if dd < best[0]:
            best = (round(dd, 4), round(float(t[k]), 2))
    checks = {'1_static': (dmin[worst] >= 0.05, dict(worst=worst, min=dmin[worst], all=dmin)),
              '2_path': (dpath >= thr, dict(min=round(dpath, 4), thr=thr)),
              '3_closest_in_carry': (lo <= t[i] <= hi and db[i] < 0.69, dict(t=round(float(t[i]), 2), base_surface=round(float(db[i]), 4))),
              '4_no_reaction_body': (best[0] > 0.0, dict(min=best[0], t=best[1]))}
    return {k: dict(ok=bool(v[0]), **v[1]) for k, v in checks.items()}


def main():
    tried = []
    for f in F_ORDER:
        CA = generate(f)
        geo = geometry(CA)
        ok = all(v['ok'] for v in geo.values())
        tried.append(dict(frac=f, ok=ok, failed=[k for k, v in geo.items() if not v['ok']]))
        if ok:
            break
    out = dict(case=CA, geometry=geo, all_ok=ok, tried=tried, generated_before_runs=True, ruling='reviews/20261102_200625_reply.md')
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('嘗試', tried)
    print(json.dumps(CA, ensure_ascii=False))
    for k, v in geo.items():
        print(('✓' if v['ok'] else '✗'), k, {kk: vv for kk, vv in v.items() if kk not in ('ok', 'all')})
    return 0 if out['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
