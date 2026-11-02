#!/usr/bin/env python3
"""在線視向候選的離線重放核對（Codex 20261102_092601；只讀既有紀錄，不跑 Isaac）。
兩條既有接近軌跡：原區域 U3（bottle_online5_u3_1）、R-B1（bottle_rb1_1）。每控制拍（0.05 s）從 U 開始到 DONE 後 0.5 s：
  底盤＝實錄量測位姿（區域座標）；序列參考＝依實錄 U／H／R 換段時刻與時長重建（q_tuck→V_HIGH→備妥；備妥取實錄 DONE 時量測手臂，誤差 ≤ ARM_TOL）；
  視向＝bottle_w1_gaze_ctrl 的同一組函式（gaze_target、track，LEAD 前瞻、U／H 段才作用、R 段歸零）。
開迴路近似：底盤軌跡取實錄、手臂取參考（不含追蹤落後）；閉迴路效應另由 test_bottle_w1_gaze 的假世界核對。
指標（評估用，真值只在這裡用）：
  覆蓋：實際瓶身取樣（16×6）與支撐環（半徑 0.03／0.06／0.09 m 桌面點）各 ≥ 50% 在畫面內；
        全區＝本批設計域 8 個角落佈置（x ±0.08、y 0.395／0.46、桌高 0.55／0.65）同時成立；窗＝U 開始到實錄承接、U 到 R、H 段。
  實錄量測手臂的覆蓋與 w1_log 的 expected 旗標逐筆比對（相機模型核對）。
  幾何：關節餘裕（LITE6_SAFE）≥ 0.05、自碰（開爪）≥ 8.017 mm、手臂對第一桌（該趟桌高）、第二桌、原障礙方塊、瓶 ≥ 0.05 m；
  視向：|g|、|ġ|、|g̈| 最大值、R 段開始時 g、歸零時刻對實錄 DONE。
    python3 evaluation/bottle_gaze_offline.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_branch as BB                              # noqa: E402
import bottle_ref as BR                                 # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_gaze_ctrl as GZ                        # noqa: E402
import bottle_w1_segtarget_vhigh_ctrl as VH             # noqa: E402

RUNS = dict(orig_u3=dict(run='bottle_online5_u3_1', table2=(2.4, -1.0)), rb1=dict(run='bottle_rb1_1', table2=(2.4, -1.0)))
OUT = os.path.join(HERE, 'results', 'bottle', 'gaze_offline.json')
Q_TUCK = np.array([0.0, 0.0, 0.035, 0.0, -1.5708, 0.0])
OBST_W = (np.array([-2.0, 0.0, 0.25]), np.array([0.40, 0.40, 0.50]))      # 原障礙方塊（世界）
SELF_FLOOR, CLEAR, MARGIN = 0.008017, 0.05, 0.05
K = G.C.K
R_, H_ = G.BOTTLE['r'], G.BOTTLE['h']
ANG = np.linspace(0, 2 * math.pi, 16, endpoint=False)
CORNERS = [((x, y), z) for x in GZ.AREA['x'] for y in GZ.AREA['y'] for z in GZ.AREA['z1']]


def body(xy, z1):
    return np.array([[xy[0] + R_ * math.cos(a), xy[1] + R_ * math.sin(a), z1 + z] for z in np.linspace(0, H_, 6) for a in ANG])


def ring(xy, z1):
    return np.array([[xy[0] + rr * math.cos(a), xy[1] + rr * math.sin(a), z1] for rr in (0.03, 0.06, 0.09) for a in ANG])


def inview(Tc, P):
    Ti = np.linalg.inv(Tc)
    Pc = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]
    ok = Pc[:, 2] > 0.05
    u = GZ.FX * Pc[:, 0] / np.where(ok, Pc[:, 2], 1) + 320
    v = GZ.FX * Pc[:, 1] / np.where(ok, Pc[:, 2], 1) + 240
    return ok & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)


def cover(q9, bxy, bz):
    Tc = K.fk(q9, 'link_eef') @ GZ.T_MOUNT
    b, r = inview(Tc, body(bxy, bz)).mean(), inview(Tc, ring(bxy, bz)).mean()
    cs = [min(inview(Tc, body(xy, z)).mean(), inview(Tc, ring(xy, z)).mean()) for xy, z in CORNERS]
    return dict(real=bool(b >= 0.5 and r >= 0.5), body=float(b), ring=float(r), area=bool(min(cs) >= 0.5), area_min=float(min(cs)))


def boxes(z1, ro, t2):
    tb = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes(dict(G.S1_TABLE, top_z=z1))]
    t2b = [('t2:' + n, c_ + np.array([t2[0] - ro[0], t2[1] - ro[1], 0.0]), s_) for n, c_, s_ in G.table_boxes(dict(G.S1_TABLE))]
    ob = [('obst', OBST_W[0] - np.r_[ro, 0.0], OBST_W[1])]
    return tb + t2b + ob


def geom(q9, B, bxy, bz):
    W = G.link_world(q9, G.Q_OPEN)
    P = np.vstack(list(W.values()))
    env = min((float(G.box_dist(P, c_, s_).min()), n) for n, c_, s_ in B)
    bot = float(G.box_dist(P, np.array([bxy[0], bxy[1], bz + H_ / 2]), np.array([2 * R_, 2 * R_, H_])).min())
    return dict(env=env[0], env_who=env[1], bottle=bot, self=float(G.C.self_clear(q9, G.Q_OPEN)[0]), margin=float(BB.margin(q9[3:])))


def one(tag, cfg):
    D = os.path.join(HERE, 'runs', cfg['run'])
    r = json.load(open(os.path.join(D, 's1_report.json')))
    ro = np.array((r.get('region') or {}).get('offset', [0.0, 0.0]), float)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']
    vl = r['vhigh_seq']['log']
    tph = {e[1]: float(e[0]) for e in vl}
    Tu, Tr = float(vl[0][3]), float(next(e[3] for e in vl if e[1] == 'R'))
    qv = np.array(r['vhigh_seq']['q_vhigh'], float)
    kd = int(np.argmin(np.abs(t - tph['DONE'])))
    q_ready = np.array([c[f'q{i}'][kd] for i in range(1, 7)])
    acc = float(next(g for g in r['online']['grasp_plans'] if g['state'] == 'accepted')['t_acc'])
    bxy = (float(c['bottle_base_x'][0] - ro[0]), float(c['bottle_base_y'][0] - ro[1]))
    bz = float(c['bottle_base_z'][0])
    z1 = float(r.get('table1_z') or bz)
    B = boxes(z1, ro, cfg['table2'])

    def qseq(tt):
        if tt < tph['U']:
            return Q_TUCK
        if tt < tph['H']:
            return Q_TUCK + (qv - Q_TUCK) * BR.min_jerk(min((tt - tph['U']) / Tu, 1.0))
        if tt < tph['R']:
            return qv
        return qv + (q_ready - qv) * BR.min_jerk(min((tt - tph['R']) / Tr, 1.0))

    def ph_at(tt):
        return 'U' if tt < tph['H'] else ('H' if tt < tph['R'] else ('R' if tt < tph['DONE'] else 'DONE'))
    # w1_log 相機模型核對
    W = np.load(os.path.join(D, 'w1_log.npz'), allow_pickle=True)
    wc = {str(k): W['rows'][:, i] for i, k in enumerate(W['cols'])}
    agree = []
    g = v = tgt = 0.0
    rows, n_why = [], {}
    gmax = dict(g=0.0, v=0.0, a=0.0)
    g_at_R, t_zero = None, None
    for tt in np.arange(tph['U'], tph['DONE'] + 0.5, 0.05):
        k = int(np.argmin(np.abs(t - tt)))
        qb = np.array([c['base_x'][k] - ro[0], c['base_y'][k] - ro[1], c['base_yaw'][k]])
        ph = ph_at(tt)
        if ph in ('U', 'H'):
            qa_lead = qseq(tt + GZ.LEAD) if ph == 'U' else qv
            s, why, _ = GZ.gaze_target(K, qb, qa_lead, tgt)
            n_why[why] = n_why.get(why, 0) + 1
            if s is not None:
                tgt = s
        else:
            tgt = 0.0
            if g_at_R is None:
                g_at_R = g
        g1, v1 = GZ.track(g, v, tgt, 0.05)
        gmax.update(g=max(gmax['g'], abs(g1)), v=max(gmax['v'], abs(v1)), a=max(gmax['a'], abs(v1 - v) / 0.05))
        g, v = g1, v1
        if ph not in ('U', 'H') and g == 0.0 and v == 0.0 and t_zero is None:
            t_zero = round(float(tt), 3)
        qa = qseq(tt)
        q_meas = np.r_[qb, [c[f'q{i}'][k] for i in range(1, 7)]]
        cb, cg, cm = cover(np.r_[qb, qa], bxy, bz), cover(np.r_[qb, qa + g * GZ.E1], bxy, bz), cover(q_meas, bxy, bz)
        kw = int(np.argmin(np.abs(wc['task_t'] - tt)))
        if abs(wc['task_t'][kw] - tt) < 0.06:
            agree.append(bool(wc['expected'][kw] > 0.5) == bool(cm['body'] >= 0.5))
        rows.append(dict(t=round(float(tt), 3), ph=ph, x=round(float(qb[0]), 4), y=round(float(qb[1]), 4), g=round(g, 5), tgt=round(tgt, 5), base=cb, gaze=cg, meas=cm,
                         geo=geom(np.r_[qb, qa + g * GZ.E1], B, bxy, bz), geo_b=geom(np.r_[qb, qa], B, bxy, bz)))

    def frac(lo, hi, who, key):
        m = [x[who][key] for x in rows if lo - 1e-9 <= x['t'] <= hi + 1e-9]
        return round(float(np.mean(m)), 3) if m else None
    win = dict(U_to_accept=(tph['U'], acc), U_to_R=(tph['U'], tph['R']), H=(tph['H'], tph['R']))
    cov = {w: {who: dict(real=frac(lo, hi, who, 'real'), area=frac(lo, hi, who, 'area')) for who in ('base', 'gaze', 'meas')} for w, (lo, hi) in win.items()}
    worst, worst_b = {}, {}
    for x in rows:
        for gk, ww in (('geo', worst), ('geo_b', worst_b)):
            for kk, vv in x[gk].items():
                if kk == 'env_who':
                    continue
                if vv < ww.get(kk, (9.9,))[0]:
                    ww[kk] = (round(vv, 5), x['t'], x['ph'], x[gk]['env_who'] if kk == 'env' else None)
    lim = dict(env=CLEAR, bottle=CLEAR, self=SELF_FLOOR - 1e-6, margin=MARGIN)
    fails = sorted(kk for kk, vv in worst.items() if vv[0] < lim[kk])
    out_gap = []
    for who in ('base', 'gaze'):
        gaps, cur = [], None
        for x in rows:
            if x['t'] > acc + 1e-9:
                break
            if not x[who]['real'] and cur is None:
                cur = x['t']
            if x[who]['real'] and cur is not None:
                gaps.append([cur, x['t']])
                cur = None
        if cur is not None:
            gaps.append([cur, round(acc, 3)])
        out_gap.append((who, gaps))
    return dict(run=cfg['run'], region_offset=ro.tolist(), phases=tph, accept_t=acc, bottle_region=[round(bxy[0], 4), round(bxy[1], 4), round(bz, 4)], table1_z=z1,
                coverage=cov, gaps_real_until_accept=dict(out_gap), w1_expected_agree=round(float(np.mean(agree)), 3) if agree else None, n_agree=len(agree),
                gaze=dict(max=gmax, g_at_R=None if g_at_R is None else round(g_at_R, 5), t_zero_after_R=t_zero, t_done_logged=tph['DONE'], zero_before_done=bool(t_zero is not None and t_zero <= tph['DONE'] + 1e-9),
                          why=n_why), geometry_worst=worst, geometry_worst_baseline=worst_b, geometry_fails=fails, q_ready_note='備妥取實錄 DONE 時量測手臂（≤ ARM_TOL）',
                rows=[{k_: x[k_] for k_ in ('t', 'ph', 'x', 'y', 'g', 'tgt')} | dict(base=int(x['base']['real']), gaze=int(x['gaze']['real']), area_b=int(x['base']['area']), area_g=int(x['gaze']['area'])) for x in rows])


def main():
    out = dict(params=dict(look=GZ.LOOK.tolist(), band_px=GZ.BAND_PX, v_g=GZ.V_G, a_g=GZ.A_G, lead=GZ.LEAD, g_max=GZ.G_MAX, area=GZ.AREA),
               note='開迴路近似（實錄底盤＋參考手臂）；真值只用於評估；閉迴路另見 test_bottle_w1_gaze')
    for k, v in RUNS.items():
        out[k] = one(k, v)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in RUNS:
        o = out[k]
        print('==', k, o['run'], '承接', o['accept_t'], '相位', o['phases'])
        for w, d in o['coverage'].items():
            print(f"   {w}: 實瓶＋支撐 基線參考 {d['base']['real']} 視向 {d['gaze']['real']} 實錄量測 {d['meas']['real']}｜全區 基線 {d['base']['area']} 視向 {d['gaze']['area']}")
        print('   承接前畫面外區段', o['gaps_real_until_accept'], '相機模型對 w1_log expected 一致', o['w1_expected_agree'], f"(n={o['n_agree']})")
        print('   視向', o['gaze'])
        print('   幾何最差', o['geometry_worst'], '未過', o['geometry_fails'])
        print('   基線參考幾何最差', o['geometry_worst_baseline'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
