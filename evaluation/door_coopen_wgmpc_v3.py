#!/usr/bin/env python3
"""連續協同通行：核心段末段持握閉合 C2 的離線 A/B（door_coopen_wgmpc_v2.py 的版本化複本；Codex r35：選 C2，一組 A/B，不掃描；無 Isaac）。
A＝原設定（同 v2 登錄核驗：登錄表 door_coopen_core_table_48_r4、h0 r4 起始分支、共同期限 105 s）；B＝A＋C2（door_coopen_closure_guidance，
參數 DEFAULT 於 A/B 前固定：門角 36° 起平滑啟用、40° 全量、τ 1.0 s、每軸變化率 2 mm/s、幅度 8 mm）。A、B 其餘完全相同。
新增報（Codex r35 §3）：
  - 閉合：全段與 40–48° 的位置範數／徑向／高度／姿態最大值；36–48° 每 1° 的平均與最大（逐門角序列）。
  - 主目標：同一定義的連續出口距離 D（雙線性、門角包絡場；同 v2）、開門同期進展占比、D 序列。
  - 可執行性：修正量（|c|、s）、底盤引導 base_vref 世界速度各軸最大、套用命令（u_app）各軸最大與相對求解器方框比例、
    一階差分加速度最大、飽和占比與最長、求解失敗、門角參考衝突。
  - 幾何：同一條軌跡的取樣淨空（非手指／手指／自碰／底盤）；門角 ≥ 36° 每 2 輪取樣，其餘每 8 輪（v2 為全程每 8 輪）。
  - 計算負擔：閉合修正每輪計算耗時（含 TCP 正運動學）p50／p99／最大。
界線（同 v2）：門角由 TCP 推得（非嚴格剛性持握受控體），閉合殘差的徑向／高度／姿態有意義、切向依定義為 0；無接觸、門動力學、E2／安全層；
  不驗兩指承載或內力；不改門角或誤差定義。
以下為 v2 說明。
連續協同通行：核心控制驗證 v2——參考平滑＋開門降速（Codex 20261026_234915 §3；一組，跑完即送，不追加掃描；無 Isaac）。

基準＝door_coopen_wgmpc_v1 的 E 組（引導 w_vref 1、w_qn 0、引導限幅 v_cap 0.035 m/s（世界座標每軸）、w_cap 0.10 rad/s）。只改兩處：
  1. 參考平滑：q48 候選逐 1° 的底盤 x、y、偏航（unwrap 後）做 7° 視窗移動平均（端點縮窗），不重掃抓法；
  2. 門角參考最大速率 r_max 1.0 → 0.6 °/s（r_min 0.25 °/s、近目標 2°、超前 0.5° 不變）。
其餘（受控體、延遲、補償、底盤增益、手臂設定點模型、目標門角 48°）同 v1。
報（受控體軌跡逐輪）：
  - 推得門角最大值、求解失敗、門角參考衝突；
  - 相對平滑參考的追蹤誤差（位置、偏航）全程與 38–48° 最大值；
  - 出口進展：門角包絡 [46, 53]° 淨空場上的真實測地距離，以**雙線性內插**取值（四角任一不可通行即記 None），
    報 38°→48° 變化，以及「進展發生時門仍在開啟」的比例（D 下降的輪次中推得門角速率 > 0.05 °/s 的占比）；
  - 底盤套用命令相對求解器本體方框（0.035255 m/s 每軸、偏航界限取求解器設定）的飽和占比與最長連續飽和時間（≥ 99%）；
    世界座標合速度最大（對照 E2 數值 0.05 m/s；**本框架未執行 E2 與安全層**）；
  - 持握幾何閉合殘差（由 TCP 推得的把手位姿 vs 推得門角的把手位姿）：徑向（水平、相對鉸鏈軸）、高度、切向、姿態角，最大值；
  - 取樣淨空（每 8 輪）非手指／手指／自碰／底盤，不稱連續安全證明。
**界線**：離線閉迴路模型（門角由 TCP 推得，非嚴格剛性持握受控體）；無接觸、門動力學、E2／安全層；不含放手與收臂。

    python3 evaluation/door_coopen_wgmpc_v2.py   # 寫 results/door/door_coopen_wgmpc_v2.json（不覆寫）
    python3 evaluation/door_coopen_wgmpc_v2.py results/door/door_coopen_core_table_48_r2.json   # 以登錄表為參考重跑（核心段登錄核驗）
    python3 evaluation/door_coopen_wgmpc_v2.py results/door/door_coopen_core_table_48_r3.json results/door/coopen_core_h0_r3.json   # 另以 h0 的停位＋抓取構型起始
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_joint_v2 as J2                      # noqa: E402
import door_coopen_release_set_v1 as RS                # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k3m_offline_replay as R                    # noqa: E402
import door_target_k3 as T3                            # noqa: E402
import door_wide_guidance as G                         # noqa: E402
import door_coopen_closure_guidance as CG              # noqa: E402  v3：C2

CL, KP, KC, C = R.CL, R.KP, R.KC, R.C
XS, YS = J2.XS, J2.YS
SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_joint_v2_phi60_s0.85_q48.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_wgmpc_v3.json')
TABLE_REG = os.path.join(HERE, 'results', 'door', 'door_coopen_core_table_48_r4.json')
H0_REG = os.path.join(HERE, 'results', 'door', 'coopen_core_h0_r4.json')
T_END_REG = 105.0
TH = {'nonfinger': 0.005, 'finger': 0.005, 'self': 0.005, 'base': 0.020}
V_CAP, W_CAP, W_VREF = 0.035, 0.10, 1.0
R_MAX_DPS, SMOOTH_DEG = 0.6, 7


def smooth_rows(tr):
    q = np.array([p['q_deg'] for p in tr])
    x = np.array([p['base'][0] for p in tr])
    y = np.array([p['base'][1] for p in tr])
    th = np.unwrap(np.array([p['base'][2] for p in tr]))
    h = SMOOTH_DEG // 2

    def ma(a):
        return np.array([a[max(0, i - h):i + h + 1].mean() for i in range(len(a))])
    xs, ys, ts = ma(x), ma(y), ma(th)
    return [{'q_door_deg': float(q[i]), 'fwd_m': float(xs[i]), 'lat_m': float(ys[i]), 'dyaw_deg': math.degrees(ts[i]),
             'q': [float(xs[i]), float(ys[i]), float(ts[i])] + list(tr[i]['qa'])} for i in range(len(tr))]


def D_bilinear(D, p):
    fx = (p[0] - XS[0]) / (XS[1] - XS[0])
    fy = (p[1] - YS[0]) / (YS[1] - YS[0])
    i, j = int(math.floor(fx)), int(math.floor(fy))
    if not (0 <= i < len(XS) - 1 and 0 <= j < len(YS) - 1):
        return None
    c = [D[i, j], D[i + 1, j], D[i, j + 1], D[i + 1, j + 1]]
    if not all(math.isfinite(v) for v in c):
        return None
    a, b = fx - i, fy - j
    return float((1 - a) * (1 - b) * c[0] + a * (1 - b) * c[1] + (1 - a) * b * c[2] + a * b * c[3])


def rot_angle(Ra, Rb):
    return math.degrees(math.acos(float(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def rows_from_table(path):
    t = json.load(open(path))['table']
    ax, ay, ath = t['anchor']
    c, s_ = math.cos(ath), math.sin(ath)
    out = []
    for q, f, l, dy in zip(t['q_door_deg'], t['fwd_m'], t['lat_m'], t['dyaw_deg']):
        out.append({'q_door_deg': q, 'fwd_m': ax + c * f - s_ * l, 'lat_m': ay + s_ * f + c * l, 'dyaw_deg': math.degrees(ath) + dy,
                    'q': [ax + c * f - s_ * l, ay + s_ * f + c * l, ath + math.radians(dy)] + [0.0] * 6})
    return out


def main():
    global OUT
    variant = sys.argv[1] if len(sys.argv) > 1 else ''          # v3：A（原設定）｜B（＋C2）；表、起始、期限固定為登錄值
    if variant not in ('A', 'B'):
        raise SystemExit('用法：door_coopen_wgmpc_v3.py A|B')
    table_path = TABLE_REG
    OUT = OUT.replace('.json', f'_{variant}.json')
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    cand = json.load(open(SRC))['full'][0]
    g = tuple(cand['grasp'])
    theta = float(cand['q_rel_deg'])
    T_hold = EG.grasp_T(0.0, *g)
    tr = cand['trajectory']
    rows = rows_from_table(table_path) if table_path else smooth_rows(tr)
    table = G.RefTable(rows, (0.0, 0.0, 0.0))
    ident = json.load(open(R.IDENT))
    q0 = np.array(list(tr[0]['base']) + list(tr[0]['qa']), float)
    h0 = json.load(open(H0_REG))                               # v3：固定 h0 r4 起始分支（核心段登錄核驗）
    q0 = np.array(list(h0['stand']) + list(h0['qa_grasp']), float)
    alpha, bias = np.asarray(ident['alpha'], float), np.asarray(ident['bias_rad'], float)
    s0 = q0[3:9] - bias / alpha
    K = CL.WholeBodyKinematics.from_urdf_file(C.URDF)
    CL.BASE_GAIN = np.array([0.9, 0.9, 0.82])
    t_run = json.load(open(os.path.join(R.RUN, 'task.json')))
    r_run = json.load(open(os.path.join(R.RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r_run['steps_cols'])}
    ev = {e['to']: e['t'] for e in t_run['policy_events'] if 'to' in e}
    hold = [s for s in r_run['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    _, Dtrue = J2.d_map(RS.field(theta))
    cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05, base_fixed=False)
    v_box = float(cfg.v_base_lin)
    w_box = float(cfg.v_base_ang)
    cfgp = KP.K3AConfig(theta_goal_deg=theta, r_max_dps=R_MAX_DPS)
    H0 = KC._H_of(0.0)
    G_T_H = np.linalg.inv(T_hold) @ H0
    a0 = math.atan2(H0[1, 3], H0[0, 3])
    gcfg = G.GuidanceCfg(w_vref=W_VREF, w_qn=0.0, v_cap=V_CAP, w_cap=W_CAP)
    st = {'dtk': T3.DoorTargetK3(T_hold, H0, cfgp.target_cfg()), 'T': T_hold, 'q_ref_prev': None, 'log': {}, 'conflict': None,
          't_prev': None, 'cg': [], 'vref_w': [], 'cg_time': []}
    corr = CG.ClosureCorrector(CG.DEFAULT) if variant == 'B' else None
    G_inv = np.linalg.inv(G_T_H)

    def pre(t, q, c):
        qi = R.door_angle(q, G_T_H, a0)
        try:
            T, _ = st['dtk'].step(KC._H_of(qi), qi, R.DT)
            st['T'] = T
        except T3.DoorTargetK3Conflict as e:
            st['conflict'] = st['conflict'] or {'t': t, 'msg': str(e)}
        qr = st['dtk'].q_ref
        rate = 0.0 if st['q_ref_prev'] is None else (qr - st['q_ref_prev']) / R.DT
        st['q_ref_prev'] = qr
        tbl = table
        if corr is not None:                                  # v3 B：C2 閉合修正（只改參考位姿平移）
            t_c0 = time.perf_counter()
            dt_c = R.DT if st['t_prev'] is None else max(t - st['t_prev'], 1e-6)
            p_tcp = K.fk(np.asarray(q[:9], float), 'link_tcp')[:3, 3]
            p_tg = (KC._H_of(qi) @ G_inv)[:3, 3]
            s_c, c_c, info_c = corr.update(dt_c, math.degrees(qi), p_tcp[:2], p_tg[:2])
            tbl = CG.ShiftedTable(table, corr.ref_offset())
            st['cg_time'].append(time.perf_counter() - t_c0)
            st['cg'].append((t, math.degrees(qi), s_c, float(np.linalg.norm(c_c)), info_c.get('why')))
        st['t_prev'] = t
        coord, info_g = G.guidance(tbl, gcfg, (q[0], q[1], q[2]), qi, rate)
        st['vref_w'].append(info_g['v_world'])
        c.w_vref, c.base_vref = coord[1], tuple(coord[2:5])
        st['log'][round(t, 6)] = qi

    ref = R.reference(T_hold, cfgp)
    t_end = ref[-1][0] + 3.0 + R.HOLD_AFTER_S
    t_end = T_END_REG                                          # v3：共同期限固定 105 s
    OUT_T = OUT
    rec = CL.run(cfg, K, q0, s0, lambda t: st['T'], t_end=t_end, reach_p=0.005, reach_r=0.02, hold_s=1e9, pre_solve=pre,
                 delay_cycles=1.4, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=0.0)['rec']
    n = len(rec['t'])
    qd = np.array([st['log'].get(round(t, 6), R.door_angle(np.asarray(rec['q'][i], float), G_T_H, a0)) for i, t in enumerate(rec['t'])])
    qdeg = np.degrees(qd)
    rate = np.gradient(qdeg, np.asarray(rec['t'], float))
    dev, clo, Dv, geo, sat, vw = [], [], [], [], [], []
    for i in range(n):
        TJ.cool()
        q = np.asarray(rec['q'][i], float)
        pr = table.pose(qd[i])
        dev.append((math.hypot(q[0] - pr[0], q[1] - pr[1]), abs(G.wrap(q[2] - pr[2]))))
        Hi = KC._tcp(list(q[:3]), list(q[3:9])) @ G_T_H
        Hq = KC._H_of(qd[i])
        ri, rq = math.hypot(Hi[0, 3], Hi[1, 3]), math.hypot(Hq[0, 3], Hq[1, 3])
        ai, aq = math.atan2(Hi[1, 3], Hi[0, 3]), math.atan2(Hq[1, 3], Hq[0, 3])
        clo.append((abs(ri - rq), abs(Hi[2, 3] - Hq[2, 3]), abs(rq * G.wrap(ai - aq)), rot_angle(Hi[:3, :3], Hq[:3, :3]),
                    float(np.linalg.norm(Hi[:3, 3] - Hq[:3, 3]))))
        Dv.append(D_bilinear(Dtrue, q))
        if i % 8 == 0 or (qdeg[i] >= 36.0 and i % 2 == 0):         # v3：末段加密
            gg = TJ.GEO([float(v) for v in q[:9]], qd[i], q_finger)
            geo.append({k: float(gg[k][0]) for k in TH} | {'who_' + k: gg[k][1] for k in TH})
        u = np.asarray(rec['u_app'][i], float)
        s_lin = max(abs(u[0]), abs(u[1])) >= 0.99 * v_box
        s_ang = (w_box is not None and abs(u[2]) >= 0.99 * w_box)
        sat.append(s_lin or s_ang)
        c_, s_ = math.cos(q[2]), math.sin(q[2])
        vw.append(math.hypot(c_ * u[0] - s_ * u[1], s_ * u[0] + c_ * u[1]))
    dev, clo = np.array(dev), np.array(clo)
    sel = qdeg >= theta - 10
    longest, cur = 0, 0
    for s in sat:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    i38 = next((i for i in range(n) if qdeg[i] >= theta - 10 and Dv[i] is not None), None)
    i48 = max((i for i in range(n) if Dv[i] is not None and qdeg[i] <= theta + 0.01), default=None)
    drops = [(i, Dv[i - 1] - Dv[i]) for i in range(1, n) if Dv[i] is not None and Dv[i - 1] is not None and Dv[i - 1] - Dv[i] > 1e-6]
    open_share = (sum(1 for i, _ in drops if rate[i] > 0.05) / len(drops)) if drops else None
    gmin = {k: min(x[k] for x in geo) for k in TH}
    res = {'schema': 'door_coopen_wgmpc/2', 'source': os.path.relpath(SRC, HERE), 'base_config': 'wgmpc_v1 E（w_vref 1、w_qn 0、v_cap 0.035、w_cap 0.10）',
           'changes': {'reference_smoothing_deg': SMOOTH_DEG, 'r_max_dps': R_MAX_DPS},
           'solver_box': {'v_axis_body': v_box, 'w': w_box},
           'q_door_max_deg': float(qdeg.max()), 'n_solve_fail': int(sum(1 for o in rec['ok'] if not o)), 'target_conflict': st['conflict'],
           'duration_s': float(rec['t'][-1]),
           'track_pos_max_mm': float(dev[:, 0].max() * 1e3), 'track_yaw_max_deg': float(np.degrees(dev[:, 1].max())),
           'track_pos_38_48_max_mm': float(dev[sel, 0].max() * 1e3) if sel.any() else None,
           'D_38_m': None if i38 is None else Dv[i38], 'D_48_m': None if i48 is None else Dv[i48],
           'D_drop_38_48_m': (Dv[i38] - Dv[i48]) if (i38 is not None and i48 is not None) else None,
           'D_drop_share_while_opening': open_share,
           'sat_share': float(np.mean(sat)), 'sat_longest_s': longest * R.DT, 'v_world_max_m_s': float(max(vw)),
           'closure_residual_max': {'radial_mm': float(clo[:, 0].max() * 1e3), 'height_mm': float(clo[:, 1].max() * 1e3),
                                    'tangential_mm': float(clo[:, 2].max() * 1e3), 'orientation_deg': float(clo[:, 3].max())},
           'geo_min_mm': {k: v * 1e3 for k, v in gmin.items()}, 'geo_min_who': {k: min(geo, key=lambda x: x[k])['who_' + k] for k in TH},
           'pass_thresholds_sampled': all(gmin[k] >= TH[k] for k in TH),
           'series': {'t': [float(t) for t in rec['t'][::5]], 'q_deg': qdeg[::5].tolist(), 'D': [Dv[i] for i in range(0, n, 5)],
                      'base': [list(map(float, rec['q'][i][:3])) for i in range(0, n, 5)]},
           'ref_smoothed': rows,
           'not_modeled': ['門角由 TCP 推得，非嚴格剛性持握受控體', '接觸、門動力學', 'E2 與安全層未執行', '放手與收臂']}
    print(f"v2（平滑 {SMOOTH_DEG}°、門 {R_MAX_DPS} °/s）：門角 {res['q_door_max_deg']:.1f}°、求解失敗 {res['n_solve_fail']}、衝突 {st['conflict'] is not None}、時長 {res['duration_s']:.0f} s；"
          f"追蹤 位置 {res['track_pos_max_mm']:.0f} mm（38–48° {res['track_pos_38_48_max_mm']}）、偏航 {res['track_yaw_max_deg']:.1f}°；"
          f"D 38→48 {res['D_38_m']}→{res['D_48_m']}（降 {res['D_drop_38_48_m']}；門在開的占比 {open_share}）；"
          f"飽和占比 {res['sat_share']:.2f}、最長 {res['sat_longest_s']:.2f} s、世界 |v| {res['v_world_max_m_s']:.3f}；"
          f"閉合殘差 {({k: round(v, 1) for k, v in res['closure_residual_max'].items()})}；取樣淨空 {({k: round(v, 1) for k, v in res['geo_min_mm'].items()})} mm；{time.time() - t0:.0f} s", flush=True)
    res['t_end_s'] = float(t_end)
    res['schema'] = 'door_coopen_wgmpc/3'
    res['variant'] = variant
    res['closure_cfg'] = None if corr is None else vars(CG.DEFAULT)
    sel40 = qdeg >= 40.0
    res['closure_max_all'] = {'norm_mm': float(clo[:, 4].max() * 1e3), 'radial_mm': float(clo[:, 0].max() * 1e3), 'height_mm': float(clo[:, 1].max() * 1e3),
                              'orientation_deg': float(clo[:, 3].max())}
    res['closure_max_40_48'] = None if not sel40.any() else {'norm_mm': float(clo[sel40, 4].max() * 1e3), 'radial_mm': float(clo[sel40, 0].max() * 1e3),
                                                             'height_mm': float(clo[sel40, 1].max() * 1e3), 'orientation_deg': float(clo[sel40, 3].max())}
    bins = []
    for lo in range(36, 48):
        w = (qdeg >= lo) & (qdeg < lo + 1)
        if w.any():
            bins.append({'q_bin_deg': [lo, lo + 1], 'n': int(w.sum()), 'norm_mean_mm': float(clo[w, 4].mean() * 1e3), 'norm_max_mm': float(clo[w, 4].max() * 1e3),
                         'radial_max_mm': float(clo[w, 0].max() * 1e3), 'height_max_mm': float(clo[w, 1].max() * 1e3), 'orientation_max_deg': float(clo[w, 3].max())})
    res['closure_bins_1deg'] = bins
    U = np.asarray(rec['u_app'], float)
    tt = np.asarray(rec['t'], float)
    dU = np.diff(U[:, :3], axis=0) / np.maximum(np.diff(tt)[:, None], 1e-9)
    VW = np.asarray(st['vref_w'], float)
    res['executability'] = {'u_app_base_absmax': np.abs(U[:, :3]).max(axis=0).tolist(), 'u_app_ratio_to_box': [float(np.abs(U[:, 0]).max() / v_box), float(np.abs(U[:, 1]).max() / v_box),
                            float(np.abs(U[:, 2]).max() / w_box) if w_box else None], 'base_acc_absmax': np.abs(dU).max(axis=0).tolist(),
                            'base_vref_world_absmax': np.abs(VW).max(axis=0).tolist() if len(VW) else None, 'v_cap': V_CAP, 'w_cap': W_CAP}
    if corr is not None:
        CGA = st['cg']
        ct = np.asarray(st['cg_time'], float)
        res['correction'] = {'n_update': corr.n_update, 'n_invalid': corr.n_invalid, 'n_clip_mag': corr.n_clip_mag, 'n_rate_limited': corr.n_rate_limited,
                             'c_norm_max_mm': float(max(x[3] for x in CGA) * 1e3), 'c_final_mm': (corr.c * 1e3).tolist(), 's_final': corr.s,
                             'series': [[float(x[0]), float(x[1]), x[2], x[3] * 1e3] for x in CGA[::10]],
                             'compute_s': {'p50': float(np.percentile(ct, 50)), 'p99': float(np.percentile(ct, 99)), 'max': float(ct.max())}}
    print(f"  v3 {variant}：閉合 全段 {({k: round(v, 2) for k, v in res['closure_max_all'].items()})}；40–48° {res['closure_max_40_48'] and {k: round(v, 2) for k, v in res['closure_max_40_48'].items()}}", flush=True)
    for b in bins:
        print(f"    {b['q_bin_deg']} 範數 均 {b['norm_mean_mm']:.2f} 最大 {b['norm_max_mm']:.2f}；徑向 {b['radial_max_mm']:.2f} 高度 {b['height_max_mm']:.2f} 姿態 {b['orientation_max_deg']:.2f}°", flush=True)
    print(f"  可執行性 {res['executability']}", flush=True)
    if corr is not None:
        print(f"  修正 {({k: v for k, v in res['correction'].items() if k != 'series'})}", flush=True)
    i48 = next((i for i in range(n) if qdeg[i] >= theta - 0.05), None)
    res['t_reach_goal_s'] = None if i48 is None else float(rec['t'][i48])
    print(f"  到達 {theta - 0.05:.2f}° 的時刻：{res['t_reach_goal_s']} s（上限 {t_end:.0f} s）", flush=True)
    json.dump(res, open(OUT_T, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
