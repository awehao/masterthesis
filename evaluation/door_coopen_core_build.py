#!/usr/bin/env python3
"""核心段接觸驗證準備：建置抓取基準、走廊／引導表、登錄檔與 H0 核對（規格 results/specs/COOPEN_core_spec_draft1.md；離線，無 Isaac）。

產出（皆不覆寫）：
  results/door/coopen_core_grasp_basis.json：args.R_grasp／grasp_depth_m 照抄正面基準（runs/phf_01_M/task.json，供核對），
    另加 args.T_HG_full（繞緣 φ 60、s 0.85、roll 0 的完整 4×4，把手座標）與 args.a_H（＝−T_HG_full[:3, 2]）；
  results/door/door_coopen_core_table_48.json：K3-M 走廊表格式（anchor＝STAND；q_door_deg、fwd_m、lat_m、dyaw_deg 皆相對錨點座標），
    參考＝q48 候選底盤位姿逐 1° 的 7° 移動平均（同 wgmpc_v2），**0° 端點釘住**：平滑偏移量在 0–7° 線性淡出
    （wgmpc_v2 的平滑使參考起點偏離實際起點 14.158 mm／4.0045°，大於走廊內帶 20 mm／0.8°，會在起點觸發；故改定義並將以本表重跑閉迴路核驗）；
  results/door/coopen_core_params_48.json：θ 48°、門速、OPEN 逾時（完整斜坡＋W1e 同一絕對餘裕 20.3 s）、走廊表與 sha256、
    包絡（W1e 同一規則：±40 mm／±4.5° 在參考位姿座標以 SE(2) 合成角點，於 [0, θ＋1°] 轉回錨點座標取範圍）、觀察器上限 θ＋4°、
    引導（k_lin／k_yaw 1、v_cap 0.035、w_cap 0.10、w_vref 1、w_qn 0）、協同話題 /door/wide_coord、STAND、Q_PRE、K2_READY；
  results/door/coopen_core_h0.json：預抓取（s_pre 0.10）IK、K2_READY → Q_PRE 關節內插 40 點、開指接近 0.10 → 0 m（每 1 cm）的取樣淨空。

r2：r1 的 Q_PRE 以「抓取構型為種子」直接解預抓取，所得構型與抓取構型不在同一連續分支（開指接近 s 0.01 m 處只能跳 3.95 rad 換分支）。
  r2 改為自候選 0° 抓取構型沿 a_H **反向連續後退**到 s_pre（每 2 mm；IK 以前一點為種子，相鄰 ≤ 0.10 rad），Q_PRE＝後退終點；
  求解器 ok 旗標為 False 但位置誤差 < 1 mm、姿態誤差 < 0.01 rad（與 door_offline_check_v1.ik 同一誤差門檻）時接受，並記錄筆數。
  r1 輸出（coopen_core_*.json、door_coopen_core_table_48.json）保留不改；r2 輸出檔名加 _r2。
r3：r2 的「0° 端點釘住」參考在閉迴路核驗（door_coopen_wgmpc_v2 以 r2 表）於 t 20.15 s、約 6.4° 觸發門角參考衝突（推得門角超前 > 0.5°；
  釘住使 0–7° 內偏航需轉約 10°，引導過急），門只到 43.3°——實機即中止，r2 作廢（保留）。
  r3 改為：參考＝wgmpc_v2 已核驗的平滑（不釘端點），**停位 STAND 改取平滑參考 0° 位姿**（不是運行中重設底盤位姿），
  於該停位以候選 0° 抓取構型為種子重解抓取構型（qa_grasp；先連續 IK，不成再多種子並記 grasp_ik_mode），核淨空，再自其反向連續後退得 Q_PRE。輸出檔名加 _r3。
  首次 r3 執行在停位抓取 IK 斷言失敗，已寫出的 r3 抓取基準與走廊表改名 *_partial_crash.json 保留。
  r3 閉迴路核驗（停位＝平滑起點、多種子抓取構型與候選差 4.364 rad＝另一手臂分支）於 t 74.5 s、約 43.4° 門角參考衝突（推得門角落後 > 0.5°）——作廢（保留）。
r4（看過 r2、r3 後的單次修改，揭露）：回到 r2 的「0° 端點釘住、停位＝候選起點、候選抓取構型分支」，只把淡出窗由 7° 加長為 FADE＝20°
  （r2 在 0–7° 內偏航需轉約 10°，引導過急）；若 r4 閉迴路核驗仍不成立，停止調整並照實送審。輸出檔名加 _r4。

    python3 evaluation/door_coopen_core_build.py
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k2_check as KC                             # noqa: E402

C = RV.C
R = os.path.join(HERE, 'results', 'door')
SRC = os.path.join(R, 'door_coopen_joint_v2_phi60_s0.85_q48.json')
GB_FRONT = os.path.join(HERE, 'runs', 'phf_01_M', 'task.json')
OUT_GB = os.path.join(R, 'coopen_core_grasp_basis_r4.json')
OUT_TB = os.path.join(R, 'door_coopen_core_table_48_r4.json')
OUT_PR = os.path.join(R, 'coopen_core_params_48_r4.json')
OUT_H0 = os.path.join(R, 'coopen_core_h0_r4.json')
import door_offline_check_v1 as OC                     # noqa: E402


def ik_cont(base, T, prev, stats):
    q9 = C.ik(base, T, prev, only_prev=True)
    if q9 is not None:
        return np.asarray(q9)[C.IDX]
    qf = np.zeros(9)
    qf[:3] = base
    qf[C.IDX] = prev
    r = OC.solve_ik(C.K, qf, T)
    if r.pos_err < 1e-3 and r.rot_err < 1e-2:
        stats['accepted_not_ok'] += 1
        return np.asarray(r.q)[C.IDX]
    return None
THETA, R_MAX, SMOOTH, S_PRE, FADE = 48.0, 0.6, 7, 0.10, 20
K2_READY = [0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0]
ENV_MM, ENV_DEG = 40.0, 4.5


def se2(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def to_anchor(anchor, p):
    c, s = math.cos(anchor[2]), math.sin(anchor[2])
    dx, dy = p[0] - anchor[0], p[1] - anchor[1]
    return (c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(p[2] - anchor[2]), math.cos(p[2] - anchor[2])))


def smooth_pinned(tr):
    x = np.array([p['base'][0] for p in tr])
    y = np.array([p['base'][1] for p in tr])
    th = np.unwrap(np.array([p['base'][2] for p in tr]))
    h = SMOOTH // 2

    def ma(a):
        return np.array([a[max(0, i - h):i + h + 1].mean() for i in range(len(a))])
    out = []
    for a in (x, y, th):
        m = ma(a)
        w = np.clip(1.0 - np.arange(len(a)) / float(FADE), 0.0, 1.0)
        out.append(m + (a[0] - m[0]) * w)
    return out


def smooth_free(tr):
    """同 door_coopen_wgmpc_v2.smooth_rows（端點縮窗、不釘端點）。"""
    x = np.array([p['base'][0] for p in tr])
    y = np.array([p['base'][1] for p in tr])
    th = np.unwrap(np.array([p['base'][2] for p in tr]))
    h = SMOOTH // 2

    def ma(a):
        return np.array([a[max(0, i - h):i + h + 1].mean() for i in range(len(a))])
    return ma(x), ma(y), ma(th)


def main():
    for f in (OUT_GB, OUT_TB, OUT_PR, OUT_H0):
        if os.path.exists(f):
            raise SystemExit(f'{f} 已存在，不覆寫')
    cand = json.load(open(SRC))['full'][0]
    g = tuple(cand['grasp'])
    tr = cand['trajectory']
    assert abs(tr[0]['q_deg']) < 1e-9 and abs(tr[-1]['q_deg'] - THETA) < 1e-9
    # 抓取基準
    front = json.load(open(GB_FRONT))['args']
    H0 = KC._H_of(0.0)
    T_HG = np.linalg.inv(H0) @ EG.grasp_T(0.0, *g)
    a_H = -T_HG[:3, 2]
    gb = {'args': {'R_grasp': front['R_grasp'], 'grasp_depth_m': front['grasp_depth_m'], 'T_HG_full': T_HG.tolist(), 'a_H': a_H.tolist()},
          'derivation': f'T_HG_full＝inv(H(0))·edgegrasp_v1.grasp_T(0, φ{g[0]:g}, s{g[1]:g}, roll{g[2]:g})；a_H＝−T_HG_full[:3,2]；R_grasp／grasp_depth_m 照抄 {os.path.relpath(GB_FRONT, WS)}（正面，供核對）',
          'front_source_sha256': hashlib.sha256(open(GB_FRONT, 'rb').read()).hexdigest()}
    json.dump(gb, open(OUT_GB, 'w'), ensure_ascii=False, indent=1)
    # 走廊／引導表
    anchor = tuple(float(v) for v in tr[0]['base'])
    xs, ys, ths = smooth_pinned(tr)
    rows = [to_anchor(anchor, (xs[i], ys[i], ths[i])) for i in range(len(tr))]
    tb = {'schema': 'door_coopen_core_table/1', 'source': os.path.relpath(SRC, HERE), 'source_sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest(),
          'smoothing': f'{SMOOTH}° 移動平均（端點縮窗），0° 端點釘住（偏移量 0–{FADE}° 線性淡出）；錨點＝候選 0° 底盤位姿',
          'table': {'anchor': list(anchor), 'q_door_deg': [float(p['q_deg']) for p in tr], 'fwd_m': [r[0] for r in rows], 'lat_m': [r[1] for r in rows],
                    'dyaw_deg': [math.degrees(r[2]) for r in rows]},
          'note': '核心段走廊監看與底盤引導共用同一表；q48 候選為 beam 運動學軌跡，只作參考，不作開迴路命令'}
    json.dump(tb, open(OUT_TB, 'w'), ensure_ascii=False, indent=1)
    tb_sha = hashlib.sha256(open(OUT_TB, 'rb').read()).hexdigest()
    # 包絡（W1e 同一規則）
    corners = []
    for i, p in enumerate(tr):
        ref = (xs[i], ys[i], ths[i])  # noqa: E501
        for dx in (-ENV_MM / 1e3, ENV_MM / 1e3):
            for dy in (-ENV_MM / 1e3, ENV_MM / 1e3):
                for dth in (-math.radians(ENV_DEG), math.radians(ENV_DEG)):
                    corners.append(to_anchor(anchor, se2(ref, (dx, dy, dth))))
    cr = np.array(corners)
    env = {'fwd_lo': float(cr[:, 0].min()), 'fwd_hi': float(cr[:, 0].max()), 'lat_lo': float(cr[:, 1].min()), 'lat_hi': float(cr[:, 1].max()),
           'yaw_abs_deg': float(np.degrees(np.abs(cr[:, 2]).max()))}
    env['cli'] = f"{env['fwd_lo']:.4f},{env['fwd_hi']:.4f},{env['lat_lo']:.4f},{env['lat_hi']:.4f},{env['yaw_abs_deg']:.4f}"
    env['rule'] = '完整外帶 ±40 mm／±4.5° 在參考位姿座標以 SE(2) 合成的角點，於表全範圍 [0, θ] 轉回錨點座標後的範圍（同 W1e；θ＋1° 取端點值）'
    # H0（r2：自抓取構型反向連續後退得 Q_PRE）
    base = np.array(anchor)
    stats = {'accepted_not_ok': 0}
    qa_grasp = ik_cont(base, H0 @ T_HG, np.array(tr[0]['qa']), stats)
    grasp_ik_mode = 'continuation'
    if qa_grasp is None:
        q9 = C.ik(base, H0 @ T_HG, np.array(tr[0]['qa']))
        assert q9 is not None, '停位處抓取 IK 無解'
        qa_grasp, grasp_ik_mode = np.asarray(q9)[C.IDX], 'multi_seed'
    gq = TJ.GEO(list(base) + list(qa_grasp), 0.0, RV.Q_FINGER)
    assert all(gq[k][0] >= th for k, th in (('nonfinger', 0.005), ('finger', 0.005), ('self', 0.005), ('base', 0.030))), f'停位抓取構型淨空不足 {gq}' 
    prev, appr = qa_grasp.copy(), []
    gg = TJ.GEO(list(base) + list(prev), 0.0, C.FINGER_OPEN)
    appr.append({'s_m': 0.0, 'ik': True, **{k: float(gg[k][0]) for k in ('nonfinger', 'finger', 'self', 'base')}})
    for k in range(1, int(round(S_PRE / 0.002)) + 1):
        s_ = 0.002 * k
        Ts = H0 @ np.vstack([np.hstack([np.eye(3), (s_ * a_H).reshape(3, 1)]), [0, 0, 0, 1]]) @ T_HG
        qa = ik_cont(base, Ts, prev, stats)
        if qa is None or float(np.max(np.abs(qa - prev))) > TJ.LIM['joint_step_rad']:
            raise SystemExit(f'後退在 s={s_:.3f} m 不連續（IK 失敗或跳變）')
        prev = qa
        if k % 5 == 0:
            gg = TJ.GEO(list(base) + list(qa), 0.0, C.FINGER_OPEN)
            appr.append({'s_m': s_, 'ik': True, **{k_: float(gg[k_][0]) for k_ in ('nonfinger', 'finger', 'self', 'base')}})
    q_pre = prev
    unfold = []
    for t in np.linspace(0, 1, 41):
        qa = (1 - t) * np.array(K2_READY) + t * q_pre
        gg = TJ.GEO(list(base) + list(qa), 0.0, C.FINGER_OPEN)
        unfold.append({k: float(gg[k][0]) for k in ('nonfinger', 'finger', 'self', 'base')} | {'who_' + k: gg[k][1] for k in ('nonfinger', 'finger', 'self', 'base')})
    h0 = {'stand': list(anchor), 'qa_grasp': qa_grasp.tolist(), 'grasp_ik_mode': grasp_ik_mode,
          'grasp_clear_mm': {k: float(gq[k][0]) * 1e3 for k in ('nonfinger', 'finger', 'self', 'base')}, 'qa_grasp_vs_candidate_max_rad': float(np.max(np.abs(qa_grasp - np.array(tr[0]['qa'])))),
          'stand_vs_candidate': [float(anchor[0] - tr[0]['base'][0]), float(anchor[1] - tr[0]['base'][1]), math.degrees(anchor[2] - tr[0]['base'][2])],
          'q_pre': q_pre.tolist(), 's_pre_m': S_PRE, 'k2_ready': K2_READY,
          'unfold_min_mm': {k: min(u[k] for u in unfold) * 1e3 for k in ('nonfinger', 'finger', 'self', 'base')},
          'unfold_min_who': {k: min(unfold, key=lambda u: u[k])['who_' + k] for k in ('nonfinger', 'finger', 'self', 'base')},
          'approach': appr, 'approach_ik_fail': sum(1 for a in appr if not a['ik']), 'ik_accepted_not_ok': stats['accepted_not_ok'],
          'approach_rule': '自候選 0° 抓取構型沿 a_H 反向連續後退（每 2 mm、相鄰 ≤ 0.10 rad），Q_PRE＝後退終點；求解器 ok＝False 但誤差在 door_offline_check_v1.ik 同一門檻內時接受',
          'approach_min_mm': {k: min(a[k] for a in appr if a['ik']) * 1e3 for k in ('nonfinger', 'finger', 'self', 'base')},
          'note': '關節內插與逐點取樣，非規劃器軌跡；開指接近 s＝0 處非手指為夾爪殼—橫桿設計距離'}
    json.dump(h0, open(OUT_H0, 'w'), ensure_ascii=False, indent=1)
    # 登錄檔
    ramp_s = (THETA - 2.0) / R_MAX + 2.0 / 0.25
    pr = {'schema': 'coopen_core_params/1', 'spec': 'evaluation/results/specs/COOPEN_core_spec_draft1.md', 'theta_goal_deg': THETA,
          'task_param_extension': {'theta_goal_deg': THETA, 'over_goal_deg': 1.0, 'policy_upper_deg': THETA + 1.0, 'r_max_dps': R_MAX, 'r_min_dps': 0.25,
                                   'ramp_s': ramp_s, 'open_margin_s': 20.3, 'open_timeout_s': math.ceil(ramp_s + 20.3),
                                   'rule': 'OPEN 逾時＝完整斜坡（含末段 2° 以 0.25 °/s 減速）＋W1e 同一絕對餘裕 20.3 s'},
          'grasp': {'basis': os.path.relpath(OUT_GB, WS), 'basis_sha256': hashlib.sha256(open(OUT_GB, 'rb').read()).hexdigest(),
                    'phi_deg': g[0], 's_m': g[1], 'roll_deg': g[2]},
          'protection_scope_change': {'corridor_table': os.path.relpath(OUT_TB, WS), 'corridor_table_sha256': tb_sha,
                                      'corridor_inner_mm': 20.0, 'corridor_inner_deg': 0.8, 'envelope': env,
                                      'observer_door_max_deg': THETA + 4.0, 'observer_rule': '固定角度餘裕 θ＋4°（同 W1e）'},
          'control_cost_change': {'guidance': {'k_lin': 1.0, 'k_yaw': 1.0, 'v_cap': 0.035, 'w_cap': 0.10, 'w_vref': 1.0, 'w_qn': 0.0},
                                  'coord_topic': '/door/wide_coord', 'phases_on': ['OPEN'], 'phases_off_weight': 0.0,
                                  'note': '引導參考速度限幅變更（世界座標每軸 0.035 m/s、偏航 0.10 rad/s；求解器本體方框 0.035255 m/s 不變）；PAUSE 起 w_vref＝0'},
          'start': {'stand': list(anchor), 'stand_cli': ','.join(f'{v:.6f}' for v in anchor), 'k2_ready_cli': ','.join(f'{v:g}' for v in K2_READY),
                    'q_pre_cli': ','.join(f'{v:.4f}' for v in q_pre), 'experimental_initialization': '底盤以 --start-pose 直接放在停位（不經導航接近），須揭露'},
          'unchanged': ['求解器與權重', 'E2', '手指暫停期上限 20 N', '幾何監看門檻', '暫停／停止定義', '判準']}
    json.dump(pr, open(OUT_PR, 'w'), ensure_ascii=False, indent=1)
    print(f"抓取基準 a_H {np.round(a_H, 4).tolist()}；表 {len(rows)} 列（起點偏移 {rows[0]}）；包絡 {env['cli']}；"
          f"H0 展開最小 {({k: round(v, 1) for k, v in h0['unfold_min_mm'].items()})} mm（{h0['unfold_min_who']}）；"
          f"接近 IK 失敗 {h0['approach_ik_fail']}、最小 {({k: round(v, 1) for k, v in h0['approach_min_mm'].items()})} mm；OPEN 逾時 {pr['task_param_extension']['open_timeout_s']} s")


if __name__ == '__main__':
    main()
