#!/usr/bin/env python3
"""(b) 觀察觸發的離線時間線核對（Codex 20261111_050821 (b)；Howard 11/11：不無條件左擺、底盤看不到深處可抬臂看）。純離線、左側實錄幾何。
底盤位姿：tpa_nom_3 實錄 s1_steps（每 0.1 s；區域座標）；物件：該趟量測瓶位；手臂抬高觀看＝收攏 → 生成視角的關節 min-jerk（時長規則同 segtarget_vhigh._T）。
可見性：bottle_view_gen.visible（畫面內、俯角 ≥ 5.1°（nom_1 校準）、手臂點雲不遮擋）；候選就緒＝首次連續 0.3 s 可見＋0.6 s（nom_1：支撐首見→就緒 0.46 s＋計畫 0.09 s）。
觸發方案：
  A 現行：實錄 U 起點（任務 6.40 s）＋固定 V_HIGH（含視向偏移的實錄不重現；此處無偏移）；
  B 底盤導引觸發（實跑推估）：v1 xy 首次有效 4.2 s＋5 Hz 三列＋收到延遲 ⇒ 4.8 s（推估，未經實跑驗證）；視角＝以導引 xy 為觀看點、q1 生成；
  C 底盤導引觸發（影格重放實得）：8.1 s；
  D 無導引後備：依區域觀看點（區域知識）在觀察期限觸發；期限＝能在底盤到 x_pre 前留下 抬高＋觀察＋就緒＋TPA 最短時長 的最晚時刻（此處以 C 的同一時刻比較）。
  E 視角可用提早觸發：bottle_w1_obs_ctrl 的 view_useful（抬高後支撐在畫面內、俯角 ≥ 5.1°、相機距 ≤ 1.8 m）沿此路徑的最早觸發點 x −3.541。
報：抬高起點、支撐首見、候選就緒、就緒時底盤到 x_pre 的剩餘距離與名目剩餘時間。
    python3 evaluation/obs_timeline_check.py OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_r0_approach as R0  # noqa: E402
import bottle_side as BS  # noqa: E402
import bottle_view_gen as VG  # noqa: E402
import bottle_w1_segtarget_vhigh_ctrl as VH  # noqa: E402
import bottle_w1_tpa_ctrl as TPA  # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L  # noqa: E402

RUN = 'bottle_motm_tpa_nom_3'


def mj(s):
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def main():
    out = sys.argv[1]
    D = os.path.join(HERE, 'runs', RUN)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    plan = json.load(open(os.path.join(D, 'plan.json')))['plan']
    O = np.array(plan['start_world']) - np.array(plan['start'])
    fr = next(e for e in r['events'] if e['what'] == 'segtarget_freeze')
    xb, yb = fr['xy']
    zs = fr['z_support']
    x_pre = float(r['relwin']['new']['x_pre'])
    t = c['t']

    def base_at(tt):
        k = int(np.argmin(np.abs(t - tt)))
        return np.array([c['base_x'][k] - O[0], c['base_y'][k] - O[1], c['base_yaw'][k]])
    P = VG.bottle_pts(xb, yb, zs)
    clouds = {n: R0.CL[n] for n in R0.ARM_LINKS}
    look_region = BS.look_point('left')
    look_meas = np.array([xb, yb, zs + 0.10])
    rows = []
    for name, t0, look, gen in (('A 現行：X_U0＋固定 V_HIGH', 6.40, None, False), ('B 底盤導引觸發（實跑推估 4.8 s）', 4.80, look_meas, True),
                                ('C 底盤導引觸發（重放實得 8.1 s）', 8.10, look_meas, True), ('D 無導引後備（區域觀看點，8.1 s）', 8.10, look_region, True),
                                ('E 視角可用提早觸發（區域觀看點；x ≥ −3.541 首拍，obs_ctrl 判定）', float(t[np.argmax(c['base_x'] - O[0] >= -3.541)]), look_region, True)):
        T_u_guess = VH._T(R0.Q_TUCK, R0.Q_VHIGH)
        if gen:
            q_obs, info = VG.view_q1(R0.K, base_at(t0 + T_u_guess), R0.Q_VHIGH, look, L.lower[0] + 0.05, L.upper[0] - 0.05)
        else:
            q_obs, info = R0.Q_VHIGH.copy(), dict(q1=float(R0.Q_VHIGH[0]), note='固定 V_HIGH')
        T_u = VH._T(R0.Q_TUCK, q_obs)
        run_, t_vis = 0, None
        for k in range(int((T_u + 6.0) / 0.1) + 1):
            tt = t0 + k * 0.1
            qa = R0.Q_TUCK + (q_obs - R0.Q_TUCK) * mj((tt - t0) / T_u)
            vs = VG.visible(R0.K, base_at(tt), qa, P, clouds)
            run_ = run_ + 1 if vs['ok'] else 0
            if t_vis is None and run_ >= 4:
                t_vis = round(tt - 0.3, 2)
                break
        t_ready = None if t_vis is None else round(t_vis + 0.6, 2)
        rem = None
        if t_ready is not None:
            b = base_at(t_ready)
            rem = dict(x=round(float(b[0]), 3), dist_to_x_pre=round(x_pre - float(b[0]), 3),
                       t_nominal_to_x_pre=round(TPA.base_time(float(b[0]), x_pre, float(r['relwin']['new']['x_slow']), 0.35, 0.1, 0.35), 2))
        row = dict(scheme=name, t_raise=t0, q1_view=info.get('q1'), T_u=T_u, t_support_visible=t_vis, t_candidate_ready=t_ready, at_ready=rem)
        rows.append(row)
        print(row, flush=True)
    json.dump(dict(run=RUN, x_pre=x_pre, rows=rows, actual=dict(t_u0=6.40, t_plan_accept=8.85, note='tpa_nom_3 實錄：U 起點 6.40 s、計畫承接 8.85 s（含視向偏移）'),
                   note='底盤位姿用實錄（舊固定偏航）；B 的 4.8 s 為推估；可見性門檻為單趟校準；右側無實錄資料不在此'),
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
