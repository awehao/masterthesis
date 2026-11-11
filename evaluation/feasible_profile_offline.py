#!/usr/bin/env python3
"""A0 離線對照 r1：在既有實跑紀錄的**實際走過的接近段幾何**（位置＋偏航，0→閉爪請求）上，比較
  舊計畫剖面（v_nav 0.35→x_slow 後 v_c 0.10、A_SLEW 梯形；站位時間成本、TPA／TPAS 用的 base_time 同型）、
  A0 輪級可行剖面（bottle_feasible_profile；同樣的 v_nav／v_c／x_slow／加減速，只多輪級矩陣與偏航率上限）、
  實測。
只讀紀錄，不改控制器、不調參。幾何取自實測軌跡（含追蹤誤差與量測路徑換版），不是重新規劃的參考；A0 接進控制器後的閉迴路行為不在本核對內。
    python3 evaluation/feasible_profile_offline.py <out.json> run1 run2 ...
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_feasible_profile as FP                              # noqa: E402

V_NAV, V_C, A, W_MAX = 0.35, 0.10, 0.4, 0.6                       # 既有設定：S1 v_nav／v_c、A_SLEW、航向層 W_MAX
print('#EARLY 設定', dict(v_nav=V_NAV, v_c=V_C, a=A, w_max=W_MAX, margin=FP.MARGIN), flush=True)


def smooth(y, n):
    k = np.ones(n) / n
    return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode='edge'), k, mode='valid')


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    s = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {n: i for i, n in enumerate(list(s['cols']))}
    R = s['rows']
    t_close = float(r['t_close'])
    m = R[:, c['t']] <= t_close
    t = R[m, c['t']][::5]                                           # 0.05 s
    xy = np.c_[R[m, c['base_x']], R[m, c['base_y']]][::5]
    yaw = np.unwrap(R[m, c['base_yaw']][::5])
    xy = np.c_[smooth(xy[:, 0], 5), smooth(xy[:, 1], 5)]
    yaw = smooth(yaw, 9)                                            # 偏航平滑 0.45 s：取參考偏航的近似，不把追蹤抖動算成 dθ/dσ
    P = FP.path_arrays(xy, yaw)
    off = float(xy[-1, 0]) - 0.0
    tpa = r.get('tpa') or {}
    x_pre_w = None
    ev = {e['what']: e for e in r['events']}
    sw = (r.get('tpas') or {}).get('sw') or {}
    x_sw_region = sw.get('x')
    # 區域 x → 世界 x 的偏移：用切換時刻的實測世界 x 對區域 x
    if x_sw_region is not None:
        i_sw = int(np.argmin(np.abs(R[:, c['t']] - float(sw['t_sw']))))
        x_off = float(R[i_sw, c['base_x']]) - float(x_sw_region)
        x_pre_w = float(tpa['base_end'][0]) + x_off
    else:
        x_off, x_pre_w = None, float(xy[-1, 0])
    x_slow_w = x_pre_w - 0.6
    # 以弧長表示分區：x 首次過 x_slow 的弧長
    i_slow = int(np.argmax(P['xy'][:, 0] >= x_slow_w)) if (P['xy'][:, 0] >= x_slow_w).any() else len(P['sig']) - 1
    i_pre = int(np.argmax(P['xy'][:, 0] >= x_pre_w)) if (P['xy'][:, 0] >= x_pre_w).any() else len(P['sig']) - 1
    vt = np.where(np.arange(len(P['sig'])) < i_slow, V_NAV, V_C)
    # 實測
    v_meas = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T) / 0.05]
    v_meas_s = smooth(v_meas, 5)
    cap, W = FP.wheel_cap()
    cc = np.load(os.path.join(D, 'cmd_chain.npz'), allow_pickle=True)
    mk = cc['t_send'] <= t_close
    use_cmd = np.max(np.abs(cc['u'][mk][:, :3] @ W.T), axis=1) / cap
    # 舊剖面（純梯形，不看輪級）
    old = FP.sweep(P, vt, 0.0, A, A)
    body_old = np.stack([np.cos(P['phi'] - P['th']), np.sin(P['phi'] - P['th']), P['dth']], axis=1) * old['v'][:, None]
    use_old = np.max(np.abs(body_old @ W.T), axis=1) / cap
    # A0
    new = FP.profile(P, vt, 0.0, A, A, w_max=W_MAX)

    def seg(tt, vv, i0, i1):
        return round(float(tt[i1] - tt[i0]), 2)
    t_meas_idx = t - t[0]
    # 實測時間在路徑點上的對應（path_arrays 可能併掉重複點 ⇒ 以弧長對回）
    sig_meas = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    t_at = np.interp(P['sig'], sig_meas, t_meas_idx)
    out = dict(run=run, L=round(float(P['sig'][-1]), 3), t_close=t_close, x_pre_world=round(x_pre_w, 4), x_slow_world=round(x_slow_w, 4),
               sig_slow=round(float(P['sig'][i_slow]), 3), sig_pre=round(float(P['sig'][i_pre]), 3))
    rows = {}
    for nm, tt, vv in (('實測', t_at, np.interp(P['sig'], sig_meas, v_meas_s)), ('舊計畫剖面', old['t'], old['v']), ('A0', new['t'], new['v'])):
        rows[nm] = dict(T_total=round(float(tt[-1]), 2), T_cruise_to_slow=seg(tt, vv, 0, i_slow), T_slow_to_pre=seg(tt, vv, i_slow, i_pre), T_pre_to_close=seg(tt, vv, i_pre, len(tt) - 1),
                        v_max=round(float(np.max(vv)), 4), v_median_cruise=round(float(np.median(vv[:max(i_slow, 1)])), 4), reversals=FP.reversals(vv, tt))
    rows['舊計畫剖面']['wheel_use_max'] = round(float(use_old.max()), 3)
    rows['舊計畫剖面']['wheel_over_frac_by_time'] = round(float(np.sum(np.diff(old['t'])[use_old[1:] > 1.0]) / max(old['t'][-1], 1e-9)), 3)
    rows['A0']['wheel_use_max'] = round(float(new['wheel_ratio'].max()), 3)
    rows['A0']['limit_src_frac'] = {k: round(float(np.mean(new['lim']['src'] == i)), 3) for i, k in enumerate(('目標速度', '輪級', '偏航率'))}
    rows['實測']['wheel_use_cmd_max'] = round(float(use_cmd.max()), 3)
    rows['實測']['wheel_use_cmd_ge97_frac'] = round(float(np.mean(use_cmd >= 0.97)), 3)
    out['rows'] = rows
    # 相位：承接點到 x_pre 的時間（TPAS 的 T 與實際）
    if tpa.get('t_e') is not None and x_sw_region is not None:
        i_e = int(np.argmin(np.abs(t_at - float(tpa['t_e']))))
        T_meas = float(sw['t_sw']) - float(tpa['t_e'])
        T_old = float(old['t'][i_pre] - old['t'][i_e])
        T_new = float(new['t'][i_pre] - new['t'][i_e])
        # 舊剖面承接點速度不是 0：用控制器當時的 t_base（紀錄）當舊計畫值，梯形值只作旁證
        out['phase'] = dict(t_handover=float(tpa['t_e']), T_plan_recorded=float(tpa['T']), t_base_recorded=float(tpa['t_base']), T_measured=round(T_meas, 2),
                            rate_recorded_mean=round(float(tpa['T']) / T_meas, 3), T_old_on_geometry=round(T_old, 2), T_A0_on_geometry=round(T_new, 2),
                            rate_if_A0=round(T_new / T_meas, 3), note='rate＝計畫時長／實際時長；1＝手臂相位照計畫速率走，<1＝底盤比計畫慢、手臂相位被拖慢')
    # 分區降速與完整停止
    i_dec = max(i_slow - 1, 0)
    out['slow_zone'] = dict(v_before_old=round(float(old['v'][max(i_slow - 20, 0)]), 4), v_before_A0=round(float(new['v'][max(i_slow - 20, 0)]), 4), v_after=V_C,
                            dv_old=round(float(old['v'][max(i_slow - 20, 0)] - V_C), 4), dv_A0=round(float(new['v'][max(i_slow - 20, 0)] - V_C), 4),
                            v_at_pre_old=round(float(old['v'][i_pre]), 4), v_at_pre_A0=round(float(new['v'][i_pre]), 4),
                            note='x_pre 處兩者都是 v_c ⇒ 切換點速度條件不變；A0 只改到 x_slow 前的巡航與轉彎')
    vmax_i = int(np.argmax(new['v']))
    out['stop'] = dict(a_dn=A, from_old_ref_v=V_NAV, dist_old=round(V_NAV ** 2 / (2 * A), 4), time_old=round(V_NAV / A, 3),
                       from_A0_vmax=round(float(new['v'][vmax_i]), 4), dist_A0=round(FP.stop_distance(new, vmax_i, A)[0], 4), time_A0=round(FP.stop_distance(new, vmax_i, A)[1], 3),
                       note='完整停止軌跡（沿路徑以 a_dn 減速到 0）：舊參考名義 0.35 m/s 實際輪級到不了；A0 的停止距離對應實際可達速度')
    out['curve'] = dict(sig=[round(float(v), 3) for v in P['sig'][::4]], v_meas=[round(float(v), 4) for v in np.interp(P['sig'], sig_meas, v_meas_s)[::4]],
                        v_old=[round(float(v), 4) for v in old['v'][::4]], v_A0=[round(float(v), 4) for v in new['v'][::4]],
                        v_wheel=[round(float(min(v, 1.0)), 4) for v in new['lim']['v_wheel'][::4]])
    return out


def main():
    res = dict(settings=dict(v_nav=V_NAV, v_c=V_C, a=A, w_max=W_MAX, margin=FP.MARGIN, wheel_cap=FP.wheel_cap()[0]), runs=[one(x) for x in sys.argv[2:]],
               scope='幾何＝實測軌跡；只比較時間參數化；未接控制器、未閉迴路')
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
    for x in res['runs']:
        print('==', x['run'], 'L', x['L'], 'σ_slow', x['sig_slow'], 'σ_pre', x['sig_pre'])
        for k, v in x['rows'].items():
            print('  ', k, json.dumps(v, ensure_ascii=False))
        print('   相位', json.dumps(x.get('phase'), ensure_ascii=False))
        print('   降速', json.dumps({k: v for k, v in x['slow_zone'].items() if k != 'note'}, ensure_ascii=False))
        print('   停止', json.dumps({k: v for k, v in x['stop'].items() if k != 'note'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
