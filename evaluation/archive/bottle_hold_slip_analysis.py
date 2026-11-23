#!/usr/bin/env python3
"""夾持保持（滑移）短分析（Codex 20261102_031502 (a)；只讀既有紀錄，不啟動模擬）。三趟：bottle_w1_vhigh_z060_p1_1、bottle_w1_segtarget_nominal1、bottle_w1_compact3_nominal1。
1 基準＝判定器抓持成立時刻 t_grasp 的夾持相對位姿 T_rel0＝inv(T_tcp)·T_bottle（量測關節 FK、瓶位姿讀回）；之後各時刻 Δp＝T_rel(t).p − T_rel0.p，
  以 TCP 座標分量報（z＝工具軸／插入方向、x、y），另報範數、各段滑移率（mm/s）與兩指實測開度、兩指對瓶受力。
  分段：抓持成立→抬升完成（LIFT）、持瓶行進（CARRY_NAV）、放置接近（PLACE_APPROACH）、下降（LOWER）、鬆爪前；本趟另分停止前、停止減速中（stop→停止有效判定時刻）、停穩後。
2 閉爪時（close_request、t_grasp）：名目插入深度（報告 grasp_depth_m，設定值）與實測插入深度（＝瓶軸在 TCP 座標的工具軸分量取負：TCP 越過瓶軸的距離）、
  瓶軸在 TCP 座標的偏心、TCP 在瓶座標的位置（inv(T_rel) 直接取；z＝抓持高度，瓶底起算）、兩指受力。
3 材質與驅動設定：資產 yaml 的摩擦設定值、手指驅動增益（讀檔，不推斷實際碰撞面綁定）。
輸出 results/bottle/hold_slip_analysis.json。
"""
import json
import os
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402

RUNS = ('bottle_w1_vhigh_outslow_z060_p1_1', 'bottle_w1_vhigh_hold_z060_p1_1', 'bottle_w1_vhigh_z060_p1_1', 'bottle_w1_segtarget_nominal1', 'bottle_w1_compact3_nominal1')
ASSET = os.path.join(os.path.dirname(HERE), 'src', 'my_omnibot_description', 'config', 'bottle_s0.yaml')


def load(run):
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    return c, json.load(open(os.path.join(D, 's1_report.json')))


def rel(c, k):
    q = np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]] + [c[f'q{j}'][k] for j in range(1, 7)])
    Tt = G.C.K.fk(q, 'link_tcp')
    Tb = np.eye(4)
    Tb[:3, :3] = Rot.from_quat([c['bottle_qx'][k], c['bottle_qy'][k], c['bottle_qz'][k], c['bottle_qw'][k]]).as_matrix()
    Tb[:3, 3] = [c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]
    return np.linalg.inv(Tt) @ Tb


def one(run):
    c, r = load(run)
    t = c['t']
    tg = float(r['t_grasp'])
    k0 = int(np.argmin(np.abs(t - tg)))
    R0 = rel(c, k0)
    ev = {}
    for e in r['events']:
        if e['what'] == 'phase':
            ev.setdefault(e['to'], e['t'])
        elif e['what'] == 'place_phase':
            ev.setdefault('P_' + e['to'], e['t'])
        elif e['what'] in ('close_request', 'stop'):
            ev.setdefault(e['what'], e['t'])
    st = r.get('stop') or {}
    se = r.get('stop_effect') or {}
    t_end = float(t[-1])
    hx = {e['what']: e['t'] for e in r['events'] if e['what'] in ('hold_settle_start', 'hold_start', 'hold_end')}
    bounds = [('抓持→抬升完成', tg, ev.get('CARRY', t_end))]
    if 'hold_start' in hx:
        bounds += [('懸持前穩定過渡', ev.get('CARRY', t_end), hx['hold_start']), ('靜態懸持', hx['hold_start'], hx.get('hold_end', t_end)),
                   ('懸持後到持瓶行進', hx.get('hold_end', t_end), ev.get('P_CARRY_NAV', t_end))]
    bounds += [('持瓶行進', ev.get('P_CARRY_NAV', t_end), ev.get('P_PLACE_APPROACH', t_end)),
              ('放置接近', ev.get('P_PLACE_APPROACH', t_end), ev.get('P_LOWER', t_end)), ('下降', ev.get('P_LOWER', t_end), min(ev.get('P_RELEASE', t_end), st.get('t', t_end)))]
    if st.get('t') is not None:
        t_se = float(se.get('t', st['t'] + 2.0))
        bounds += [('停止減速中', st['t'], t_se), ('停穩後', t_se, t_end)]

    def at(tt):
        k = int(np.argmin(np.abs(t - tt)))
        dp = (rel(c, k)[:3, 3] - R0[:3, 3]) * 1e3
        return k, dp
    segs = []
    for nm, a, b in bounds:
        if b <= a:
            continue
        ka, da = at(a)
        kb, db = at(b)
        f1, f2 = c['f_finger1_bottle'][ka:kb + 1], c['f_finger2_bottle'][ka:kb + 1]
        segs.append(dict(seg=nm, t0=round(a, 2), t1=round(b, 2), d0_mm=round(float(np.linalg.norm(da)), 2), d1_mm=round(float(np.linalg.norm(db)), 2),
                         dvec_tcp_mm=[round(float(v), 2) for v in (db - da)], rate_mm_s=round(float(np.linalg.norm(db - da)) / (b - a), 3),
                         finger_q=[round(float(np.mean(c['finger1_q'][ka:kb + 1])), 5), round(float(np.mean(c['finger2_q'][ka:kb + 1])), 5)],
                         finger_f_N=[round(float(np.mean(f1)), 2), round(float(np.mean(f2)), 2)], finger_f_min_N=[round(float(np.min(f1)), 2), round(float(np.min(f2)), 2)]))
    kc = int(np.argmin(np.abs(t - ev.get('close_request', tg))))
    Rc = rel(c, kc)
    tcp_in_b = np.linalg.inv(R0)[:3, 3]                     # TCP 在瓶座標（瓶底中心為原點、z 沿瓶軸向上）
    grasp = dict(t_close=ev.get('close_request'), t_grasp=tg, grasp_depth_nominal_m=r.get('grasp_depth_m'),
                 bottle_axis_in_tcp_at_grasp_mm=[round(float(v) * 1e3, 2) for v in R0[:3, 3]],
                 insertion_measured_mm=round(float(-R0[2, 3]) * 1e3, 2),
                 tcp_in_bottle_mm=[round(float(v) * 1e3, 2) for v in tcp_in_b], tcp_height_on_bottle_mm=round(float(tcp_in_b[2]) * 1e3, 2),
                 at_close_rel_mm=[round(float(v) * 1e3, 2) for v in Rc[:3, 3]], finger_f_at_grasp_N=[round(float(c['f_finger1_bottle'][k0]), 2), round(float(c['f_finger2_bottle'][k0]), 2)],
                 finger_q_at_grasp=[round(float(c['finger1_q'][k0]), 5), round(float(c['finger2_q'][k0]), 5)])
    kend = len(t) - 1
    return dict(run=run, stop=st or None, slip_max_judge=r.get('slip_max_mm_deg'), final_d_mm=round(float(np.linalg.norm((rel(c, kend)[:3, 3] - R0[:3, 3]) * 1e3)), 2), grasp=grasp, segments=segs)


def main():
    out = dict(runs=[one(r) for r in RUNS])
    A = yaml.safe_load(open(ASSET))
    out['settings'] = dict(asset=os.path.relpath(ASSET, os.path.dirname(HERE)), bottle={k: A.get('bottle', {}).get(k) for k in A.get('bottle', {})},
                           physics={k: v for k, v in (A.get('physics') or {}).items()}, note='摩擦設定值讀自資產 yaml；各實際碰撞面的有效綁定未在此核實；接觸力 API 不含切向力，不能據此判摩擦飽和')
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'hold_slip_analysis.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for R in out['runs']:
        print(R['run'], '停止', R['stop'] and R['stop'].get('why'), '最終 d', R['final_d_mm'], 'mm；閉爪', R['grasp'])
        for s in R['segments']:
            print('   ', s)
    print('設定', out['settings'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
