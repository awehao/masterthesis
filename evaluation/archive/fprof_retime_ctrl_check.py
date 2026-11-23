#!/usr/bin/env python3
"""手臂來不及 ⇒ 共同重新定時：控制器層反例（Codex 20261112 第 17 輪）。不啟動模擬。
真 dpred 控制器類別的 _tpa_timing（建 Timing）→ 真 bottle_w1_tpa_ctrl.plan_tpa（控制器 _tpa_start 用的同一函式，含終點 IK、關節軌跡、retime、必要時重規劃）→ 真 _tpas_prof。
情境：直線車道、底盤已在 x_pre 前 0.30 m、速度 0.10 m/s（底盤名目約 3 s 到位），手臂由收攏姿態起（需要更久）。
核：retime 種類為 retimed；最終剖面到位時間＝手臂時長（差 ≤ 0.01 s）；v_nav'／v_c' 低於名目；關節速度／加速度比 ≤ 1；進度表終點＝x_pre、終點時間＝T、時間軸比例≈1；剖面硬限制通過；無停止。
另一案：底盤只剩 0.05 m（最低速度也早到）⇒ 受控停止 fprof_infeasible（不縮放、不退回舊剖面）。    python3 evaluation/fprof_retime_ctrl_check.py OUT.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_heading as HD                                       # noqa: E402
import bottle_plan_worker as PW                                   # noqa: E402
import bottle_r0_approach as R0                                   # noqa: E402
import bottle_ref as BR                                           # noqa: E402
import bottle_s0_ctrl as C0                                       # noqa: E402
import bottle_w1_dpred_ctrl as DP                                 # noqa: E402
import bottle_w1_fprof_ctrl as FC                                 # noqa: E402
import bottle_w1_tpa_ctrl as TPA                                  # noqa: E402
import fprof_failure_paths_check as FF                            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L         # noqa: E402

C = DP.W1PlaceController
FAIL, res = [], {}


def chk(n, c, info=None):
    res[n] = dict(ok=bool(c), info=info)
    print(('PASS ' if c else 'FAIL ') + n, '' if info is None else json.dumps(info, ensure_ascii=False, default=float)[:420], flush=True)
    if not c:
        FAIL.append(n)


def case(back):
    xb, yb, zs, yl = 0.0, 0.4275, 0.58, 0.0
    x_pre = xb - 0.35 + 0.05
    P = FF.Straight()
    P.sig0 = -5.0
    hd = HD.HeadingRef(P, x_pre - 2.0, 0.0)
    qa0 = np.asarray(R0.Q_TUCK, float)
    T_g = C0.grasp_target(xb, yb, zs + 0.10)
    T_pre = BR.offset_along_tool(T_g, -0.06)
    ok, why, rr = PW.plan_grasp(dict(xy=[xb, yb], z_support=zs, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0, y_lane=yl,
                                     q_seeds=[s.tolist() for s in [qa0] + list(R0.RNG_SEEDS)], q_now=qa0.tolist(), q_seed=None, side='left'))
    assert ok, why
    sig = x_pre - back
    o = FF.ctrl(P, hd, sig, 0.10, x_pre, x_pre - 0.05)
    o.slow_back, o.dp_stat = 0.05, {}
    o._fp_sig_e = sig
    tm = C._tpa_timing(o, sig)
    if tm is None:
        return dict(stop=o.judge.stop, stage='timing')
    R, why = TPA.plan_tpa(R0.K, qa0, np.zeros(6), np.array([x_pre, yl, 0.0]), T_pre, np.asarray(rr['q_pre_lo'], float), sig, x_pre, x_pre - 0.05, 0.35, 0.10, 0.10, R0.IDX, L.lower, L.upper, timing=tm)
    if R is None:
        return dict(stop=None, stage='plan', why=why)
    Pd = dict(T=R['P']['T'], v_nav_eff=R['v_nav_eff'], v_c_eff=R['v_c_eff'])
    o._fp_timing = tm
    if not tm.F.get('ok'):                                               # 同 _tpa_start 的處置
        o.judge._stop(o.t, 'fprof_infeasible', reason=str(tm.F.get('why')), where='tpa_retime')
        return dict(stop=o.judge.stop, stage='retime', log=tm.log[-1], T_arm=R['P']['T'], T_base=tm.t_base())
    tk, sk = C._tpas_prof(o, sig, Pd)
    return dict(stop=o.judge.stop, stage='ok', T_arm=float(R['P']['T']), T_base=float(tm.t_base()), T_final=float(tm.F['T']), log=tm.log, v_nav_eff=R['v_nav_eff'], v_c_eff=R['v_c_eff'], replan=R['replan'],
                info={k: R['info'].get(k) for k in ('v_ratio', 'a_ratio', 'joint_margin')}, verify=tm.F.get('verify'), tk_end=float(tk[-1]), sk_end=float(sk[-1]), x_pre=x_pre, scale=Pd.get('fprof', {}).get('tau_scale'),
                qd1_max=float(np.abs(R['qd1']).max()))


def main():
    a = case(0.30)
    chk('重新定時觸發（手臂時長 > 底盤名目時間）', a['stage'] == 'ok' and a['T_arm'] > a['T_base'] + 0.2 and a['log'][-1]['kind'] == 'retimed', {k: a.get(k) for k in ('stage', 'T_arm', 'T_base', 'T_final', 'log', 'stop', 'why')})
    if a['stage'] == 'ok':
        chk('最終剖面到位時間＝手臂時長（≤0.01 s）、速度降低', abs(a['T_final'] - a['T_arm']) <= 0.01 and (a['v_nav_eff'] < 0.35 or a['v_c_eff'] < 0.10), dict(T_final=a['T_final'], T_arm=a['T_arm'], v_nav_eff=a['v_nav_eff'], v_c_eff=a['v_c_eff'], replan=a['replan']))
        chk('關節速度／加速度比 ≤ 1、剖面硬限制內', a['info']['v_ratio'] <= 1.0 and a['info']['a_ratio'] <= 1.0 and a['verify']['wheel_ratio_max'] <= 1.0, dict(info=a['info'], verify=a['verify'], qd1_max=a['qd1_max']))
        chk('進度表：終點＝x_pre、終點時間＝T、比例≈1、無停止', abs(a['sk_end'] - a['x_pre']) < 1e-9 and abs(a['tk_end'] - a['T_arm']) < 1e-9 and abs(a['scale'] - 1.0) < 2e-3 and a['stop'] is None, dict(sk_end=a['sk_end'], tk_end=a['tk_end'], scale=a['scale']))
    b = case(0.05)
    chk('最低速度仍早到 ⇒ 受控停止（不縮放）', b['stage'] in ('retime', 'timing') and (b['stop'] or {}).get('why') == 'fprof_infeasible', {k: b.get(k) for k in ('stage', 'stop', 'log', 'T_arm', 'T_base')})
    res['verdict'] = dict(ok=not FAIL, fails=FAIL)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
    print('判定', 'PASS' if not FAIL else 'FAIL', FAIL)
    sys.exit(0 if not FAIL else 1)


if __name__ == '__main__':
    main()
