#!/usr/bin/env python3
"""偏差前推原型：真參考鏈反例（Codex 20261111 第 13 輪第 1 項）。不啟動模擬。
走實際 MRO 解析到的 _T_seq（tpas→liftb→w1→s1）→ dpred._rel_target → 真 _d_rates／d_at；底盤以等加速度橫移的合成運動推進多個控制拍，
每拍照控制器做法更新 P['D']、u_last、τ，再取時域參考序列 T_seq（21 步，TCP 相對底盤）。量：
  同拍預覽內的參考速度／加速度；跨拍：第 k 拍預覽第 2 步（+0.1 s）對第 k+1 拍（0.1 s 後）當拍值的位置差、當拍參考速度的跨拍變化（換算加速度）。
情境：A 等速橫移過零；B 橫向加速度 0.4 m/s²；C 橫移速度升過適用上限再降回（進入／退出常數 D 回退）；D 同 B 但常數 D（對照，lateslow 類別）。
判準（事前定）：跨拍參考位置跳動 ≤ 2 mm（切換位置門檻 5 mm 的 40%）；跨拍參考速度變化換算加速度 ≤ 1.0 m/s²；進入／退出回退的速度跳動另列，
  r1：退回常數 D 的進出瞬間參考速度跳 0.095 m/s > SW_V ⇒ 回退不是無縫；r2 起改為超出適用域即受控停止，本檔核停止時刻。
    python3 evaluation/dpred_realchain_check.py OUT.json"""
import inspect
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_r0_approach as R0                                   # noqa: E402
import bottle_s0_ctrl as C0                                       # noqa: E402
import bottle_s0b_ctrl as B                                       # noqa: E402
import bottle_w1_dpred_ctrl as DP                                 # noqa: E402
import bottle_w1_lateslow_ctrl as LS                              # noqa: E402
import bottle_w1_tpas_ctrl as TS                                  # noqa: E402

TICK, LAT = float(os.environ.get('DP_TICK', str(C0.NPH * C0.PHYS))), C0.D_STEPS * C0.PHYS   # 控制拍＝NPH·PHYS（0.05 s）；DP_TICK=0.10 保留為壓力案例（r1／r2 誤用 0.10）
POS_TOL, ACC_TOL = 0.002, 1.0
TAU0 = float(os.environ.get('DP_TAU0', '5.2'))                    # 5.2＝切換點之後（τ>T）；DP_TAU0=2.0＝接近中段（τ<T，rel_ref 走關節軌跡）
FAIL, res = [], {}


def mk(cls):
    o = cls.__new__(cls)
    o.K = R0.K
    o.st = types.SimpleNamespace(phase='CRUISE', t_phase=0.0, t_pre=float('inf'))
    o._xt_prev = None
    o.blending, o.no_tcp_preview, o.preview, o.w_goal, o.lb = False, False, {}, None, None
    o.hd, o.s_ref = None, 0.0
    import bottle_s0_logic as L0
    o.judge = L0.S0Judge.__new__(L0.S0Judge)
    o.judge.stop, o.judge.log = None, []
    o.dp_stat = dict(n=0, n_clip=0, dy_rate_max=0.0, dth_rate_max=0.0)
    T_pre = np.eye(4)
    T_pre[:3, 3] = [0.10, 0.45, 0.70]
    import bottle_w1_tpa_ctrl as TPAm
    from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as LL
    q0, q1 = np.asarray(R0.Q_TUCK, float), np.asarray(R0.Q_VHIGH, float)
    PJ, _ = TPAm.plan_joint(q0, np.zeros(6), q1, np.zeros(6), 5.0, B.W_ARM['v_arm'], C0.A_ARM, LL.lower, LL.upper)
    o.tpa = dict(switched=False, P=PJ, T=float(PJ['T']), tau=TAU0, rate=1.0, t_last=0.0, base_end=np.array([-0.30, 0.0, 0.0]), v_c_eff=0.10, T_pre=T_pre, D=None, sig_raw=0.0)
    o.t = 0.0
    return o


def run(name, cls, vy_fn, n=14, y0=-0.02):
    o = mk(cls)
    C = cls
    impl = dict(T_seq=os.path.basename(inspect.getsourcefile(C._T_seq)), rel=os.path.basename(inspect.getsourcefile(C._rel_target)))
    y, rows, prev = y0, [], None
    for k in range(n):
        t = k * TICK
        vy = float(vy_fn(t))
        # 控制器每拍：D＝命令生效時刻的預測偏差（在途延遲內照上一命令走）；u_last＝這拍送出的命令
        dy_eff = y + vy * LAT
        o.t, o.tpa['t_last'] = t, t
        o.tpa['tau'] = TAU0 + t
        o.tpa['D'] = TS._tb(np.r_[0.0, dy_eff, 0.0])
        o.tpa.pop('Dd', None)
        o.u_last = np.r_[0.10, vy, 0.0, np.zeros(6)]
        if cls is DP.W1PlaceController:
            o.tpa['Dd'], o.tpa['Dd_t'] = C._d_rates(o, o.tpa), t                 # 同控制器：變化率在 _advance（求解前）算好
        if cls is DP.W1PlaceController and o.judge.stop is None:
            C._xtick_gate(o)                                                     # 同控制器：求解前的 2 mm 跨拍檢查
        if o.judge.stop is not None:
            rows.append(dict(t=round(t, 2), vy=round(vy, 4), stopped=o.judge.stop.get('why'), fallback=False))
            break
        Tq = C._T_seq(o, np.zeros(9), t + LAT)
        p = Tq[:, :3, 3]
        v_in = np.diff(p, axis=0) / B.DT
        a_in = np.abs(np.diff(v_in, axis=0)).max() / B.DT
        row = dict(t=round(t, 2), vy=round(vy, 4), dy=round(dy_eff, 4), ref_y=round(float(p[0, 1]), 5), v_ref_y=round(float(v_in[0, 1]), 4), a_preview=round(float(a_in), 4),
                   fallback=bool(isinstance(o.tpa.get('Dd'), tuple) and o.tpa['Dd'] == (0.0, 0.0) and abs(vy) > 1e-9), stopped=None if o.judge.stop is None else o.judge.stop.get('why'))
        if o.judge.stop is not None:
            rows.append(row)
            break
        if prev is not None:
            row['jump_pos_mm'] = round(float(np.linalg.norm(prev['pn'] - p[0])) * 1e3, 3)       # 「同一絕對時刻的預覽更新差」：上一拍預覽在 +TICK 的值 對 這一拍當拍值（不是實際參考速度）
            row['dv_cross'] = round(float(np.linalg.norm(v_in[0] - prev['v0'])), 4)
            row['a_cross'] = round(row['dv_cross'] / TICK, 3)
        kk = int(round(TICK / B.DT))
        prev = dict(pn=p[kk].copy(), v0=v_in[0].copy())
        rows.append(row)
        y += vy * TICK + 0.5 * (float(vy_fn(t + TICK)) - vy) * TICK        # 梯形積分
    jp = max((r.get('jump_pos_mm', 0.0) for r in rows))
    ac = max((r.get('a_cross', 0.0) for r in rows))
    dvc = max((r.get('dv_cross', 0.0) for r in rows))
    fb = [r['t'] for r in rows if r['fallback']]
    out = dict(name=name, impl=impl, jump_pos_max_mm=jp, a_cross_max=ac, dv_cross_max=dvc, a_preview_max=max(r.get('a_preview', 0.0) for r in rows), fallback_ticks=fb, n_clip=o.dp_stat['n_clip'] if cls is DP.W1PlaceController else None, rows=rows)
    return out


def chk(n, c, info=None):
    res.setdefault('checks', {})[n] = dict(ok=bool(c), info=info)
    print(('PASS ' if c else 'FAIL ') + n, '' if info is None else json.dumps(info, ensure_ascii=False, default=float)[:300], flush=True)
    if not c:
        FAIL.append(n)


def main():
    P_ = DP.W1PlaceController
    A = run('A 等速過零', P_, lambda t: 0.04, y0=-0.02)
    Bc = run('B 橫向加速 0.4', P_, lambda t: min(0.4 * t, 0.09), y0=-0.05)
    Cc = run('C 升過上限再降回', P_, lambda t: 0.06 + 0.07 * math.sin(math.pi * min(t, 1.2) / 1.2), y0=-0.08)
    Dc = run('D 常數 D 對照（同 B）', LS.W1PlaceController, lambda t: min(0.4 * t, 0.09), y0=-0.05)
    res['cases'] = [A, Bc, Cc, Dc]
    for x in (A, Bc, Cc, Dc):
        print('==', x['name'], x['impl'], {k: x[k] for k in ('jump_pos_max_mm', 'dv_cross_max', 'a_cross_max', 'a_preview_max', 'fallback_ticks', 'n_clip')})
    chk('MRO：_T_seq＝tpas 檔、_rel_target＝dpred 檔', A['impl'] == dict(T_seq='bottle_w1_tpas_ctrl.py', rel='bottle_w1_dpred_ctrl.py'), A['impl'])
    zc = [r for r in A['rows'] if r.get('dy', -1) > 0]
    chk('A 過零：跨拍位置跳動 ≤ 2 mm、速度無跳動', A['jump_pos_max_mm'] <= POS_TOL * 1e3 and A['a_cross_max'] <= ACC_TOL and len(zc) > 3 and not A['fallback_ticks'], dict(jump=A['jump_pos_max_mm'], a=A['a_cross_max'], n_after_zero=len(zc)))
    lastB = Bc['rows'][-1]
    chk('B 橫向加速 0.4（2.1 mm 壓力輸入）⇒ 求解前受控停止 dpred_preview_jump，之後不再產生參考', lastB.get('stopped') == 'dpred_preview_jump' and 'ref_y' not in lastB,
        dict(t_stop=lastB['t'], stop=lastB.get('stopped'), n_ticks_before=len(Bc['rows']) - 1, jump_before_max_mm=Bc['jump_pos_max_mm']))
    first_over = next((r for r in Cc['rows'] if abs(r['vy']) > DP.DY_RATE_MAX), None)
    chk('C 速率超出適用域 ⇒ 該拍即受控停止（不做非無縫回退）', Cc['rows'][-1].get('stopped') == 'dpred_rate_out_of_domain' and first_over is not None and Cc['rows'][-1]['t'] == first_over['t'],
        dict(t_stop=Cc['rows'][-1]['t'], vy=Cc['rows'][-1]['vy'], limit=DP.DY_RATE_MAX, r1_finding='退回常數 D 的進出瞬間參考速度跳 0.095 m/s > SW_V 0.03'))
    res['baseline_const_D'] = dict(jump_pos_max_mm=Dc['jump_pos_max_mm'], a_cross_max=Dc['a_cross_max'], note='常數 D（既有行為）在同一橫向加速情境下的跨拍值，作對照')
    res['setup'] = dict(tick=TICK, tau0=TAU0, T=float(mk(P_).tpa['T']), note='D 以 y＋v·延遲 合成（未走真在途管線 _stop_rest）；真 step 的核對在假世界注入案')
    res['verdict'] = dict(ok=not FAIL, fails=FAIL, criteria=dict(pos_mm=POS_TOL * 1e3, acc=ACC_TOL, sw_v=TS.SW_V))
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
    print('判定', 'PASS' if not FAIL else 'FAIL', FAIL)
    sys.exit(0 if not FAIL else 1)


if __name__ == '__main__':
    main()
