#!/usr/bin/env python3
"""A0 接線的失敗路徑反例（Codex 20261111_195733 四項）。全部呼叫實際 MRO 解析到的方法＋真 S0Judge._stop；不啟動模擬。
  1 不可行 ⇒ 鎖存受控停止（導航剖面、TPA 建立、共同定時失敗、TPAS 進度表），不拋例外。
  2 站位：時間成本非有限的候選不得被選；全部不可行 ⇒ 回 None（由 _standoff 走既有 standoff_none_feasible）。
  3 共同定時失敗不被時間軸縮放掩蓋：Timing.retime 拒絕、tpas_table 回 None。
  4 硬限制是接受閘門：起速 0.30 直線 ⇒ 拒絕（輪級 > 1）；起速略高於工程餘裕但 ≤ 硬界 ⇒ 接受並標記 over_margin。
判定不過 ⇒ 離開碼 1。    python3 evaluation/fprof_failure_paths_check.py OUT.json"""
import inspect
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_feasible_profile as FP                              # noqa: E402
import bottle_heading as HD                                       # noqa: E402
import bottle_standoff as BSO                                     # noqa: E402
import bottle_tlane as TLN                                        # noqa: E402
import bottle_r0_approach as R0                                   # noqa: E402
import bottle_w1_fprof_ctrl as FC                                 # noqa: E402
import bottle_w1_tpas_ctrl as TS                                  # noqa: E402
import obs_ctrl_counterexamples as OX                             # noqa: E402
import bottle_s0_logic as L0                                      # noqa: E402

C = FC.W1PlaceController
FAIL, res = [], {}


def chk(name, cond, info=None):
    res[name] = dict(ok=bool(cond), info=info)
    print(('PASS ' if cond else 'FAIL ') + name, '' if info is None else json.dumps(info, ensure_ascii=False, default=str)[:300], flush=True)
    if not cond:
        FAIL.append(name)


class Straight:
    """直線路徑（σ＝x、y＝0）：最小的路徑物件，介面同 PlannedPath（xy、tangent）。"""
    sig0, L, x_lane = 0.0, 5.0, 0.0

    def xy(self, s):
        s = np.atleast_1d(np.asarray(s, float))
        return np.column_stack([s, np.zeros(len(s))])

    def tangent(self, s):
        s = np.atleast_1d(np.asarray(s, float))
        return np.column_stack([np.ones(len(s)), np.zeros(len(s))])


def ctrl(path, hd, s_ref, v_b, x_pre, x_slow):
    o = C.__new__(C)
    o.judge = L0.S0Judge.__new__(L0.S0Judge)                       # 真 _stop
    o.judge.stop, o.judge.log = None, []
    o.t, o.events = 3.0, []
    o._ev = lambda w, **k: o.events.append((w, k))
    o.path, o._hd_path, o.hd = path, path, hd
    o.s_ref, o.sig_robot, o.v_b, o.x_pre, o.x_slow, o.v_nav, o.v_c = s_ref, s_ref, v_b, x_pre, x_slow, 0.35, 0.10
    o.tpa, o.fp, o._fp_key, o.fp_log, o._fp_timing, o.tlane = None, None, None, [], None, None
    o._yaw_meas = 0.0
    return o


def why(o):
    st = o.judge.stop
    return None if st is None else (st.get('why') if isinstance(st, dict) else str(st))


def main():
    P = Straight()
    hd = HD.HeadingRef(P, 0.0, 0.0)
    # ---- 4 硬限制閘門
    F = FC.build(P, hd, 0.0, 0.30, 3.0, 2.4, 0.35, 0.10)
    chk('4a 起速 0.30（輪級 1.081）⇒ 拒絕', (not F['ok']) and F['why'].startswith('hard_limit') and 'wheel' in F['why'], dict(why=F.get('why'), detail=F.get('detail')))
    F2 = FC.build(P, hd, 0.0, 0.275, 3.0, 2.4, 0.35, 0.10)
    chk('4b 起速 0.275（> 餘裕、≤ 硬界）⇒ 接受並標記', F2['ok'] and F2['over_margin'] and F2['entry_over_cap'] and F2['verify']['wheel_ratio_max'] <= 1.0, F2.get('verify'))
    F3 = FC.build(P, hd, 0.0, 0.0, 3.0, 2.4, 0.35, 0.10)
    chk('4c 名目：不標記、硬限制內', F3['ok'] and not F3['over_margin'] and F3['verify']['alpha_max'] <= HD.ALPHA_MAX and F3['verify']['wheel_ratio_max'] <= FP.MARGIN + 1e-6, F3.get('verify'))
    # ---- 1 受控停止（真方法、真 _stop）
    o = ctrl(P, hd, 0.0, 0.30, 3.0, 2.4)
    impl = os.path.basename(inspect.getsourcefile(C._fp_ensure))
    try:
        r = C._fp_ensure(o)
        chk('1a 導航剖面不可行 ⇒ 鎖存 fprof_infeasible', r is None and why(o) == 'fprof_infeasible', dict(stop=o.judge.stop, impl=impl))
    except Exception as e:                                               # noqa: BLE001
        chk('1a 導航剖面不可行 ⇒ 鎖存 fprof_infeasible', False, repr(e))
    o = ctrl(P, hd, 0.0, 0.30, 3.0, 2.4)
    try:
        tm = C._tpa_timing(o, 0.0)
        chk('1b TPA 建立不可行 ⇒ 鎖存、回 None', tm is None and why(o) == 'fprof_infeasible' and o.judge.stop.get('where') == 'tpa', o.judge.stop)
    except Exception as e:                                               # noqa: BLE001
        chk('1b TPA 建立不可行 ⇒ 鎖存、回 None', False, repr(e))
    # 共同定時失敗：短直線 0.44 m、要求 T＝20 s（最低速度也早到）
    tm = FC.Timing(P, hd, 0.0, 0.0, 0.44, -1.0, 0.35, 0.10)
    vn, vc, ta = tm.retime(40.0)                                     # 0.44 m 以最低速度 0.02 m/s 也只要約 22 s ⇒ 40 s 無法共同定時（20 s 其實可行，首輪測例設錯）
    chk('3a retime：最低速度仍早到 ⇒ 拒絕', (not tm.F.get('ok')) and tm.F.get('why') == 'retime_cannot_meet_T', tm.log[-1])
    Fs = FC.Timing(P, hd, 0.0, 0.0, 0.44, -1.0, 0.35, 0.10).F0
    tk, sk, sc = FC.tpas_table(Fs, 20.0)
    chk('3b tpas_table：T 不符 ⇒ 回 None（不縮放）', tk is None and sk is None and sc > 1.5, dict(scale=round(sc, 3), T_profile=round(Fs['T'], 3)))
    tk, sk, sc = FC.tpas_table(Fs, Fs['T'] + 0.004)
    chk('3c tpas_table：毫秒級殘差 ⇒ 接受、終點對齊', tk is not None and abs(tk[-1] - (Fs['T'] + 0.004)) < 1e-9 and abs(sk[-1] - 0.44) < 1e-9, dict(scale=round(sc, 6)))
    # 1c：真 _tpa_start（fprof 層）＋父鏈以樁取代（只設定 tpa 與失敗的定時物件）；1d：真 _tpas_prof
    o = ctrl(P, hd, 0.0, 0.0, 0.44, -1.0)
    base = TS.W1PlaceController._tpa_start

    def stub(self, q):
        self.tpa = dict(T=40.0)
        self._fp_timing = tm
    TS.W1PlaceController._tpa_start = stub
    try:
        # fprof._tpa_start → super()…：MRO 上 fprof 與 TS 之間的類別沒有覆寫 _tpa_start 的話樁會被叫到；否則改樁最近的父類
        nxt = [k for k in C.__mro__[1:] if '_tpa_start' in k.__dict__][0]
        orig = nxt.__dict__['_tpa_start']
        setattr(nxt, '_tpa_start', stub)
        C._tpa_start(o, np.zeros(9))
        chk('1c 共同定時失敗 ⇒ 鎖存（where＝tpa_retime）', why(o) == 'fprof_infeasible' and o.judge.stop.get('where') == 'tpa_retime', dict(stop=o.judge.stop, stubbed=nxt.__module__))
        setattr(nxt, '_tpa_start', orig)
    except Exception as e:                                               # noqa: BLE001
        chk('1c 共同定時失敗 ⇒ 鎖存（where＝tpa_retime）', False, repr(e))
    finally:
        TS.W1PlaceController._tpa_start = base
    o = ctrl(P, hd, 0.0, 0.0, 0.44, -1.0)
    o._fp_timing = FC.Timing(P, hd, 0.0, 0.0, 0.44, -1.0, 0.35, 0.10)     # 剖面可用但 T 與手臂時長不符
    try:
        prof = C._tpas_prof(o, 0.0, dict(T=20.0, v_nav_eff=0.35, v_c_eff=0.10))
        chk('1d TPAS 進度表 T 不符 ⇒ 鎖存（where＝tpas）', why(o) == 'fprof_infeasible' and o.judge.stop.get('where') == 'tpas', o.judge.stop)
    except Exception as e:                                               # noqa: BLE001
        chk('1d TPAS 進度表 T 不符 ⇒ 鎖存（where＝tpas）', False, repr(e))
    # ---- 2 站位：真 _pick_standoff（obs_ctrl）＋fprof 的 _standoff_t_base
    r3 = json.load(open(os.path.join(HERE, 'runs', 'bottle_motm_tpa_nom_3', 's1_report.json')))
    tl, pp = r3['tlane']['accepted'], r3['planned_path']
    path = TLN.PlannedPathLane(np.array(pp['pts'], float), float(tl['x_entry']), float(tl['y_lane']))
    ox = OX.mk(path, s_ref=-2.9)
    ox.__class__ = C
    ox.v_b = 0.0
    ox.guide_intake = None
    import bottle_s1_geom as G
    legs = [(c.tolist(), sz.tolist()) for n_, c, sz in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]
    q_now = ox._arm_now()
    ok, _, rr = BSO.evaluate(dict(xy=[-0.0227, 0.4081], q_seeds=[q_now.tolist()] + [s.tolist() for s in R0.RNG_SEEDS], legs=legs, y_tube=[-0.04, 0.10]))
    g = dict(cam='v1', xy=[-0.0227, 0.4081])
    best, costs = C._pick_standoff(ox, rr['rows'], g)
    chk('2a A0 成本下正常選到有限成本候選', best is not None and np.isfinite(best['cost']), [(c_['y_reach'], c_.get('cost'), c_.get('t_base')) for c_ in costs])
    C_tb = C._standoff_t_base
    try:
        C._standoff_t_base = lambda self, pn, sg, xp: float('inf')
        best2, costs2 = C._pick_standoff(ox, rr['rows'], g)
        chk('2b 全部成本無限大 ⇒ 不選（None）', best2 is None, [(c_['y_reach'], c_.get('cost')) for c_ in costs2])
        n = [0]

        def one_finite(self, pn, sg, xp):
            n[0] += 1
            return 12.0 if n[0] == 2 else float('inf')
        C._standoff_t_base = one_finite
        best3, costs3 = C._pick_standoff(ox, rr['rows'], g)
        chk('2c 只有一個有限 ⇒ 選它', best3 is not None and np.isfinite(best3['cost']), [(c_['y_reach'], c_.get('cost')) for c_ in costs3])
    finally:
        C._standoff_t_base = C_tb
    res['verdict'] = dict(ok=not FAIL, fails=FAIL)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str)
    print('判定', 'PASS' if not FAIL else 'FAIL', FAIL)
    sys.exit(0 if not FAIL else 1)


if __name__ == '__main__':
    main()
