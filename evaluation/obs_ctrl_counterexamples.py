#!/usr/bin/env python3
"""觀察層（bottle_w1_obs_ctrl）反例與站位成本選擇（Codex 20261111_054341 第 2–4 項）。純離線、方法級（假狀態，不開模擬）。
反例：
  1 視角生成失敗：期限前 ⇒ 保持收攏、記錄、不停止、不採用樣板；期限已到 ⇒ 受控停止 obs_view_infeasible。
  2 站位結果晚到：送出後導引更新（xy 偏 5 mm）⇒ 舊結果丟棄（standoff_stale）、以新導引重送，第二次結果才決定車道。
  3 站位全不可行 ⇒ 受控停止 standoff_none_feasible（不退名目站位）；工作者不可用 ⇒ planner_unavailable。
  4 導引狀態不一致 ⇒ 快取導引失效，不建車道、不重新瞄準。
站位成本：tpa_nom_3 實際接近路徑，承接時 σ（任務 5 s 附近）；bottle_standoff.evaluate 實算候選（含分支構型）→ _pick_standoff（實際橫向接回後底盤時間、手臂轉移時間）。
    python3 evaluation/obs_ctrl_counterexamples.py OUT.json
"""
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_heading as HD  # noqa: E402
import bottle_r0_approach as R0  # noqa: E402
import bottle_standoff as BSO  # noqa: E402
import bottle_tlane as TLN  # noqa: E402
import bottle_view_gen as VG  # noqa: E402
import bottle_w1_gaze_ctrl as GZ  # noqa: E402
import bottle_w1_obs_ctrl as OC  # noqa: E402
import bottle_w1_tpa_ctrl as TPA  # noqa: E402

TPA.W1PlaceController._vstep = lambda self, q: None                    # 隔離：只測觀察層


def mk(path, s_ref=-4.0):
    C = OC.W1PlaceController
    o = C.__new__(C)
    o.K, o.t, o.n_tick = R0.K, 5.0, 1
    o.judge = types.SimpleNamespace(stop=None)
    o.judge._stop = lambda t, why, **k: setattr(o.judge, 'stop', dict(why=why, **k))
    o.vh_seq = dict(ph='TUCK', t0=0.0, T=0.0, qa=R0.Q_TUCK.copy(), qb=R0.Q_TUCK.copy(), t_wait=None)
    o.vh_log, o.vh_wait = [], {}
    o.gz = dict(g=0.0, v=0.0, tgt=0.0, t0=0.0, P=np.zeros(1), why='off')
    o.look = GZ.LOOK.copy()
    o.guide, o.obs_trig, o.guide_n, o.guide_lane, o.so, o.guide_block, o.obs_fail = None, None, 0, None, None, None, []
    o.x_dead = o.x_u0 = -2.654
    o.q_view_tmpl, o.q_vhigh = R0.Q_VHIGH.copy(), R0.Q_VHIGH.copy()
    o.path, o._hd_path = path, path
    o.hd = HD.HeadingRef(path, path.sig0, 0.0)
    o.s_ref = o.sig_robot = s_ref
    o.v_b, o.v_nav, o.v_c = 0.35, 0.35, 0.10
    o.win, o.x_nom = (-0.35, 0.15), 0.0
    o.tlane, o.tlane_log = None, []
    o._rng_seeds = list(R0.RNG_SEEDS)
    o.q_prev2 = np.r_[s_ref, 0.3, 0.0, R0.Q_TUCK]
    o.ev = []
    o._ev = lambda w, **k: o.ev.append((w, k))
    o.lane_calls = []

    def lp(obs, t, sig, d, y_reach=None):
        o.lane_calls.append(y_reach)
        return dict(ok=True, why=None, x_entry=-0.92, y_lane=obs['xy'][1] - (y_reach or 0.4275), sig_join=sig + 0.2, dsig0=0.0, cost_max=1, path_new=path)
    o.lane_planner = lp
    return o


def fake_pc(o, results):
    st = dict(n_sub=0, polls=0, inputs=[])

    def submit(kind, inp):
        st['n_sub'] += 1
        st['inputs'].append(inp)
        return st['n_sub']

    def poll(rid):
        st['polls'] += 1
        return results(rid, st)
    o.pc = types.SimpleNamespace(submit=submit, poll=poll, alive=lambda: True)
    return st


def main():
    out = sys.argv[1]
    r3 = json.load(open(os.path.join(HERE, 'runs', 'bottle_motm_tpa_nom_3', 's1_report.json')))
    tl, pp = r3['tlane']['accepted'], r3['planned_path']
    path = TLN.PlannedPathLane(np.array(pp['pts'], float), float(tl['x_entry']), float(tl['y_lane']))
    q = np.r_[-4.0, 0.3, 0.0, R0.Q_TUCK]
    res = {}
    # 1 視角生成失敗
    orig = VG.view_q1
    VG.view_q1 = lambda *a, **k: (None, dict(why='forced_failure'))
    o = mk(path)
    o.guide_intake(5.0, dict(cam='v1', state='selected', xy=[-0.0227, 0.4081], t_last=4.95, reject=False))
    o.lane_planner = None
    o._vstep(q)
    a = dict(ph=o.vh_seq['ph'], trig=o.obs_trig, n_fail=len(o.obs_fail), stop=o.judge.stop, q_vhigh_is_template=bool(np.allclose(o.q_vhigh, o.q_view_tmpl)))
    q2 = q.copy()
    q2[0] = -2.6
    o._vstep(q2)
    b = dict(ph=o.vh_seq['ph'], stop=o.judge.stop)
    VG.view_q1 = orig
    res['1_view_fail'] = dict(before_deadline=a, at_deadline=b)
    print('1', res['1_view_fail'], flush=True)
    # 2 站位結果晚到
    o = mk(path)
    o.guide_intake(5.0, dict(cam='v1', state='selected', xy=[-0.0227, 0.4081], t_last=4.95, reject=False))

    def results2(rid, st):
        if st['polls'] < 2:
            return None
        return dict(ok=True, error=None, result=dict(rows=[dict(y_reach=0.40, ok=True, margin_min=0.46, base_leg_clear=0.16, q_pre_lo=None)]), wall=dict(t0=0.0, t1=3.7))
    st = fake_pc(o, results2)
    o._lane_from_guide()                                                # 送出 #1
    o.t = 5.05
    o.guide_intake(5.05, dict(cam='w1', state='selected', xy=[-0.0227 + 0.005, 0.4081], t_last=5.0, reject=False))   # 導引更新 5 mm
    o._lane_from_guide()                                                # 輪詢（未到）
    o._lane_from_guide()                                                # 結果到 ⇒ 版本不符 ⇒ 丟棄
    o.t = 5.10
    o._lane_from_guide()                                                # 以新導引重送 #2
    st['polls'] = 0
    for _ in range(3):
        o._lane_from_guide()
    res['2_stale_result'] = dict(n_submit=st['n_sub'], events=[e[0] for e in o.ev], lane_calls=o.lane_calls, so_state=o.so['state'], so_xy=o.so['xy'])
    print('2', res['2_stale_result'], flush=True)
    # 3 全不可行／工作者不可用
    o = mk(path)
    o.guide_intake(5.0, dict(cam='v1', state='selected', xy=[-0.0227, 0.4081], t_last=4.95, reject=False))
    fake_pc(o, lambda rid, st: dict(ok=False, error='no_feasible_standoff', result=dict(rows=[]), wall=dict(t0=0.0, t1=3.0)))
    o._lane_from_guide()
    o._lane_from_guide()
    c1 = dict(stop=o.judge.stop, lane_calls=o.lane_calls)
    o = mk(path)
    o.guide_intake(5.0, dict(cam='v1', state='selected', xy=[-0.0227, 0.4081], t_last=4.95, reject=False))
    o.pc = types.SimpleNamespace(submit=lambda k, i: None, poll=lambda r: None, alive=lambda: False)
    o._lane_from_guide()
    res['3_none_feasible'] = dict(none_feasible=c1, worker_unavailable=dict(stop=o.judge.stop, lane_calls=o.lane_calls))
    print('3', res['3_none_feasible'], flush=True)
    # 4 導引失效
    o = mk(path)
    o.guide_intake(5.0, dict(cam='v1', state='selected', xy=[-0.0227, 0.4081], t_last=4.95, reject=False))
    o.guide_intake(5.02, dict(cam=None, state='inconsistent', grasp_state='inconsistent', reject=False))
    st = fake_pc(o, lambda rid, st: None)
    o._lane_from_guide()
    res['4_guide_block'] = dict(guide=o.guide, guide_block=o.guide_block, n_submit=st['n_sub'], lane_calls=o.lane_calls, events=[e[0] for e in o.ev])
    print('4', res['4_guide_block'], flush=True)
    # 站位成本：實算候選 → 控制端選擇
    o = mk(path, s_ref=-2.9)
    o.guide_intake(5.0, dict(cam='v1', state='selected', xy=[-0.0227, 0.4081], t_last=4.95, reject=False))
    q_now = o._arm_now()
    import bottle_s1_geom as G
    legs = [(c.tolist(), sz.tolist()) for n_, c, sz in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]
    w0 = time.time()
    ok, why, rr = BSO.evaluate(dict(xy=[-0.0227, 0.4081], q_seeds=[q_now.tolist()] + [s.tolist() for s in R0.RNG_SEEDS], legs=legs, y_tube=[-0.04, 0.10]))
    wall = round(time.time() - w0, 2)
    best, costs = o._pick_standoff(rr['rows'])
    res['standoff_cost'] = dict(worker_ok=ok, worker_wall_s=wall, worker_rows=[{k: v for k, v in r.items() if k != 'q_pre_lo'} for r in rr['rows']],
                                costs=[{k: v for k, v in c.items() if k != 'q_pre_lo'} for c in costs], chosen=None if best is None else {k: v for k, v in best.items() if k != 'q_pre_lo'})
    print('成本', [(c['y_reach'], c.get('cost'), c.get('t_base'), c.get('t_arm'), c.get('dsig0'), c['margin_min']) for c in costs], '選', None if best is None else best['y_reach'], 'wall', wall, flush=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=str)


if __name__ == '__main__':
    main()
