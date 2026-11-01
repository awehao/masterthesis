#!/usr/bin/env python3
"""多任務 v2w（bottle_w1_vhigh_relwin_slipwarn_ctrl）針對性核對（Codex 20261102_055603；假世界、假候選，非物理）。
幾何用 A4（第一桌 0.60、瓶 (+0.08, 0.46)），假候選同 test_bottle_w1_relwin_uadopt（U 段開始後 3.6 s 起送，物理 A1／A2 實測時機）。
注入在假受控對象（持瓶相對位姿 Trel、手指力、禁止接觸量測），控制器只看量測：
  nominal   不注入；同條件再跑 v2（v2nom）對照，核退化（相位時刻、完成、放置）
  trans     離桌成立起瓶沿 TCP −x（物理 A1–A3 實測方向，世界向下）以 0.5 mm/s 平移，上限 10 mm ⇒ 期望：警告、不停、警告後仍走完放置／鬆爪／保持、hold_ok False、原標準不成功、slip_max > 5 mm
            （第一版判準寫 slip_max ≥ 9.9 mm 是錯的：鬆爪前只滑到約 7.8 mm）
  angle     離桌成立起繞 TCP y 以 0.5°/s 轉 ⇒ 期望：約 6 s 後轉角 > 3° 照停（slip）
  trans_ang 離桌成立起以 1 mm/s 平移（上限 10 mm），離桌成立後 7 s 起再加 1°/s 轉角 ⇒ 期望：先警告、後轉角照停（警告不關掉後續停止）
            （第一版轉角排在離桌後 25 s，晚於鬆爪，未生效）
  loss      離桌成立後 5 s 起手指 1 力 0 ⇒ 期望 grasp_loss_finger1
  contact   離桌成立後 5 s 起 bottle_env 禁止接觸 1 N ⇒ 期望 forbidden_contact:bottle_env
另報持瓶對兩桌最小距離：假受控對象實際瓶位姿（含注入滑移），離桌成立起到放置下放（LOWER）開始，瓶身取樣點對第一桌與第二桌（桌面＋桌腳）方塊；放置量測另列。
    python3 evaluation/test_bottle_w1_slipwarn.py all
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'bottle', 'slipwarn_offline.json')
MODES = ('nominal', 'v2nom', 'trans', 'angle', 'trans_ang', 'loss', 'contact')

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=3600)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return k, (json.loads(line[-1]) if line else dict(mode=k, error=p.stderr[-2000:]))
    with ThreadPoolExecutor(3) as ex:
        res = dict(ex.map(one, MODES))
    a, b = res['nominal'], res['v2nom']
    res['degeneracy'] = dict(same_phases=a.get('phases') == b.get('phases'), t_done=[a.get('t_done'), b.get('t_done')], place_mm=[a.get('place_mm'), b.get('place_mm')])
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in MODES:
        r = res[k]
        print(k, {kk: r.get(kk) for kk in ('pass', 'expect', 'stop', 'mission_done', 'hold_ok', 'success', 'slip_max', 'warn', 'held_min_mm', 'place_mm', 't_done', 'error')})
    print('退化', res['degeneracy'])
    sys.exit(0)

MODE = sys.argv[1]
Z1, DXB, DYB, D_U = 0.60, 0.08, 0.04, 3.6
sys.argv = [sys.argv[0], f'{Z1:.2f}', 'nom']
sys.path.insert(0, HERE)
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = 0.0 + DXB, 0.42 + DYB
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import bottle_w1_vhigh_relwin_ctrl as RW                # noqa: E402
import bottle_w1_vhigh_relwin_slipwarn_ctrl as SW       # noqa: E402

T = TV.T
TB_LOG = []


def rot_y(deg):
    a = np.radians(deg)
    R = np.eye(4)
    R[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    return R


def run():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = RW.W1PlaceController if MODE == 'v2nom' else SW.W1PlaceController
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, Trel0=None, t0=None)
    step0, meas0 = c.step, f.meas

    def meas(**ov):
        J = c.judge
        if f.grasped and not getattr(f, 'released', False) and J.t_lift_ok is not None and MODE in ('trans', 'angle', 'trans_ang'):
            if st['Trel0'] is None:
                st['Trel0'], st['t0'] = np.array(f.Trel, float).copy(), float(J.t_lift_ok)
            dt_ = f.t - st['t0']
            Tr = st['Trel0'].copy()
            if MODE in ('trans', 'trans_ang'):
                Tr[0, 3] -= min((0.0005 if MODE == 'trans' else 0.001) * dt_, 0.010)
            if MODE == 'angle':
                Tr = Tr @ rot_y(0.5 * dt_)
            if MODE == 'trans_ang' and dt_ >= 7.0:
                Tr = Tr @ rot_y(1.0 * (dt_ - 7.0))
            f.Trel = Tr
        m = meas0(**ov)
        TB_LOG.append((float(f.t), np.array(f.Tb, float).copy()))
        if J.t_lift_ok is not None and f.t >= J.t_lift_ok + 5.0:
            if MODE == 'loss':
                m['f_fingers'] = (0.0, m['f_fingers'][1])
            if MODE == 'contact':
                m['forbidden'] = dict(m['forbidden'], bottle_env=1.0)
        return m
    f.meas = meas

    def step(t, q, **kw):
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if c.vh_seq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + D_U - 1e-9:
                xy = np.array([T.T.BX, T.T.BY]) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
        return step0(t, q, **kw)
    c.step = step
    r, _ = T.run(c, f, t_end=75.0)
    return c, f, r


EXPECT = dict(nominal='完成、無警告、原標準成功', v2nom='完成（v2 對照）', trans='警告、不停、完成放置、hold_ok False、原標準不成功',
              angle='轉角 > 3° 照停（slip）', trans_ang='先警告、後轉角照停', loss='grasp_loss_finger1 照停', contact='forbidden_contact:bottle_env 照停')


def main():
    c, f, r = run()
    why = (r['stop'] or {}).get('why')
    done = bool(r.get('mission_done'))
    sw = r.get('slip_warn') or {}
    w = sw.get('warn')
    ph = {}
    for e in r['events']:
        if e['what'] in ('phase', 'place_phase'):
            ph.setdefault(e['to'], e['t'])
    mins, sm, ns, rate, marg = PCK.check(f)
    t_lok, t_low = r.get('t_lift_ok'), ph.get('LOWER')
    boxes = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes()] + [('t2:' + n, c_, s_) for n, c_, s_ in TP.table2_boxes()]
    BP = G.bottle_points()
    held = {}
    if t_lok is not None:
        t_hi = t_low if t_low is not None else ((r['stop'] or {}).get('t') or 1e9)
        for tt, Tb in TB_LOG[::5]:
            if t_lok + 0.2 <= tt <= t_hi:
                P = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
                for n, c_, s_ in boxes:
                    d = float(G.box_dist(P, c_, s_).min())
                    if d < held.get(n, (9.9,))[0]:
                        held[n] = (round(d * 1e3, 1), round(tt, 2))
    hv = [v[0] for v in held.values()]
    pm = (r.get('place') or {}).get('metrics') or {}
    hold_ok = (r.get('result') or {}).get('hold_ok')
    succ = bool(r.get('success'))
    sl = r['slip_max_mm_deg']
    ok = {'nominal': done and w is None and succ, 'v2nom': done,
          'trans': done and w is not None and hold_ok is False and not succ and sl[0] > 5.0 and why is None and w['t_first'] < (ph.get('RELEASE') or 0.0),
          'angle': why is not None and why.startswith('slip') and sl[1] > 3.0 and not done,
          'trans_ang': w is not None and why is not None and why.startswith('slip') and sl[1] > 3.0 and not done,
          'loss': why == 'grasp_loss_finger1', 'contact': why == 'forbidden_contact:bottle_env'}[MODE]
    out = dict(mode=MODE, expect=EXPECT[MODE], **{'pass': bool(ok)}, stop=r['stop'], mission_done=done, hold_ok=hold_ok, success=succ, slip_max=[round(sl[0], 2), round(sl[1], 2)],
               warn=None if w is None else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items()}, done_quality=sw.get('done_quality'),
               phases=ph, t_done=next((e['t'] for e in r['events'] if e['what'] == 'mission_done'), None), place_mm=round(pm['offset'] * 1e3, 2) if pm.get('offset') is not None else None,
               place_ok=pm.get('ok'), held_min_mm=min(hv) if hv else None, held_mins=held, held_window=[t_lok, t_low], tables_min_mm=min((round(v[0] * 1e3, 1) for k, v in mins.items() if isinstance(v, tuple) and 'held' not in k), default=None),
               self_min_mm=round(sm[0] * 1e3, 2), finite=T.finite_cmds(f), judge_log=[x for x in c.judge.log][:6])
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
