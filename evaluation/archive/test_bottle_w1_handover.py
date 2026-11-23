#!/usr/bin/env python3
"""持瓶承接候選的最小離線驗收（Codex 20261101_085238 (a)；假受控對象＋合成估計，非物理）。
峰值算法（基準與候選、離線與物理全用同一個）：取任務時間 0.10 s 整數倍的樣本（t＝0.1·k），TCP 位置中央二階差分得加速度，
在承接窗口 [t_sw − 0.5, t_sw + 1.0] 取最大值；t_sw＝place_phase→CARRY_NAV 事件時刻。舊總表的數字不回改。
事前要求：承接窗口峰值 ≤ 基準 0.9 倍；DONE 增加 ≤ 0.5 s；承接前命令與基準逐位元相同；完成、無停止、命令有限、放置判定通過；重錨定的三項檢查照原門檻通過。
另用同一算法重算既有物理趟（不跑新物理）作對照。
    python3 evaluation/test_bottle_w1_handover.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_ctrl as W1                             # noqa: E402
import bottle_w1_handover_ctrl as HO                    # noqa: E402
import test_bottle_w1 as T                              # noqa: E402

BAD = []
WIN = (-0.5, 1.0)
PHYS_RUNS = ('bottle_w1_1', 'bottle_w1_g2_clean1', 'bottle_w1_p_nominal_auto1', 'bottle_w1_p_minus5_auto1')


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def handover_peak(t, Q, t_sw):
    """t：任務時間（0.01 s 步）；Q：(n, 9) 量測底盤＋關節。回傳 (峰值 m/s², 峰值時刻, 窗口樣本數)。"""
    k = np.flatnonzero(np.abs(t * 10.0 - np.round(t * 10.0)) < 1e-6)
    tt, Qk = t[k], Q[k]
    tcp = np.array([G.C.K.fk(v, 'link_tcp')[:3, 3] for v in Qk])
    a = np.linalg.norm(tcp[2:] - 2.0 * tcp[1:-1] + tcp[:-2], axis=1) / 0.01
    at = tt[1:-1]
    m = (at >= t_sw + WIN[0] - 1e-9) & (at <= t_sw + WIN[1] + 1e-9)
    i = int(np.argmax(np.where(m, a, -1.0)))
    return float(a[i]), float(at[i]), int(m.sum())


def ev_t(r, what, to=None):
    return next((e['t'] for e in r['events'] if e.get('what') == what and (to is None or e.get('to') == to)), None)


def run(cls):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = cls
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    r, _ = T.run(c, f)
    t = np.array([h['t'] for h in f.hist])
    Q = np.array([h['q'] for h in f.hist])
    U = np.array([np.r_[h['u'], h['s']] for h in f.hist])
    t_sw = ev_t(r, 'place_phase', 'CARRY_NAV')
    pk = handover_peak(t, Q, t_sw) if t_sw is not None else (None, None, 0)
    return c, f, r, t, U, t_sw, pk


def phys(run_id):
    D = os.path.join(HERE, 'runs', run_id)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    Q = np.column_stack([c[k] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
    t_sw = ev_t(r, 'place_phase', 'CARRY_NAV')
    pk = handover_peak(c['t'], Q, t_sw)
    return dict(run=run_id, t_sw=t_sw, peak=pk[0], peak_t=pk[1], n=pk[2], t_done=ev_t(r, 'mission_done'))


def main():
    out = dict(window_s=WIN, algorithm='t=0.1k 樣本、TCP 中央二階差分、承接窗口最大值', physical=[phys(x) for x in PHYS_RUNS])
    for p in out['physical']:
        print(f"   物理 {p['run']}：承接 {p['t_sw']} s，窗口峰 {p['peak']:.3f} m/s² @ {p['peak_t']:.2f}（{p['n']} 樣本）")
    c0, f0, r0, t0, U0, sw0, pk0 = run(W1.W1PlaceController)
    c1, f1, r1, t1, U1, sw1, pk1 = run(HO.W1PlaceController)
    d0, d1 = ev_t(r0, 'mission_done'), ev_t(r1, 'mission_done')
    print(f"   假受控對象 基準 G：承接 {sw0} s 窗口峰 {pk0[0]:.3f} @ {pk0[1]:.2f}，DONE {d0}")
    print(f"   假受控對象 候選：承接 {sw1} s 窗口峰 {pk1[0] if pk1[0] is None else round(pk1[0], 3)} @ {pk1[1]}，DONE {d1}，重錨定 {json.dumps(c1.ho, ensure_ascii=False, default=float)[:400]}")
    n_pre = int(np.searchsorted(t0, sw0))
    out['fake'] = dict(baseline=dict(t_sw=sw0, peak=pk0[0], peak_t=pk0[1], t_done=d0), candidate=dict(t_sw=sw1, peak=pk1[0], peak_t=pk1[1], t_done=d1, handover=c1.ho,
                       stop=r1['stop'], clear_mm=[None if r1['clear_bottle'][k] is None else round(r1['clear_bottle'][k] * 1e3, 2) for k in ('meas_min', 'pred_min', 'next_min')],
                       slip=r1['slip_max_mm_deg'], place=(r1['place'].get('metrics') or {}).get('ok')))
    chk('候選：完成、無停止、命令有限、放置判定通過', bool(r1['mission_done']) and r1['stop'] is None and T.finite_cmds(f1) and (r1['place'].get('metrics') or {}).get('ok') is True, r1['stop'])
    chk('候選：重錨定已執行且三項檢查照原門檻通過', c1.ho is not None and any(e.get('what') == 'handover_reanchor' for e in r1['events']), c1.ho)
    chk('承接前命令與基準逐位元相同', U1.shape[0] >= n_pre and np.array_equal(U1[:n_pre], U0[:n_pre]))
    chk('DONE 增加 ≤ 0.5 s', d1 is not None and d0 is not None and d1 - d0 <= 0.5 + 1e-9, (d0, d1))
    chk('承接窗口峰值 ≤ 基準 0.9 倍', pk1[0] is not None and pk1[0] <= 0.9 * pk0[0], (pk0[0], pk1[0]))
    out['pass'] = not BAD
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_handover_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
