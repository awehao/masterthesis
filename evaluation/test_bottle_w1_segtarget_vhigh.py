#!/usr/bin/env python3
"""V_HIGH 接近序列（bottle_w1_segtarget_vhigh_ctrl）的離線閉迴路核對（Codex 20261102_023000 第 4 項；假世界、假候選，非物理）。
用法：python3 evaluation/test_bottle_w1_segtarget_vhigh.py Z1 CASE；python3 evaluation/test_bottle_w1_segtarget_vhigh.py all（依序跑全部，寫 results/vision/segtarget_vhigh_offline.json）。
假候選同 test_bottle_w1_segtarget（known_type 5 Hz、中心＝真值 ± 0.3 mm、z_support＝Z1、快照 3 s 換序號），只在序列 H 段（手臂在 V_HIGH 保持）送——模擬只渲染核對中 V_HIGH 段 30/30 有效；
不送 U 段（渲染中 U 段後半已就緒，這裡保守不用）。
案例：
  nom    Z1 0.60：完整抓放；序列 TUCK→U→H→R→DONE 依序、DONE 早於漸變起點；採用發生在 H 段
  never  Z1 0.60：從不送 ⇒ segtarget_no_fix（漸變起點），不閉爪
  late_r Z1 0.60：假受控對象 j6 在 R 段開始 1.0 s 後卡住 2.0 s（響應歸零；模擬實際過渡較慢，量測照實）⇒ DONE 延後、不得跳段；底盤在 x_blend 減速等手臂；之後結果照實報
  stall  Z1 0.60：j6 在 U 段開始 1.0 s 後卡住 8.0 s（U 時長 4.65 s 走完後仍卡 4.35 s） ⇒ U 完成條件等候 > 3 s ⇒ vhigh_seq_stalled 受控停止、不閉爪
  slow   Z1 0.60：手臂響應 ×0.35（模型不符壓力案）⇒ 序列依序不跳段；結果須與基線控制器同條件一致（基線 segtarget 同條件 19.3 s clearance_bottle 受控停止）
  offset Z1 0.60：起始手臂 j1 偏 +0.10 rad ⇒ TUCK 段先等收斂才進 U（等候紀錄 tuck_not_reached 或 MPC 先收斂），之後完整抓放
  z075   Z1 0.75：H 段送 0.75 候選 ⇒ 基線全窗規則 segtarget_unreachable 受控停止、不閉爪（0.75 真實時序可達另核，不在此判定）
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
CASES = {'nom': (0.60, {}), 'never': (0.60, dict(send=False)), 'late_r': (0.60, dict(stuck=('R', 1.0, 2.0))), 'stall': (0.60, dict(stuck=('U', 1.0, 8.0))), 'slow': (0.60, dict(alpha=0.35)),
         'offset': (0.60, dict(j1=0.10)), 'z075': (0.75, {})}
OUT = os.path.join(HERE, 'results', 'vision', 'segtarget_vhigh_offline.json')

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    res, bad = {}, []
    for k, (z1, _) in CASES.items():
        p = subprocess.run([sys.executable, __file__, str(z1), k], capture_output=True, text=True, timeout=3600)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        r = json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2000:])
        res[k] = r
        bad += [] if r.get('pass') else [k]
        print(f"{'✓' if r.get('pass') else '✗'} {k}（Z1 {z1}）：{r.get('outcome')}｜{r.get('note', r.get('error', ''))}", flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"\n{'全過' if not bad else '未過 ' + str(bad)}")
    sys.exit(1 if bad else 0)

Z1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
CASE = sys.argv[2] if len(sys.argv) > 2 else 'nom'
import bottle_s1_geom as G                              # noqa: E402
_TB = G.table_boxes
_S060 = dict(G.S1_TABLE)


def table1_boxes(t=None, x0=0.0):
    return _TB(dict(_S060, top_z=Z1) if t is None else t, x0)


G.table_boxes = table1_boxes
import test_bottle_s0b as S0B                           # noqa: E402
S0B.TOP, S0B.HG = Z1, Z1 + 0.10
import test_bottle_place as TP                          # noqa: E402
TP.table2_boxes = lambda: [(n, c + np.array([TP.DX, TP.DY, 0.0]), sz) for n, c, sz in _TB(_S060)]
import bottle_w1_compactcarry2_ctrl as C2               # noqa: E402
C2.TOP_Z, C2._TOP = Z1, table1_boxes()[0]
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_w1_segtarget_vhigh_ctrl as VH             # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_compactcarry as TC                # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402


def run(send=True, alpha=1.0, j1=0.0, stuck=None, seed=0):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = VH.W1PlaceController
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    f.alpha = np.array(f.alpha * alpha, float)
    f.q[3] += j1
    rng = np.random.default_rng(seed)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, n_sent=0, ph_at_freeze=None)
    step0 = c.step

    def step(t, q, **kw):
        if send and t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if c.vh_seq['ph'] == 'H' and c.vfix is None:
                xy = np.array([T.T.BX, T.T.BY]) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
                st['n_sent'] += 1
        if stuck is not None:
            sq = c.vh_seq
            if sq['ph'] == stuck[0] and st.get('t_off') is None and t >= sq['t0'] + stuck[1] - 1e-9:
                st['t_off'], st['alpha5'] = t, float(f.alpha[5])
            if st.get('t_off') is not None:
                f.alpha[5] = 0.0 if t < st['t_off'] + stuck[2] else st['alpha5']   # 受控對象 j6 卡住（量測照實）
        out = step0(t, q, **kw)
        if c.vfix is not None and st['ph_at_freeze'] is None:
            st['ph_at_freeze'] = c.vh_seq['ph']
        return out
    c.step = step
    r, _ = T.run(c, f, t_end=75.0)
    return c, f, r, st


def main():
    kw = CASES[CASE][1]
    c, f, r, st = run(**kw)
    o = TC.summarize(f'Z1 {Z1} {CASE}', c, f, r)
    why = (r['stop'] or {}).get('why')
    done = bool(o['done'] and (o['place'] or {}).get('ok'))
    vs = r['vhigh_seq']
    log = vs['log']
    phs = [e[1] for e in log]
    t_of = {e[1]: e[0] for e in log}
    ev = {e['what']: e for e in r['events']}
    t_blend = (ev.get('blend_start') or {}).get('t')
    mins, sm, ns, rate, marg = PCK.check(f)
    S = np.array([h['s'] for h in f.hist])
    acc = float(np.abs(np.diff(S, 2, axis=0)).max() / 0.01 ** 2) if len(S) > 3 else None
    order_ok = phs == ['U', 'H', 'R', 'DONE'][:len(phs)]
    # 不跳段：每段開始時，上一段的時長已走完（U、R 段的 T）
    no_skip = all(log[i][0] >= log[i - 1][0] + log[i - 1][3] - 1e-6 for i in range(1, len(log)) if log[i - 1][1] in ('U', 'R'))
    xs = np.array([h['x'] for h in f.hist])
    vb = np.array([h['v_b'] for h in f.hist])
    th = np.array([h['t'] for h in f.hist])
    slow_at_blend = None if t_blend is None else float(vb[(th >= t_blend - 1.0) & (th <= t_blend)].min())
    out = dict(case=CASE, z1=Z1, outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=o['t_done'], seq=log, waits=vs['waits'], order_ok=order_ok, no_skip=no_skip,
               t_blend=t_blend, done_before_blend=None if t_blend is None or 'DONE' not in t_of else t_of['DONE'] <= t_blend + 1e-9, ph_at_freeze=st['ph_at_freeze'],
               v_b_min_1s_before_blend=slow_at_blend, n_sent=st['n_sent'], close_req=c.t_close_req, finite=T.finite_cmds(f), forb=o['forb'], held=o['held'],
               self_min_mm=round(sm[0] * 1e3, 2), rate_peak=round(rate, 3), acc_peak=None if acc is None else round(acc, 2), joint_margin=round(marg, 3),
               mins={k: (round(v[0] * 1e3, 1) if isinstance(v, tuple) else v) for k, v in mins.items()}, segtarget=dict((k, r['segtarget'][k]) for k in ('fix', 'rejected')))
    if CASE in ('nom', 'offset'):
        ok = done and order_ok and no_skip and out['done_before_blend'] and st['ph_at_freeze'] == 'H' and o['forb'] == 0 and out['finite'] and all(v[0] >= TL.REJ * 1e3 for v in o['held'].values())
    elif CASE == 'late_r':
        ok = (order_ok and no_skip and out['done_before_blend'] is not False and t_of.get('DONE', 0) >= t_of.get('R', 0) + 2.35 + 0.5 and out['finite'] and o['forb'] == 0
              and (done or why in ('clearance_bottle', 'segtarget_expired')))
    elif CASE == 'stall':
        ok = why == 'vhigh_seq_stalled' and c.t_close_req is None and out['finite'] and o['forb'] == 0 and 'R' not in phs
    elif CASE == 'slow':
        ok = order_ok and no_skip and out['finite'] and o['forb'] == 0 and why == 'clearance_bottle'
    elif CASE == 'never':
        ok = why == 'segtarget_no_fix' and c.t_close_req is None and out['finite'] and order_ok
    else:
        ok = why == 'segtarget_unreachable' and c.t_close_req is None and out['finite'] and o['forb'] == 0 and st['ph_at_freeze'] is None
    out['pass'] = bool(ok)
    out['note'] = f"序列 {log}；漸變起點 {t_blend}；採用時段 {st['ph_at_freeze']}；等候 {vs['waits']}；漸變前 1 s 最低參考速度 {slow_at_blend}；完成 {o['t_done']}；速率峰 {rate:.3f}"
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
