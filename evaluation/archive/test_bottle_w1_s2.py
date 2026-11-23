#!/usr/bin/env python3
"""持瓶段二階差分候選的最小離線驗收（假受控對象＋合成估計，非物理；Codex 20261101_082911）。
1 jerk：_build_jerk_terms 與逐項直接計算的成本一致（隨機 z）。
2 zero：λ＝0 的候選 vs 原 G（同程序，新 gmpc_wb）vs 原 G（子程序，git HEAD 的 gmpc_wb），到 CARRY_NAV 之後命令逐位元相同。
3 sweep：基準 λ＝0 與 λ 依序 (0.03, 0.1, 0.3) 各跑一次名目全流程，第一個符合事前要求者凍結：
  完成且無停止、命令有限；DONE 增加 ≤ 0.5 s；持瓶導航 TCP 峰值加速度（0.10 s 取樣差分，同 bottle_w1_pair_report）≤ 基準的 0.9 倍；
  CARRY_NAV 之前的命令與基準逐位元相同（抓取段不受影響，近停不變）；安全淨空、抓持、放置判定照原標準。
    python3 evaluation/test_bottle_w1_s2.py [jerk|zero|sweep] [--head-child <out.npz>]
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))

BAD = []
T_ZERO = 27.0                                           # 測試外殼在 t_end−3 s 觸發 sim_limit：比對涵蓋 CARRY_NAV 開始（假受控對象約 21.65 s）後 2 s 以上


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def build(cls, **kw):
    import test_bottle_w1 as T
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = (lambda *a, **k: cls(*a, **dict(k, **kw))) if kw else cls
    try:
        return T.build()
    finally:
        T.W1.W1PlaceController = orig


def run(cls, t_end=75.0, **kw):
    import test_bottle_w1 as T
    c, f = build(cls, **kw)
    r, _ = T.run(c, f, t_end=t_end)
    return c, f, r


def arr(f):
    return np.array([np.r_[h['u'], h['s']] for h in f.hist]), np.array([h['t'] for h in f.hist])


def case_jerk():
    import ammr_wholebody_mpc.gmpc_wb as GW
    rng = np.random.default_rng(1)
    N, m = 6, 3
    S2 = np.diag(rng.uniform(0.5, 3.0, m))
    xr, xp, xp2 = rng.normal(size=(N + 1, m)), rng.normal(size=m), rng.normal(size=m)
    P, q = GW._build_jerk_terms(S2, xr, xp, xp2, N)
    errs = []
    for _ in range(5):
        z = rng.normal(size=N * m)
        u = [xp2, xp] + [xr[k] + z[k * m:(k + 1) * m] for k in range(N)]
        J = sum(float((u[k + 2] - 2 * u[k + 1] + u[k]) @ S2 @ (u[k + 2] - 2 * u[k + 1] + u[k])) for k in range(N))
        z0 = np.zeros(N * m)
        u0 = [xp2, xp] + [xr[k] for k in range(N)]
        J0 = sum(float((u0[k + 2] - 2 * u0[k + 1] + u0[k]) @ S2 @ (u0[k + 2] - 2 * u0[k + 1] + u0[k])) for k in range(N))
        errs.append(abs(J - (0.5 * z @ P @ z + q @ z + J0)))
    chk('二階差分成本：P、q 與逐項直接計算一致', max(errs) < 1e-9, max(errs))


def head_child(out):
    """子程序：載入 git HEAD 版 gmpc_wb 取代新版，跑原 G 到 T_ZERO，存命令。"""
    import importlib.util
    import ammr_wholebody_mpc
    src = subprocess.run(['git', '-C', WS, 'show', 'HEAD:src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py'], capture_output=True, text=True, check=True).stdout
    p = out + '.gmpc_wb_head.py'
    open(p, 'w').write(src)
    spec = importlib.util.spec_from_file_location('ammr_wholebody_mpc.gmpc_wb', p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['ammr_wholebody_mpc.gmpc_wb'] = mod
    spec.loader.exec_module(mod)
    ammr_wholebody_mpc.gmpc_wb = mod
    import bottle_w1_ctrl as W1
    assert 'S2' not in {f_.name for f_ in __import__('dataclasses').fields(mod.GMPCConfig)}, 'HEAD 版應無 S2'
    _, f, _ = run(W1.W1PlaceController, t_end=T_ZERO)
    U, t = arr(f)
    np.savez(out, U=U, t=t)
    os.remove(p)


def case_zero():
    import bottle_w1_ctrl as W1
    import bottle_w1_s2_ctrl as S2C
    tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'w1_s2_head_cmds.npz')
    rc = subprocess.run([sys.executable, os.path.abspath(__file__), '--head-child', tmp], capture_output=True, text=True)
    if rc.returncode != 0:
        chk('HEAD 版子程序', False, rc.stderr[-800:])
        return
    H = np.load(tmp)
    _, f0, r0 = run(W1.W1PlaceController, t_end=T_ZERO)
    c1, f1, r1 = run(S2C.W1PlaceController, t_end=T_ZERO, s2_lambda=0.0)
    U0, t0 = arr(f0)
    U1, _ = arr(f1)
    t_cn = next(e['t'] for e in r0['events'] if e.get('what') == 'place_phase' and e.get('to') == 'CARRY_NAV')
    print(f'   比對到 {T_ZERO} s（CARRY_NAV 於 {t_cn} s 開始），{len(U0)} 步')
    chk('零權重：候選 λ＝0 與原 G（新 gmpc_wb）命令逐位元相同', U0.shape == U1.shape and np.array_equal(U0, U1), None if U0.shape != U1.shape else float(np.abs(U0 - U1).max()))
    chk('零權重：原 G 在新、舊（HEAD）gmpc_wb 下命令逐位元相同', U0.shape == H['U'].shape and np.array_equal(U0, H['U']), None if U0.shape != H['U'].shape else float(np.abs(U0 - H['U']).max()))
    chk('零權重：候選從未啟用 S2', c1.n_s2 == 0, c1.n_s2)


def metrics(c, f, r):
    import bottle_s1_geom as G
    ev = lambda w, to=None: next((e['t'] for e in r['events'] if e.get('what') == w and (to is None or e.get('to') == to)), None)   # noqa: E731
    t_done = ev('mission_done')
    lo, hi = ev('place_phase', 'CARRY_NAV'), ev('place_phase', 'PLACE_APPROACH')
    t = np.array([h['t'] for h in f.hist])
    Q = np.array([h['q'] for h in f.hist])
    k = np.flatnonzero(t <= (t_done if t_done is not None else t[-1]))[::10]
    tt, Qk = t[k], Q[k]
    tcp = np.array([G.C.K.fk(v, 'link_tcp')[:3, 3] for v in Qk])
    v = np.diff(tcp, axis=0) / np.diff(tt)[:, None]
    a = np.linalg.norm(np.diff(v, axis=0) / ((tt[2:] - tt[:-2]) / 2)[:, None], axis=1)
    at = tt[1:-1]
    m = (at >= lo) & (at <= hi) if lo is not None and hi is not None else np.zeros(len(at), bool)
    i = int(np.argmax(np.where(m, a, -1))) if m.any() else None
    return dict(done=bool(r['mission_done']), stop=(r['stop'] or {}).get('why'), t_done=t_done, t_carry=(lo, hi), carry_peak=None if i is None else float(a[i]), carry_peak_t=None if i is None else float(at[i]),
                carry_p95=float(np.percentile(a[m], 95)) if m.any() else None, task_peak=float(a.max()), clear_mm=[None if r['clear_bottle'][x] is None else round(r['clear_bottle'][x] * 1e3, 2) for x in ('meas_min', 'pred_min', 'next_min')],
                slip=r['slip_max_mm_deg'], place_ok=(r['place'].get('metrics') or {}).get('ok'), n_s2=getattr(c, 'n_s2', 0))


def case_sweep():
    import bottle_w1_s2_ctrl as S2C
    import test_bottle_w1 as T
    c0, f0, r0 = run(S2C.W1PlaceController, s2_lambda=0.0)
    m0 = metrics(c0, f0, r0)
    U0, t0 = arr(f0)
    print(f"   基準 λ＝0：{json.dumps(m0, ensure_ascii=False, default=float)}")
    res, chosen = dict(baseline=m0, runs=[]), None
    for lam in S2C.LAMBDAS:
        c, f, r = run(S2C.W1PlaceController, s2_lambda=lam)
        m = metrics(c, f, r)
        U, t = arr(f)
        n_pre = int(np.searchsorted(t0, m0['t_carry'][0]))
        same_pre = U.shape[0] >= n_pre and np.array_equal(U[:n_pre], U0[:n_pre])
        ok = dict(complete=m['done'] and m['stop'] is None and T.finite_cmds(f), time=m['t_done'] is not None and m['t_done'] - m0['t_done'] <= 0.5 + 1e-9,
                  carry_peak=m['carry_peak'] is not None and m['carry_peak'] <= 0.9 * m0['carry_peak'], pre_carry_identical=bool(same_pre), place=m['place_ok'] is True)
        m.update(lam=lam, S2_diag=np.diag(c.S2).tolist(), checks=ok, ok=all(ok.values()))
        res['runs'].append(m)
        print(f"   λ＝{lam}：DONE {m['t_done']}（基準 {m0['t_done']}）持瓶峰 {m['carry_peak'] and round(m['carry_peak'], 3)}@{m['carry_peak_t']}（基準 {round(m0['carry_peak'], 3)}@{m0['carry_peak_t']}）"
              f" p95 {m['carry_p95'] and round(m['carry_p95'], 3)}（基準 {round(m0['carry_p95'], 3)}）全任務峰 {round(m['task_peak'], 3)} 淨空 {m['clear_mm']} S2 啟用 {m['n_s2']} 次；{ok}")
        if m['ok']:
            chosen = lam
            break
    res['chosen'] = chosen
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_s2_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    chk(f'凍結 λ（依序第一個符合）：{chosen}', chosen is not None, '三個值都未達事前要求 ⇒ 照實記錄、轉下一包')


CASES = dict(jerk=case_jerk, zero=case_zero, sweep=case_sweep)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--head-child':
        head_child(sys.argv[2])
        return 0
    for n in sys.argv[1:] or list(CASES):
        CASES[n]()
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
