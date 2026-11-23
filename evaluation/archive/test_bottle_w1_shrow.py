#!/usr/bin/env python3
"""第一拍 Shield 一致列的最小離線核對（Codex 20261101_130612）。非物理。
1 列與 Shield 等價性：隨機點、年齡、命令 3000 組；列滿足（餘裕 1e-7）⇒ Shield._limit 不改寫（必須 0 例）；
  列違反但 Shield 不改只允許出現在點已進 d0＋eps 的退回區（Shield 退回改成「不接近」，列較嚴）。
2 前進／切向／退離／過期：正前方點限前進、不限切向與退離；年齡越大上界越小。
3 原觸發拍回放（bottle_w1_dyn2lc_g1 22.65 s 求解、22.66 s Shield 改寫）：掃描未記錄 ⇒ 以紀錄底盤位姿＋場景幾何＋移動物軌跡離線射線重建；
  重建掃描下 Shield 對原命令的改寫與實錄比較；同一求解紀錄（s1_cbf.npz）不加列重解須重現原命令，加列重解：回傳命令滿足列、重建掃描下 Shield 不改寫。
4 退化：u0_rows=None 與原求解逐位元相同（同紀錄）。
5 名目、6 原第二案：假受控對象閉迴路（完成或受控停止、Shield 介入次數、列貼合次數、違反、命令有限、切換後持瓶↔桌、禁止接觸）。
    python3 evaluation/test_bottle_w1_shrow.py
"""
import dataclasses
import json
import os
import sys

import numpy as np

os.environ.setdefault('DYN_CASE', 'w1_dyn_overtake1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import test_bottle_w1_box2 as TB2                       # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402
import test_bottle_w1_liftc as TC                       # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402

BAD = []
RUN = 'bottle_w1_dyn2lc_g1'
T_SOLVE = 22.65


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def case_equiv(sh):
    rng = np.random.default_rng(1)
    n_bad, n_fb_only, n = 0, 0, 3000
    for _ in range(n):
        k = rng.integers(1, 6)
        ang, rr = rng.uniform(-np.pi, np.pi, k), rng.uniform(0.45, 1.2, k)
        pts = np.c_[rr * np.cos(ang), rr * np.sin(ang)]
        age = rng.uniform(0, 0.2)
        A, b, _ = SR.shield_rows(sh, pts, age, prune=False)
        u = rng.uniform(-1, 1, 3) * sh.u_max
        s = (A @ u - b).max()
        _, dv, *_ = sh._limit(u, pts, age)
        if s <= -1e-7 and dv > 0:
            n_bad += 1
        if s > 1e-4 and dv == 0:
            in_fb = bool((np.hypot(*pts.T) - sh.R - sh.d0 - sh.eps < 0).any())
            n_bad += int(not in_fb)
            n_fb_only += int(in_fb)
    chk(f'1 列滿足 ⇒ Shield 不改寫（{n} 組）；列較嚴只出現在退回區（{n_fb_only} 組）', n_bad == 0, n_bad)
    return dict(n=n, bad=n_bad, stricter_in_fallback=n_fb_only)


def case_dirs(sh):
    pts = np.array([[0.70, 0.0]])
    A, b, info = SR.shield_rows(sh, pts, 0.05)
    fw = float(b[0])
    for nm, u, ok_exp in (('前進 0.9×上界', [0.9 * fw, 0, 0], True), ('前進 1.1×上界', [1.1 * fw, 0, 0], False), ('切向 vy 0.27', [0.0, 0.27, 0], True), ('退離 vx −0.27', [-0.27, 0, 0], True)):
        ok = bool(np.all(A @ np.array(u) - b <= 1e-9))
        _, dv, *_ = sh._limit(np.array(u, float), pts, 0.05)
        chk(f'2 {nm}：列{"允許" if ok_exp else "拒絕"}、Shield {"不改" if ok_exp else "改寫"}', ok == ok_exp and ((dv == 0) == ok_exp), (ok, dv))
    b_age = [float(SR.shield_rows(sh, pts, a)[1][0]) for a in (0.0, 0.1, 0.5)]
    chk(f'2 過期：年齡 0／0.1／0.5 s ⇒ 前進上界 {np.round(b_age, 4).tolist()} 遞減', b_age[0] > b_age[1] > b_age[2], b_age)
    return dict(fw_bound=fw, age_bounds=b_age)


def recon_scan(t_scan):
    D = os.path.join(HERE, 'runs', RUN)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    col = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    k = int(np.argmin(np.abs(col['t'] - t_scan)))
    q = np.array([col['base_x'][k], col['base_y'][k], col['base_yaw'][k]])
    bx = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']
    r = NC.ray_boxes(q[:2], float(q[2]), bx, G.LIDAR_Z)
    CA = DC.CASES['w1_dyn_overtake1']
    oc, _ = DC.state(t_scan, CA)
    return np.minimum(r, DC.ray_cyl(q[:2], NC.scan_angles() + float(q[2]), oc, CA['radius'], NC.RANGE_MAX)), q


def case_replay(sh):
    D = os.path.join(HERE, 'runs', RUN)
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    sh_first = rep['nav_chain']['shield_first']
    st = rep['stop']
    u_in, u_out = np.array(st['u_in']), np.array(st['u_out'])
    t_scan = round(T_SOLVE - 0.05, 2)
    rr, q_scan = recon_scan(t_scan)
    pts = SR.scan_points(sh, rr)
    out, dv, dmin, *_ = sh._limit(u_in, pts, 0.05)
    print(f'   實錄：Shield 首次改寫 {sh_first[0]}（t、dv、d_min），套用 22.73 s：u_in {np.round(u_in, 4).tolist()} → {np.round(u_out, 4).tolist()}')
    print(f'   重建掃描（{t_scan} s，底盤 {np.round(q_scan, 3).tolist()}，{len(pts)} 點）：Shield 對原命令改寫 dv {dv:.4f}、d_min {dmin:.4f} → {np.round(out, 4).tolist()}')
    chk('3 重建掃描重現改寫（dv > SHIELD_DV 1 mm/s）', dv > 1e-3, dv)
    TB2.RUN = RUN
    kw, _, tt, ph = TB2.record(T_SOLVE)
    cfg = TB2.nav(None, 0.0)
    r0 = GW.GMPC(cfg).solve(**kw)
    Zc = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    i = int(np.argmin(np.abs(Zc['t'] - T_SOLVE)))
    u_rec = np.asarray(Zc['u'][i][:3], float)
    rep0 = float(np.abs(np.asarray(r0.u_opt, float) - u_rec).max())
    print(f'   紀錄 {tt:.2f} s {ph}：不加列重解 {np.round(r0.u_opt, 4).tolist()}，紀錄命令 {np.round(u_rec, 4).tolist()}，差 {rep0:.1e}')
    chk('3 不加列重解重現紀錄命令（差 ≤ 1e-6）', rep0 <= 1e-6, rep0)
    r_none = GW.GMPC(cfg).solve(**kw, u0_rows=None)
    chk('4 u0_rows=None 與原求解逐位元相同', np.array_equal(np.asarray(r_none.u_opt), np.asarray(r0.u_opt)), (r_none.u_opt, r0.u_opt))
    A, b, info = SR.shield_rows(sh, pts, (T_SOLVE - t_scan) + SR.AGE_PAD)
    r1 = GW.GMPC(cfg).solve(**kw, u0_rows=(A, b))
    u1 = np.asarray(r1.u_opt, float)
    out1, dv1, *_ = sh._limit(u1, pts, 0.05)
    out1b, dv1b, *_ = sh._limit(u1, pts, (T_SOLVE - t_scan) + SR.AGE_PAD)
    fb = (r1.arm_check or {}).get('fallback')
    print(f'   加列重解：{r1.status}／{r1.accept_action}／fallback {fb}，命令 {np.round(u1, 4).tolist()}，列違反 {r1.u0_rows_viol:+.1e}（{info["n_rows"]} 列），Shield 年齡 0.05 ⇒ dv {dv1:.1e}、年齡上界 ⇒ dv {dv1b:.1e}')
    chk('3 加列重解：回傳命令滿足列且重建掃描下 Shield 不改寫（無退回）', fb is None and r1.u0_rows_viol <= SR.TOL and dv1 == 0 and dv1b == 0, (fb, r1.u0_rows_viol, dv1, dv1b))
    return dict(shield_rec=sh_first[0], u_in=u_in.tolist(), u_out=u_out.tolist(), recon=dict(t_scan=t_scan, n_pts=len(pts), dv=dv, d_min=dmin, out=out.tolist()),
                resolve_err=rep0, u_rows=u1.tolist(), viol=r1.u0_rows_viol, n_rows=info['n_rows'], dv_after=[dv1, dv1b], status=str(r1.status), fallback=fb)


def case_loop(dyn):
    tag = 'dyn2' if dyn else 'nominal'
    c, f, r = TL.run(SR.W1PlaceController, dyn)
    o = TL.summarize(f'{tag} SHROW', c, f, r, dyn)
    ps = r['paired_strategy']
    u0 = ps['u0_rows']
    t_end = o['t_done'] if o['t_done'] is not None else (o['stop'] or {}).get('t', f.hist[-1]['t'])
    o['near_run_s'], _ = TC.near_stop(f, 0.2, t_end)
    o['u0_rows'] = {k: u0[k] for k in ('n_solve', 'n_bind', 'rows_max', 'violation', 'bind_first', 'bind_last')}
    o['shield'] = r['nav_chain']['shield']
    o['carry_wait'] = ps.get('carry_wait')
    print(f"   [{tag} SHROW] 列：求解 {u0['n_solve']}、貼合 {u0['n_bind']}、最多 {u0['rows_max']} 列、違反 {u0['violation']}；貼合首 {u0['bind_first'][:3]}；Shield {o['shield']}；全程近停 {o['near_run_s']:.2f} s")
    ha = o.get('held_after_switch_mm')
    held_ok = ha is None or all(v[0] >= TL.REJ * 1e3 for v in ha.values())
    chk(f'{5 if not dyn else 6} {tag}：完成或受控停止、命令有限、無禁止接觸、切換後持瓶↔桌 ≥ 5.5 mm、自碰 > 0', (o['done'] or o['stop'] is not None) and o['finite'] and o['forb'] == 0 and held_ok and o['self_min_mm'] > 0,
        (o['stop'], o['forb'], ha))
    if not dyn:
        chk('5 名目：候選完成', o['done'], o['stop'])
    return o


def main():
    sh = SR.make_shield()
    res = dict(age_pad=SR.AGE_PAD, tol=SR.TOL, shield=dict(R=sh.R, d0=sh.d0, eps=sh.eps, tau=sh.tau, alpha=sh.alpha, a_brake=sh.a_brake, u_max=list(sh.u_max)))
    res['equiv'] = case_equiv(sh)
    res['dirs'] = case_dirs(sh)
    res['replay'] = case_replay(sh)
    res['nominal'] = case_loop(False)
    res['dyn2'] = case_loop(True)
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_shrow_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
