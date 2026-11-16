#!/usr/bin/env python3
"""主線 B（Codex 20261117_020831）：沿原導航 Smac2D **實際路徑**（門前線，往 −a）找一個完整接近候選——
  1 觀察構型族：相機注視把手（光軸過把手中心；把手中心與兩端都在影像內、邊距 25 px），相機到把手距離 ρ、仰角 e、方位偏角 az 逐點自選，
    同一延拓（Δq＜0.3）、σ≥下限、限位、淨空（規劃門檻）；導航姿態（K2_READY 實測）→第一個觀察構型的過渡另核（最小急動、速率／加速度上限、沿途 σ 與淨空）。
  2 觀察→預抓位過渡：自觀察族上的起點 a_t0 以最小急動（時長＝底盤以 V 由 a_t0 走到接手點 a_w 的時間）轉到接手點的門旁分支預抓位（r＝S_PRE），
    沿途逐點核 σ≥下限、限位、淨空、關節速率／加速度上限、相機可見（記最後可見時刻），以及「該點立即停止」的停止掃掠（底盤以 A_BODY 減速、手臂以 A_ARM 減速到停的終點構型淨空）。
  3 盲段＝最後可見時刻 → 綁定（過渡剩餘＋抓持排程到 READY 2.05 s＋交接 1 拍），以真實時間計。
  4 接手狀態：a_w 沿門 −a 的固定方向抓持窗口（door_turn_search.window_margins）。
選盲段最短的可行 a_t0；全部不可行就照實回報。導航遠端目標不在此決定（另由窗口與 GMPC 到達容差反推並核 Smac 路徑）。運動學層離線；把手＝far_door_9 近距鎖定參考。
    python3 door_obs_candidate.py RUN_DIR PROBE_JSON OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
V, A_W, T_READY = 0.05, -0.10, 2.05
Q_NAV = np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018])
V_ARM, A_ARM = float(TS.W.V_ARM), float(TS.W.A_ARM)
A_BODY = np.asarray(TS.W.A_BODY, float)
P = json.load(open(sys.argv[2]))
path = np.array([p[:2] for p in P[0]['plan']], float)
rel = path - TS.h_xy
SB_LINE = float(np.median((rel @ TS.n_)[np.abs(rel @ TS.n_ - (rel @ TS.n_)[-1]) < 0.03]))
q9_ref = np.r_[TS.base(A_W, SB_LINE), TS.end_config(A_W, SB_LINE)]
T_TC = np.linalg.inv(TS.K.fk(q9_ref, 'link_tcp')) @ TS.K.fk(q9_ref, TS.W0.OPT)   # 相機相對 TCP（固定）
H = TS.T_H[:3, 3]


def sigma6(q9):
    J = TS.K.jacobian(q9, 'link_tcp')[:, 3:9]
    return float(np.linalg.svd(J, compute_uv=False).min())


def look_at_tcp(b, rho, e_deg, az_deg):
    u = np.r_[b[:2] - H[:2], 0.0]; u /= np.linalg.norm(u)
    c, s = math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))
    u = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1], 0.0])
    e = math.radians(e_deg)
    p = H + rho * (math.cos(e) * u + np.array([0, 0, math.sin(e)]))
    z = (H - p) / np.linalg.norm(H - p)
    x = np.cross(np.array([0, 0, 1.0]), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Tc = np.eye(4); Tc[:3, :3] = np.c_[x, y, z]; Tc[:3, 3] = p
    return Tc @ np.linalg.inv(T_TC)


def ok_cfg(b, q):
    lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
    cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
    return lim >= 0 and all(cl[k] >= TS.THRP[k] for k in ('env', 'base', 'self')), lim, cl


# ---- 1 觀察構型族（沿門前線由遠到近：a 由 0.60 到 0.0）
OBS_P = [(rho, e, az) for rho in (0.35, 0.42, 0.50) for e in (10, 25, 40) for az in (0, -20, 20)]
fam, seed = [], None
for a in np.round(np.arange(0.60, -0.0001, -0.02), 3):
    b = TS.base(float(a), SB_LINE)
    best = None
    for (rho, e, az) in OBS_P:
        tgt = look_at_tcp(b, rho, e, az)
        for sd in ([seed] if seed is not None else []) + [Q_NAV, TS.end_config(A_W, SB_LINE)]:
            q, res, sm = TS.DFK.ik_closure_fc(tgt, b, sd)
            if not (res < 1e-6 and np.isfinite(q).all()) or sm < TS.SIG:
                continue
            if seed is not None and np.abs(q - seed).max() > 0.3:
                continue
            okc, lim, cl = ok_cfg(b, q)
            if not okc or not TS.visible(np.r_[b, q]):
                continue
            cost = 0.0 if seed is None else float(np.abs(q - seed).max())
            if best is None or cost < best[0]:
                best = (cost, q, sm, lim, cl, (rho, e, az))
            break
    if best is None:
        if fam:
            break
        continue
    seed = best[1]
    fam.append(dict(a=float(a), q=best[1], sigma=best[2], lim=best[3], self=float(best[4]['self']), env=float(best[4]['env']), par=best[5]))
print('obs family: a %.2f -> %.2f, n %d' % (fam[0]['a'], fam[-1]['a'], len(fam)) if fam else 'obs family: none', flush=True)
out = dict(note='運動學層離線；沿 Smac 實際門前線；把手＝far_door_9 近距鎖定參考', sb_line=round(SB_LINE, 4), a_w=A_W, V=V,
           obs_family=[dict(a=x['a'], sigma=round(x['sigma'], 4), self=round(x['self'], 4), par=x['par']) for x in fam])
q_pre = TS.end_config(A_W, SB_LINE)


def minjerk_check(qa, qb, T, a0, a1, n=60, stop_sweep=True):
    """關節空間最小急動 qa→qb（T 秒），底盤沿門前線 a0→a1 等速；回 dict 或失敗原因。"""
    dq = qb - qa
    if float(np.abs(dq).max()) * 1.875 / T > V_ARM or float(np.abs(dq).max()) * 5.7735 / T ** 2 > A_ARM:
        return None, 'rate_or_acc'
    last_vis, rows = None, []
    for k in range(n + 1):
        u = k / n
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        sd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
        q = qa + s * dq
        a = a0 + (a1 - a0) * u
        b = TS.base(a, SB_LINE)
        okc, lim, cl = ok_cfg(b, q)
        sm = sigma6(np.r_[b, q])
        if not okc:
            return None, 'clear_or_limit@u=%.2f' % u
        if sm < TS.SIG:
            return None, 'sigma@u=%.2f(%.4f)' % (u, sm)
        if TS.visible(np.r_[b, q]):
            last_vis = u * T
        if stop_sweep:                                                # 立即停止：底盤 V 以 A_BODY 減速、手臂各軸以 A_ARM 減速，終點構型淨空
            qd = sd * dq
            q_st = q + qd * np.abs(qd) / (2 * A_ARM)
            d_base = V * V / (2 * float(A_BODY[1]))
            b_st = TS.base(a - d_base, SB_LINE)
            okc2, _, _ = ok_cfg(b_st, q_st)
            if not okc2:
                return None, 'stop_sweep@u=%.2f' % u
        rows.append(dict(u=u, sigma=sm, self=float(cl['self']), env=float(cl['env'])))
    return dict(last_vis_t=last_vis, sigma_min=min(x['sigma'] for x in rows), self_min=min(x['self'] for x in rows), env_min=min(x['env'] for x in rows),
                rate=float(np.abs(dq).max()) * 1.875 / T, acc=float(np.abs(dq).max()) * 5.7735 / T ** 2), None


# ---- 導航姿態 → 第一個觀察構型
if fam:
    f0 = fam[0]
    T0 = max(float(np.abs(f0['q'] - Q_NAV).max()) * 1.875 / V_ARM, float(np.sqrt(float(np.abs(f0['q'] - Q_NAV).max()) * 5.7735 / A_ARM)), 1.0) * 1.05
    r0, w0 = minjerk_check(Q_NAV, f0['q'], T0, f0['a'] + 0.15 * T0, f0['a'], stop_sweep=True)
    out['nav_to_obs'] = dict(T=round(T0, 2), ok=r0 is not None, why=w0, info=None if r0 is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r0.items()})
    print('nav->obs', out['nav_to_obs'], flush=True)
# ---- 2 觀察 → 預抓位（逐一試過渡起點）
cands = []
for f in fam:
    T = (f['a'] - A_W) / V
    r, why = minjerk_check(f['q'], q_pre, T, f['a'], A_W)
    rec = dict(a_t0=f['a'], T=round(T, 2), ok=r is not None, why=why)
    if r is not None:
        lv = r['last_vis_t']
        blind = (T - (lv if lv is not None else 0.0)) + T_READY + 0.05
        rec.update(last_vis_t=None if lv is None else round(lv, 2), blind_s=round(blind, 2), sigma_min=round(r['sigma_min'], 4), self_min=round(r['self_min'], 4), env_min=round(r['env_min'], 4), rate=round(r['rate'], 3), acc=round(r['acc'], 3))
    cands.append(rec)
ok = sorted([c for c in cands if c['ok']], key=lambda c: c['blind_s'])
out['transition'] = cands
out['best'] = ok[0] if ok else None
wm = TS.window_margins(A_W, SB_LINE, q_pre)
out['window'] = None if wm is None else {k: round(v, 4) for k, v in wm.items()}
fails = {}
for c in cands:
    if not c['ok']:
        k = c['why'].split('@')[0]; fails[k] = fails.get(k, 0) + 1
out['fails'] = fails
json.dump(out, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=str)
print('transition cands', len(cands), 'ok', len(ok), fails, flush=True)
for c in ok[:5]:
    print(c, flush=True)
print('window', out['window'])
