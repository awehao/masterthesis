#!/usr/bin/env python3
"""門把手：遠距估計 → 途中重瞄 → 由當下構型接近（離線 r3；Codex 20261112 選 A）。不跑模擬、不閉爪、不開門；只用既有 m8doorcap_1 與凍結偵測／估計。
r2 的補正：
  a 碰撞對：觀察轉移、重瞄、預抓及其停止軌跡核**全部**連桿對橫桿（含手指、夾爪殼），門檻同手臂對環境；只有預抓→抓取的指定接觸段豁免手指對橫桿，夾爪殼不豁免。
  b 停止取樣：減速步用梯形積分（r2 先扣一整步速度再積分，低估位移）；另做步長收斂核對（0.04 對 0.01 s）。仍是簡化停止模型，不是真在途命令管線。
  c 撤回「<0.8 m 一律偵測失效」：開發集距門面 0.9 m 有 7／18 視點成功；近距有效性與視角有關，不按距離一概而論。
途中重瞄候選（由遠距估計在線生成，不是固定必經姿態）：
  底盤位置取在「起點→所選停位」連線的 λ∈{0.5, 0.75, 1.0}；手腕相機位姿＝在把手門外法向上方的幾個方向（俯仰 {0°, 25°}、方位 {0°, ±25°}）、距離 {0.35, 0.50} m、光軸指向把手中心、影像水平軸沿橫桿。
  幾何可見性判準（兩顆相機分別評估；不是偵測成功的證據）：橫桿兩端與中心、深度探測區（中心上下各 6 cm 的門面點）都在畫面內且離邊 ≥ 40 px；
    距離在 [0.25, 1.2] m；相機到這些點的視線不被自己的連桿擋住（連桿點雲到視線 ≥ 2 cm，相機所在連桿除外）。
  每個可見的重瞄構型：起點→重瞄（有界同步、全碰撞、停止取樣）→ 觀測停留 T_LOOK → 重瞄→預抓（同）→ 預抓→抓取接近。
  報：總時間、最後有效觀測的位置、之後未再觀測的 TCP 行程與時間、計算耗時。
    python3 evaluation/m8_door_reaim_online_r3.py m8doorcap_1 --group dev|holdout"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_door_standoff_online_r2 as R2                            # noqa: E402
import bottle_v1_cam_visibility as VZ                             # noqa: E402

R1, PG, DOC, E, K, IDX = R2.R1, R2.PG, R2.DOC, R2.E, R2.K, R2.IDX
T_LOOK = 0.5                                                      # 重瞄停留：既有估計用同視點多幀（N-obs），取 0.5 s（工程設定）
EDGE, D_RNG, RAY_CLR, PROBE = 40.0, (0.25, 1.2), 0.02, 0.06
LAMS, DCAMS, ELEVS, AZIS = (0.5, 0.75, 1.0), (0.35, 0.50), (0.0, 25.0), (0.0, -25.0, 25.0)
print('#EARLY r3', dict(T_LOOK=T_LOOK, EDGE=EDGE, D_RNG=D_RNG, RAY_CLR=RAY_CLR), flush=True)


# ---- a 全碰撞對
def cfg_clear_full(qf, T_WH, contact=False):
    """contact=False：全部連桿對橫桿都核。contact=True（指定接觸段）：只豁免手指對橫桿。"""
    cc = R1.cfg_clear(qf, T_WH)
    _, bar = PG.handle_shapes(T_WH)
    W = DOC.world_points(qf, DOC.ARM_LINKS)
    names = ['uflite_gripper_link'] + ([] if contact else list(DOC.FINGERS))
    d, who = min((float(PG.bar_dist(W[n], bar).min()), n) for n in names if n in W)
    cc['bar_tool'], cc['bar_tool_who'] = d, who
    cc['ok'] = bool(cc['ok'] and d >= R1.ENV_M)
    return cc


# ---- b 停止取樣（梯形）
def stop_traj(qf, qd, extra, dt=0.04):
    tl = R1.LAT + (R1.DT_CTRL if extra else 0.0)
    out, q, v, t = [], qf.copy(), qd.copy(), 0.0
    while t < tl - 1e-9:
        h = min(dt, tl - t)
        q = q + v * h
        t += h
        out.append(q.copy())
    for _ in range(400):
        c_, s_ = math.cos(q[2]), math.sin(q[2])
        vb = np.array([c_ * v[0] + s_ * v[1], -s_ * v[0] + c_ * v[1], v[2]])
        vb1 = vb - np.clip(vb, -R1.A_BASE * dt, R1.A_BASE * dt)
        va1 = v[IDX] - np.clip(v[IDX], -R1.A_ARM * dt, R1.A_ARM * dt)
        v1 = np.zeros(9)
        v1[0], v1[1], v1[2] = c_ * vb1[0] - s_ * vb1[1], s_ * vb1[0] + c_ * vb1[1], vb1[2]
        v1[IDX] = va1
        q = q + 0.5 * (v + v1) * dt                                # 梯形
        v = v1
        out.append(q.copy())
        if float(np.abs(v).max()) < 1e-6:
            break
    return out


def seg_check(base0, q0, b1, q1, T_WH, tag):
    """有界同步段：界回核、路徑全碰撞、停止取樣全碰撞。回 dict（ok、T、最小淨空）與 f。"""
    T, _ = R2.bounded_T(base0, q0, b1, q1)
    _, f = R1.schedule('sync', base0, q0, b1, q1, T, T)
    vb = R2.verify_bounds(f, T)
    if not vb['ok']:
        return dict(ok=False, why=tag + ':bounds', T=round(T, 3)), None
    mn = dict(env=np.inf, base=np.inf, self=np.inf, bar_tool=np.inf)
    for t in np.linspace(0.0, T, 21):
        cc = cfg_clear_full(f(t)[0], T_WH)
        for k_ in mn:
            mn[k_] = min(mn[k_], cc[k_])
        if not cc['ok']:
            return dict(ok=False, why=tag + ':path', T=round(T, 3), at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()}), None
    sm, n = dict(env=np.inf, base=np.inf, self=np.inf, bar_tool=np.inf), 0
    for t in np.arange(0.0, T + 1e-9, 0.15):
        qf, qd = f(t)
        for extra in (0, 1):
            for qs in stop_traj(qf, qd, extra)[::2]:
                cc = cfg_clear_full(qs, T_WH)
                n += 1
                for k_ in sm:
                    sm[k_] = min(sm[k_], cc[k_])
                if not cc['ok']:
                    return dict(ok=False, why=tag + ':stop_sweep', T=round(T, 3), at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()}), None
    return dict(ok=True, T=round(T, 3), bounds=vb, path_min={k: round(v, 4) for k, v in mn.items()}, stop_min={k: round(v, 4) for k, v in sm.items()}, stop_n=n), f


# ---- 可見性（幾何）
def K_of(hfov, w=640, h=480):
    fx = (w / 2) / math.tan(hfov / 2)
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])


K_BASE = K_of(VZ.CAMS['base']['hfov'])


def see(Tc, Kint, T_WH, qf, parent, rng=D_RNG):
    """把手兩端、中心、深度探測區在畫面內（離邊 EDGE）＋距離＋自遮擋。回 dict。"""
    _, (c, u, hl, r_) = PG.handle_shapes(T_WH)
    up = np.array([0.0, 0.0, 1.0])
    pts = dict(center=c, end_a=c + hl * u, end_b=c - hl * u, probe_up=c + PROBE * up + 0.04 * T_WH[:3, 1], probe_dn=c - PROBE * up + 0.04 * T_WH[:3, 1])
    Ti = np.linalg.inv(Tc)
    W = DOC.world_points(qf, DOC.ARM_LINKS)
    cam = Tc[:3, 3]
    res, ok, dmin, why = {}, True, np.inf, None
    for n, p in pts.items():
        pc = Ti @ np.r_[p, 1.0]
        z = float(pc[2])
        uv = (Kint @ (pc[:3] / max(z, 1e-9)))[:2] if z > 1e-6 else np.array([-1e3, -1e3])
        inimg = bool(z > 0 and EDGE <= uv[0] <= 640 - EDGE and EDGE <= uv[1] <= 480 - EDGE)
        dist = float(np.linalg.norm(pc[:3]))
        blk = None
        if inimg:
            d = p - cam
            L = float(np.linalg.norm(d))
            d /= L
            for ln, P in W.items():
                if ln == parent:
                    continue
                s = np.clip((P - cam) @ d, 0.03, L - 0.01)
                dd = float(np.linalg.norm(P - (cam + s[:, None] * d), axis=1).min())
                if dd < RAY_CLR:
                    blk = ln
                    break
        good = inimg and rng[0] <= dist <= rng[1] and blk is None
        if not good and why is None:
            why = f'{n}:' + ('out_of_image' if not inimg else 'range' if not (rng[0] <= dist <= rng[1]) else f'occluded_by_{blk}')
        ok = ok and good
        dmin = min(dmin, dist)
        res[n] = dict(uv=[round(float(v), 1) for v in uv], dist=round(dist, 3), ok=bool(good))
    return dict(ok=bool(ok), why=why, dist_min=round(float(dmin), 3), pts=res)


def cam_targets(T_WH):
    c, R = T_WH[:3, 3], T_WH[:3, :3]
    n = -R[:3, 1] / np.linalg.norm(R[:3, 1])                      # 門外法向
    n[2] = 0.0
    n /= np.linalg.norm(n)
    u = R[:3, 0] / np.linalg.norm(R[:3, 0])
    out = []
    for d in DCAMS:
        for el in ELEVS:
            for az in AZIS:
                e, a = math.radians(el), math.radians(az)
                hz = math.cos(a) * n + math.sin(a) * np.cross(np.array([0, 0, 1.0]), n)
                dirv = math.cos(e) * hz + math.sin(e) * np.array([0, 0, 1.0])
                p = c + d * dirv
                z = (c - p) / np.linalg.norm(c - p)
                x = u - (u @ z) * z
                x /= np.linalg.norm(x)
                for sgn in (1.0, -1.0):                          # 影像水平軸沿桿的兩個方向
                    xx = sgn * x
                    T = np.eye(4)
                    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xx, np.cross(z, xx), z, p
                    out.append(dict(T=T, d=d, el=el, az=az, flip=sgn < 0))
    return out


def plan3(T_WH, T_pre, T_gr, T_WE_at, base0, q_obs, T_mount, Kint):
    t0 = time.perf_counter()
    base_sel = R2.plan_viewpoint2(T_WH, T_pre, T_gr, T_WE_at, base0, q_obs, T_mount, Kint)
    t_stand = time.perf_counter() - t0
    out = dict(standoff_ok=base_sel['ok'], t_standoff_s=round(t_stand, 2))
    if not base_sel['ok']:
        return dict(out, ok=False, why='standoff:' + str(base_sel.get('why')))
    b1 = np.array(base_sel['chosen']['base'])
    q_pre, q_gr = np.array(base_sel['_q']['q_pre']), np.array(base_sel['_q']['q_gr'])
    out['standoff'] = dict(base=b1.round(4).tolist(), T_direct=base_sel['path']['T'])
    qf0 = np.zeros(9)
    qf0[:3], qf0[IDX] = base0, q_obs
    T_tcp_cam = np.linalg.inv(K.fk(qf0, 'link_tcp')) @ K.fk(qf0, 'link_eef') @ T_mount
    T_cam_tcp = np.linalg.inv(T_tcp_cam)
    # r2 直接接近時，全碰撞對下的預抓構型與底盤相機
    cpre = cfg_clear_full(q_pre, T_WH)
    out['pregrasp_full_collision'] = dict(ok=cpre['ok'], bar_tool=round(cpre['bar_tool'], 4), who=cpre['bar_tool_who'])
    Tb = K.fk(q_pre, 'base_link') @ VZ.CAMS['base']['T']
    sb = see(Tb, K_BASE, T_WH, q_pre, 'base_link', rng=(VZ.CAMS['base']['near'], 3.0))
    out['base_cam_at_pregrasp'] = dict(ok=sb['ok'], why=sb['why'], dist_min=sb['dist_min'])
    rej, cands, n_ik = {}, [], 0
    db = np.r_[b1[:2] - base0[:2], R1.wrap(b1[2] - base0[2])]
    for lam in LAMS:
        bl = base0 + lam * db
        if R1.base_clear(bl) < R1.BASE_M:
            rej['base'] = rej.get('base', 0) + 1
            continue
        seed = q_obs if lam < 0.9 else q_pre[IDX]
        for ct in cam_targets(T_WH):
            n_ik += 1
            ql = PG.ik(bl, seed, ct['T'] @ T_cam_tcp, extra_seeds=True)
            if ql is None:
                rej['ik'] = rej.get('ik', 0) + 1
                continue
            if DOC.jmargin(ql[IDX])[0] < R1.LIM_M:
                rej['joint'] = rej.get('joint', 0) + 1
                continue
            cc = cfg_clear_full(ql, T_WH)
            if not cc['ok']:
                rej['collision'] = rej.get('collision', 0) + 1
                continue
            Tc = K.fk(ql, 'link_eef') @ T_mount
            sv = see(Tc, Kint, T_WH, ql, 'link_eef')
            if not sv['ok']:
                key = 'vis:' + sv['why'].split(':')[1]
                rej[key] = rej.get(key, 0) + 1
                continue
            T1, _ = R2.bounded_T(base0, q_obs, bl, ql[IDX])
            T2, _ = R2.bounded_T(bl, ql[IDX], b1, q_pre[IDX])
            cands.append(dict(lam=lam, bl=bl, ql=ql, cam=ct, vis=sv, T_est=T1 + T_LOOK + T2))
    out.update(n_look_ik=n_ik, n_look_visible=len(cands), look_rejected=rej)
    cands.sort(key=lambda z: z['T_est'])
    tried = []
    for cd in cands[:6]:
        s1, _ = seg_check(base0, q_obs, cd['bl'], cd['ql'][IDX], T_WH, 'to_look')
        if not s1['ok']:
            tried.append(dict(lam=cd['lam'], d=cd['cam']['d'], el=cd['cam']['el'], az=cd['cam']['az'], ok=False, why=s1['why']))
            continue
        s2, _ = seg_check(cd['bl'], cd['ql'][IDX], b1, q_pre[IDX], T_WH, 'to_pre')
        if not s2['ok']:
            tried.append(dict(lam=cd['lam'], d=cd['cam']['d'], el=cd['cam']['el'], az=cd['cam']['az'], ok=False, why=s2['why']))
            continue
        # 預抓→抓取：指定接觸段（只豁免手指對橫桿）
        ap_ok, ap_min, prev = True, np.inf, q_pre
        for s in np.linspace(PG.S_PRE, 0.0, 5):
            q = PG.ik(b1, prev[IDX], T_WE_at(s), extra_seeds=False)
            if q is None:
                ap_ok = False
                break
            cc = cfg_clear_full(q, T_WH, contact=True)
            ap_min = min(ap_min, cc['bar_tool'])
            ap_ok = ap_ok and cc['ok']
            prev = q
        tcp_l = K.fk(cd['ql'], 'link_tcp')[:3, 3]
        tcp_p, tcp_g = K.fk(q_pre, 'link_tcp')[:3, 3], K.fk(q_gr, 'link_tcp')[:3, 3]
        Tb2 = K.fk(cd['ql'], 'base_link') @ VZ.CAMS['base']['T']
        sb2 = see(Tb2, K_BASE, T_WH, cd['ql'], 'base_link', rng=(VZ.CAMS['base']['near'], 3.0))
        t_ins = 1.0                                                # 預抓→抓取時長（既有插入時長）
        tried.append(dict(lam=cd['lam'], d=cd['cam']['d'], el=cd['cam']['el'], az=cd['cam']['az'], ok=bool(ap_ok)))
        if not ap_ok:
            continue
        return dict(out, ok=True, chosen=dict(lam=cd['lam'], base_look=cd['bl'].round(4).tolist(), cam_dist=cd['cam']['d'], elev_deg=cd['cam']['el'], azi_deg=cd['cam']['az'], flip=cd['cam']['flip'],
                                              vis=cd['vis'], base_cam_at_look=dict(ok=sb2['ok'], why=sb2['why'], dist_min=sb2['dist_min'])),
                    seg_to_look=s1, seg_to_pre=s2, approach=dict(ok=True, shell_bar_min=round(float(ap_min), 4)),
                    T_total=round(s1['T'] + T_LOOK + s2['T'] + t_ins, 3), T_direct_plus_insert=round(base_sel['path']['T'] + t_ins, 3),
                    last_obs=dict(base=cd['bl'].round(3).tolist(), cam_to_handle=cd['vis']['pts']['center']['dist'], base_to_standoff_m=round(float(np.linalg.norm(b1[:2] - cd['bl'][:2])), 3)),
                    blind_after=dict(tcp_travel_m=round(float(np.linalg.norm(tcp_p - tcp_l) + np.linalg.norm(tcp_g - tcp_p)), 3), time_s=round(s2['T'] + t_ins, 3)), tried=tried, _q=dict(q_gr=q_gr.tolist()))
    return dict(out, ok=False, why='no_look_candidate' if not cands else 'no_look_path', tried=tried)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--group', choices=('dev', 'holdout'), default='dev')
    ap.add_argument('--limit', type=int, default=0)
    aa = ap.parse_args()
    D = os.path.join(HERE, 'runs', aa.run, 'wrist_door')
    det = json.load(open(os.path.join(D, f'door_detect_{aa.group}.json')))
    meta = json.load(open(os.path.join(D, 'meta.json')))
    reg = json.load(open(os.path.join(R2.WS, meta['registration'])))
    VP = {v['id']: v for v in reg['viewpoints']}
    FR = {f['stem']: f for f in meta['frames']}
    Kint = np.array(meta['intrinsics_readback']['K'])
    Rg, T_HG, a_H = PG.baseline_reference()
    rows, t0 = [], time.perf_counter()
    # 步長收斂核對（一次）：同一狀態 dt 0.04 對 0.01 的停止終點差
    qf = np.zeros(9)
    qd = np.zeros(9)
    qd[0], qd[1], qd[2] = 0.2, 0.1, 0.3
    qd[IDX] = 0.6
    conv = float(np.abs(stop_traj(qf, qd, 1, 0.04)[-1] - stop_traj(qf, qd, 1, 0.01)[-1]).max())
    conv_old = float(np.abs(R2.stop_traj(qf, qd, 1, 0.04)[-1] - stop_traj(qf, qd, 1, 0.01)[-1]).max())
    print(f'停止積分收斂：梯形 dt 0.04 對 0.01 終點差 {conv:.5f}；r2 積分對參考差 {conv_old:.5f}', flush=True)
    for v in det['per_viewpoint']:
        rr = [r for r in det['rows'] if r['viewpoint'] == v['viewpoint'] and r['cls'] == 'center_valid']
        if v['n_center_valid'] < 2 or not rr:
            continue
        if aa.limit and len(rows) >= aa.limit:
            break
        r = rr[0]
        f = FR[r['stem']]
        dep = np.load(os.path.join(D, 'frames', f"{r['stem']}_depth_m.npy"))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        obs = dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=Kint, T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time'])
        est = E.estimate(obs, 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
        row = dict(viewpoint=v['viewpoint'], est_ok=est['ok'], obs_dist=r.get('dist_cam_handle_m'))
        if not est['ok']:
            rows.append(row)
            continue
        T_WH = np.asarray(est['T_WH'], float)

        def T_WE_at(s, T_WH=T_WH):
            return np.asarray(E.OT.object_target(T_WH, T_HG, a_H, float(s))['T_WE'], float)
        vp = VP[v['viewpoint']]
        base0, q_obs = np.array(vp['base'], float), np.array(vp['q_arm'], float)
        qf0 = np.zeros(9)
        qf0[:3], qf0[IDX] = base0, q_obs
        T_mount = np.linalg.inv(K.fk(qf0, 'link_eef')) @ np.array(f['cam_T_fk'])
        w0 = time.perf_counter()
        res = plan3(T_WH, T_WE_at(PG.S_PRE), T_WE_at(0.0), T_WE_at, base0, q_obs, T_mount, Kint)
        res['wall_s'] = round(time.perf_counter() - w0, 2)
        res.pop('_q', None)
        row.update(res)
        rows.append(row)
        ch = row.get('chosen') or {}
        print(f"{v['viewpoint']}: ok {row['ok']} {row.get('why', '')} 可見重瞄 {row.get('n_look_visible')}／{row.get('n_look_ik')} 拒 {row.get('look_rejected')} 選 λ {ch.get('lam')} 距 {ch.get('cam_dist')} 仰 {ch.get('elev_deg')} 方位 {ch.get('azi_deg')} "
              f"總 {row.get('T_total')}（直接 {row.get('T_direct_plus_insert')}）盲段 {row.get('blind_after')} 預抓全碰撞 {row.get('pregrasp_full_collision')} {row['wall_s']} s", flush=True)
    okr = [r for r in rows if r.get('ok')]
    S = dict(run=aa.run, group=aa.group, n=len(rows), standoff_ok=sum(1 for r in rows if r.get('standoff_ok')), reaim_ok=len(okr),
             fail_why={w: sum(1 for r in rows if not r.get('ok') and str(r.get('why', '')).startswith(w)) for w in ('standoff', 'no_look_candidate', 'no_look_path')},
             look_rejected_total={k: sum(r.get('look_rejected', {}).get(k, 0) for r in rows) for k in sorted({k for r in rows for k in r.get('look_rejected', {})})},
             pregrasp_full_collision_ok=sum(1 for r in rows if (r.get('pregrasp_full_collision') or {}).get('ok')), pregrasp_bar_tool_min=min((r['pregrasp_full_collision']['bar_tool'] for r in rows if r.get('pregrasp_full_collision')), default=None),
             base_cam_at_pregrasp_ok=sum(1 for r in rows if (r.get('base_cam_at_pregrasp') or {}).get('ok')), base_cam_at_look_ok=sum(1 for r in okr if r['chosen']['base_cam_at_look']['ok']),
             T_total=sorted(r['T_total'] for r in okr), T_direct_plus_insert=sorted(r['T_direct_plus_insert'] for r in okr), extra_time=sorted(round(r['T_total'] - r['T_direct_plus_insert'], 2) for r in okr),
             last_obs_cam_dist=sorted(r['last_obs']['cam_to_handle'] for r in okr), blind_tcp_travel=sorted(r['blind_after']['tcp_travel_m'] for r in okr), blind_time=sorted(r['blind_after']['time_s'] for r in okr),
             chosen_lam=sorted(r['chosen']['lam'] for r in okr), stop_min={k: min(min(r['seg_to_look']['stop_min'][k], r['seg_to_pre']['stop_min'][k]) for r in okr) for k in ('env', 'base', 'self', 'bar_tool')} if okr else None,
             approach_shell_bar_min=min((r['approach']['shell_bar_min'] for r in okr), default=None), stop_conv=dict(trapezoid_dt004_vs_dt001=round(conv, 5), r2_vs_ref=round(conv_old, 5)),
             compute_s=dict(per_viewpoint=sorted(r['wall_s'] for r in rows if 'wall_s' in r), standoff_part=sorted(r['t_standoff_s'] for r in rows if 't_standoff_s' in r)), wall_s=round(time.perf_counter() - t0, 1),
             params=dict(T_LOOK=T_LOOK, EDGE=EDGE, D_RNG=D_RNG, RAY_CLR=RAY_CLR, LAMS=LAMS, DCAMS=DCAMS, ELEVS=ELEVS, AZIS=AZIS))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(D, f'door_reaim_online_r3_{aa.group}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in S.items() if k != 'params'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
