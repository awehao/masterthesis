#!/usr/bin/env python3
"""門把手視覺 → 在線停位與全身接近 r2（Codex 20261112：回完整場景主線；低負載離線，不跑模擬、不閉爪、不開門）。
在 r1（m8_door_standoff_online.py；候選生成、IK、選取規則不變）上補門包缺口：
  1 環境：接回完整場景 r2 的已知幾何（mission_room_m8_fullwalls.sdf 的外牆、隔牆、家具；門扇與門洞牆同 r1）。未知箱體／行人**不放進模型**
    （本擷取沒有掃描；場景生成座標不能餵進規劃）——列為缺口，只在事後以場景登錄算「若未知物在場，所選停位與路徑離它多遠」（評估用，規劃不讀）。
  2 速度／加速度界：同步 min-jerk 的時長由各界反推（底盤平移 V_BASE、偏航 W_BASE、底盤加速度 A_SLEW、偏航角加速度 ALPHA_MAX、輪級 wheel_matrix×margin、
    手臂 V_ARM／A_ARM），再逐取樣數值回核；不過界才採用。r1 的 T 只是距離／速度上限，min-jerk 峰值 1.875 倍會超界。
  3 完整停止取樣：每 0.1 s 的路徑點，現在停／晚一拍停，**停止過程每 0.04 s 取樣**（在途延遲續行＋各軸最大減速），逐構型核淨空（r1 只核終點）。
  4 近距重瞄可見性（幾何）：沿路徑與預抓構型，把手中心在手腕相機畫面內且距離在 [NEAR, FAR]；不核遮擋。另報預抓時相機到把手距離
    （既有結果：0.7 m 近距視點偵測全部無效中心 ⇒ 距離落在已知失效區就標記）。
  5 接觸准入：估計誤差界取**開發集**沿接近軸誤差最大值（事前、不看保留集），與 DL3 根部名目餘裕 2.7 mm 比；保留集若有任一案超過該界 ⇒ 界未被驗證 ⇒ 遠距估計不得放行閉爪。
真值只在最後安全網與誤差統計使用；規劃、選擇、准入規則都不讀。
    python3 evaluation/m8_door_standoff_online_r2.py m8doorcap_1 --group dev|holdout"""
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import m8_door_standoff_online as R1                              # noqa: E402
import bottle_feasible_profile as FP                              # noqa: E402

PG, DOC, E, K, IDX = R1.PG, R1.DOC, R1.E, R1.K, R1.IDX
A_LIN, ALPHA = 0.4, 1.0                                           # 底盤參考加速度（S1 A_SLEW）、偏航角加速度（航向層 ALPHA_MAX）
MJ_V, MJ_A = 1.875, 5.7735                                        # min-jerk 峰值速度、峰值加速度係數
CAM_NEAR, CAM_FAR, DET_BAD_BELOW = 0.20, 2.0, 0.80                # 可見距離（幾何）；0.7 m 視點偵測全部無效 ⇒ < 0.80 m 標為已知偵測失效區（1.2 m 起才有有效估計）
ROOT_MARGIN_MM = 2.7                                              # DL3 根部名目餘裕
print('#EARLY r2', dict(A_LIN=A_LIN, ALPHA=ALPHA, margin=FP.MARGIN), flush=True)

# ---- 1 完整場景已知幾何
REG = json.load(open(os.path.join(HERE, 'experiments', os.environ.get('M8_SCENE_REG', '20261111_m8_full_scene.json'))))   # 預設＝舊兩房場景（r2／r3 結果的前提）；住宅場景設 M8_SCENE_REG=20261112_m8_house_scene.json


def scene_known():
    raw = re.sub(r'<!--.*?-->', '', open(os.path.join(WS, REG['world']['path']), encoding='utf-8').read(), flags=re.S)
    out = []
    for m in ET.fromstring(raw).find('world').findall('model'):
        b = m.find('.//collision/geometry/box')
        if b is None or m.get('name') == 'ground_plane':
            continue
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        sz = [float(v) for v in b.findtext('size').split()]
        if m.get('name') in ('partition_w', 'partition_e'):       # 隔牆由門資產的門洞牆（r1 static_shapes）代表，避免重複與尺寸不一
            continue
        yaw = pose[5]
        c_, s_ = math.cos(yaw), math.sin(yaw)
        out.append(('scene:' + m.get('name'), np.array(pose[:3]), np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]]), np.array(sz) / 2))
    return out


SCENE = scene_known()
_ss0 = PG.static_shapes
PG.static_shapes = lambda: _ss0() + SCENE                         # r1 的 base_clear／env_eval 都經 PG.static_shapes
UNKNOWN = json.load(open(os.path.join(WS, REG['obstacles']['path'])))


def unknown_clear(b):
    """評估用（規劃不讀）：底盤圓盤到場景登錄的未知箱體、行人路線的最小距離。"""
    d = []
    for o in UNKNOWN['static_unknown']:
        d.append((float(np.hypot(b[0] - o['x'], b[1] - o['y']) - 0.5 * math.hypot(o['size'][0], o['size'][1]) - DOC.RB), o['name']))
    for w in UNKNOWN['walkers']:
        p0, p1 = np.array(w['p0']), np.array(w['p1'])
        u = np.clip(np.dot(np.array(b[:2]) - p0, p1 - p0) / max(np.dot(p1 - p0, p1 - p0), 1e-9), 0, 1)
        d.append((float(np.linalg.norm(np.array(b[:2]) - (p0 + u * (p1 - p0))) - w['radius'] - DOC.RB), w['name'] + '_route'))
    return min(d)


# ---- 2 速度／加速度界
def bounded_T(base0, q_obs, b1, q_pre):
    db = np.r_[b1[:2] - base0[:2], R1.wrap(b1[2] - base0[2])]
    dq = np.abs(q_pre - q_obs)
    cap, W = FP.wheel_cap()
    L = float(np.linalg.norm(db[:2]))
    T = max(MJ_V * L / R1.V_BASE, MJ_V * abs(db[2]) / R1.W_BASE, math.sqrt(MJ_A * L / A_LIN), math.sqrt(MJ_A * abs(db[2]) / ALPHA),
            MJ_V * float(dq.max()) / R1.V_ARM, math.sqrt(MJ_A * float(dq.max()) / R1.A_ARM), 0.3)
    for _ in range(6):                                            # 輪級：逐取樣本體速度，超過 margin 就等比拉長
        _, f = R1.schedule('sync', base0, q_obs, b1, q_pre, T, T)
        use = 0.0
        for t in np.linspace(0, T, 41):
            qf, qd = f(t)
            c_, s_ = math.cos(qf[2]), math.sin(qf[2])
            vb = np.array([c_ * qd[0] + s_ * qd[1], -s_ * qd[0] + c_ * qd[1], qd[2]])
            use = max(use, float(np.max(np.abs(W @ vb))) / cap)
        if use <= FP.MARGIN + 1e-6:
            break
        T *= use / FP.MARGIN
    return T, use


def verify_bounds(f, T):
    cap, W = FP.wheel_cap()
    ts = np.linspace(0, T, 201)
    Q = np.array([f(t)[1] for t in ts])
    vb = np.hypot(Q[:, 0], Q[:, 1])
    acc = np.abs(np.diff(Q, axis=0)) / (ts[1] - ts[0])
    use = 0.0
    for t in ts:
        qf, qd = f(t)
        c_, s_ = math.cos(qf[2]), math.sin(qf[2])
        use = max(use, float(np.max(np.abs(W @ np.array([c_ * qd[0] + s_ * qd[1], -s_ * qd[0] + c_ * qd[1], qd[2]])))) / cap)
    out = dict(v_base=float(vb.max()), w_base=float(np.abs(Q[:, 2]).max()), a_base=float(np.hypot(acc[:, 0], acc[:, 1]).max()), alpha=float(acc[:, 2].max()),
               v_arm=float(np.abs(Q[:, IDX]).max()), a_arm=float(acc[:, IDX].max()), wheel=use)
    out['ok'] = bool(out['v_base'] <= R1.V_BASE * 1.001 and out['w_base'] <= R1.W_BASE * 1.001 and out['a_base'] <= A_LIN * 1.02 and out['alpha'] <= ALPHA * 1.02
                     and out['v_arm'] <= R1.V_ARM * 1.001 and out['a_arm'] <= R1.A_ARM and out['wheel'] <= 1.0)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


# ---- 3 完整停止取樣
def stop_traj(qf, qd, extra, dt=0.04):
    tl = R1.LAT + (R1.DT_CTRL if extra else 0.0)
    out = []
    q, v = qf.copy(), qd.copy()
    t = 0.0
    while t < tl - 1e-9:                                           # 在途命令續行
        h = min(dt, tl - t)
        q = q + v * h
        t += h
        out.append(q.copy())
    for _ in range(200):
        c_, s_ = math.cos(q[2]), math.sin(q[2])
        vb = np.array([c_ * v[0] + s_ * v[1], -s_ * v[0] + c_ * v[1], v[2]])
        vb = vb - np.clip(vb, -R1.A_BASE * dt, R1.A_BASE * dt)
        va = v[IDX] - np.clip(v[IDX], -R1.A_ARM * dt, R1.A_ARM * dt)
        v = np.zeros(9)
        v[0], v[1], v[2] = c_ * vb[0] - s_ * vb[1], s_ * vb[0] + c_ * vb[1], vb[2]
        v[IDX] = va
        q = q + v * dt
        out.append(q.copy())
        if float(np.abs(v).max()) < 1e-6:
            break
    return out


def path_check2(base0, q_obs, b1, q_pre, T_WH):
    T, use = bounded_T(base0, q_obs, b1, q_pre)
    _, f = R1.schedule('sync', base0, q_obs, b1, q_pre, T, T)
    vb = verify_bounds(f, T)
    if not vb['ok']:
        return dict(ok=False, why='bounds', T=round(T, 3), bounds=vb)
    worst = dict(env=np.inf, base=np.inf, self=np.inf)
    for t in np.linspace(0.0, T, 31):
        cc = R1.cfg_clear(f(t)[0], T_WH)
        for k_ in worst:
            worst[k_] = min(worst[k_], cc[k_])
        if not cc['ok']:
            return dict(ok=False, why='path', T=round(T, 3), t_fail=round(float(t), 3), at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()}, bounds=vb)
    stop = dict(env=np.inf, base=np.inf, self=np.inf, n=0, who=None)
    for t in np.arange(0.0, T + 1e-9, 0.1):
        qf, qd = f(t)
        for extra in (0, 1):
            for qs in stop_traj(qf, qd, extra):
                cc = R1.cfg_clear(qs, T_WH)
                stop['n'] += 1
                for k_ in ('env', 'base', 'self'):
                    if cc[k_] < stop[k_]:
                        stop[k_] = cc[k_]
                        if k_ == 'env':
                            stop['who'] = cc['env_who']
                if not cc['ok']:
                    return dict(ok=False, why='stop_sweep', T=round(T, 3), stop_fail=dict(t=round(float(t), 3), extra=extra, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()}), bounds=vb)
    return dict(ok=True, T=round(T, 3), bounds=vb, path_min={k: round(v, 4) for k, v in worst.items()}, stop_min={k: (round(v, 4) if isinstance(v, float) else v) for k, v in stop.items()}, _f=f)


# ---- 4 近距重瞄可見性（幾何）
def visibility(f, T, T_WH, T_mount, Kint, q_gr=None):
    c = T_WH[:3, 3]
    rows = []
    for fr in (0.0, 0.25, 0.5, 0.75, 1.0):
        qf, _ = f(fr * T)
        Tc = K.fk(qf, 'link_eef') @ T_mount
        p = np.linalg.inv(Tc) @ np.r_[c, 1.0]
        z = float(p[2])
        uv = (Kint @ (p[:3] / max(z, 1e-9)))[:2] if z > 0 else np.array([np.nan, np.nan])
        inimg = bool(z > 0 and 0 <= uv[0] < 640 and 0 <= uv[1] < 480)
        rows.append(dict(frac=fr, dist=round(float(np.linalg.norm(p[:3])), 3), in_image=inimg, in_range=bool(CAM_NEAR <= z <= CAM_FAR), uv=[round(float(v), 1) for v in uv]))
    last = rows[-1]
    return dict(samples=rows, visible_frac=round(float(np.mean([r['in_image'] and r['in_range'] for r in rows])), 2), pregrasp_visible=bool(last['in_image'] and last['in_range']),
                pregrasp_dist=last['dist'], pregrasp_in_known_bad_detect_zone=bool(last['dist'] < DET_BAD_BELOW), note='幾何（畫面內＋距離）；未核遮擋；偵測在近距是否有效另見既有 0.7 m 結果')


def plan_viewpoint2(T_WH, T_pre, T_gr, T_WE_at, base0, q_obs, T_mount, Kint):
    cands = R1.candidates(T_WH, T_pre, base0)
    feas, n_ik, rej = [], 0, {}
    for cd in cands:
        if n_ik >= R1.N_IK or len(feas) >= R1.K_FEAS:
            break
        n_ik += 1
        sol, info = PG.plan_at(cd['b'], q_obs, T_WH, T_pre, T_gr)
        if sol is None:
            rej['ik'] = rej.get('ik', 0) + 1
            continue
        q_pre, q_gr = sol
        if DOC.self_clear(q_pre) < R1.SELF_M or DOC.self_clear(q_gr) < R1.SELF_M:
            rej['self_at_goal'] = rej.get('self_at_goal', 0) + 1
            continue
        Tb, _ = bounded_T(base0, q_obs, cd['b'], q_pre[IDX])
        feas.append(dict(cd, q_pre=q_pre, q_gr=q_gr, T=Tb, margin=min(info['pre']['joint_margin'], info['grasp']['joint_margin'])))
    feas.sort(key=lambda z: (round(z['T'], 2), -z['margin']))
    tried = []
    for fz in feas:
        pc = path_check2(base0, q_obs, fz['b'], fz['q_pre'][IDX], T_WH)
        tried.append(dict(d=fz['d'], s=fz['s'], dpsi_deg=fz['dpsi_deg'], ok=pc['ok'], why=pc.get('why'), T=pc['T']))
        if not pc['ok']:
            rej[pc['why']] = rej.get(pc['why'], 0) + 1
            continue
        f = pc.pop('_f')
        ap = PG.sweeps(fz['b'], q_obs, fz['q_pre'], T_WH, T_WE_at)
        vis = visibility(f, pc['T'], T_WH, T_mount, Kint)
        uc = unknown_clear(fz['b'])
        return dict(ok=True, chosen=dict(base=fz['b'].round(4).tolist(), d=fz['d'], s=fz['s'], dpsi_deg=fz['dpsi_deg'], margin=round(fz['margin'], 4)), path=pc, approach=dict(ok=ap['approach_ok'], env=ap['approach_env']),
                    visibility=vis, unknown_eval=dict(clear_m=round(uc[0], 3), who=uc[1], note='評估用：場景登錄的未知物（規劃不讀）'), n_candidates=len(cands), n_ik=n_ik, n_feasible=len(feas), rejected=rej, tried=tried,
                    _q=dict(q_pre=fz['q_pre'].tolist(), q_gr=fz['q_gr'].tolist()))
    return dict(ok=False, why='no_candidate' if not cands else ('no_ik_feasible' if not feas else 'no_path'), n_candidates=len(cands), n_ik=n_ik, n_feasible=len(feas), rejected=rej, tried=tried)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--group', choices=('dev', 'holdout'), default='dev')
    aa = ap.parse_args()
    D = os.path.join(HERE, 'runs', aa.run, 'wrist_door')
    det = json.load(open(os.path.join(D, f'door_detect_{aa.group}.json')))
    meta = json.load(open(os.path.join(D, 'meta.json')))
    reg = json.load(open(os.path.join(WS, meta['registration'])))
    VP = {v['id']: v for v in reg['viewpoints']}
    FR = {f['stem']: f for f in meta['frames']}
    Kint = np.array(meta['intrinsics_readback']['K'])
    Rg, T_HG, a_H = PG.baseline_reference()
    rows, t0 = [], time.perf_counter()
    for v in det['per_viewpoint']:
        rr = [r for r in det['rows'] if r['viewpoint'] == v['viewpoint'] and r['cls'] == 'center_valid']
        if v['n_center_valid'] < 2 or not rr:
            continue
        r = rr[0]
        f = FR[r['stem']]
        dep = np.load(os.path.join(D, 'frames', f"{r['stem']}_depth_m.npy"))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        obs = dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=Kint, T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time'])
        est = E.estimate(obs, 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
        row = dict(viewpoint=v['viewpoint'], est_ok=est['ok'])
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
        T_mount = np.linalg.inv(K.fk(qf0, 'link_eef')) @ np.array(f['cam_T_fk'])       # 相機外參由該影格的 FK 相機位姿反推（同擷取端）
        w0 = time.perf_counter()
        res = plan_viewpoint2(T_WH, T_WE_at(PG.S_PRE), T_WE_at(0.0), T_WE_at, base0, q_obs, T_mount, Kint)
        res['wall_s'] = round(time.perf_counter() - w0, 2)
        row.update(res)
        rows.append(row)
        ch, pth = row.get('chosen') or {}, row.get('path') or {}
        print(f"{v['viewpoint']}: ok {row['ok']} {row.get('why', '')} 可行 {row['n_feasible']} 拒 {row.get('rejected')} 選 d {ch.get('d')} s {ch.get('s')} δψ {ch.get('dpsi_deg')} T {pth.get('T')} "
              f"停止掃掠 {pth.get('stop_min', {}).get('env')} 可見 {row.get('visibility', {}).get('visible_frac')} 預抓距 {row.get('visibility', {}).get('pregrasp_dist')} {row['wall_s']} s", flush=True)
    truth = json.load(open(os.path.join(D, 'truth.json')))                # 真值：規劃、選擇完成後才讀
    for row in rows:
        if '_q' not in row:
            continue
        Tt = np.array(truth['per_viewpoint'][row['viewpoint']]['M_world_handle_prim'])
        qg = np.array(row['_q']['q_gr'])
        T_true = np.asarray(E.OT.object_target(Tt, T_HG, a_H, 0.0)['T_WE'], float)
        e = K.fk(qg, 'link_tcp')[:3, 3] - T_true[:3, 3]
        row['D_truth'] = dict(grasp_ok=PG.feasible(qg, Tt)[0], tcp_err_mm=round(float(np.linalg.norm(e)) * 1e3, 2), along_z_mm=round(float(e @ T_true[:3, 2]) * 1e3, 2))
        row.pop('_q')
    okr = [r for r in rows if r.get('ok')]
    az = [r['D_truth']['along_z_mm'] for r in rows if 'D_truth' in r]
    S = dict(run=aa.run, group=aa.group, n_valid_center=len(rows), est_ok=sum(1 for r in rows if r['est_ok']), ok=len(okr), approach_ok=sum(1 for r in okr if r['approach']['ok']),
             fail_why={w: sum(1 for r in rows if r.get('est_ok') and not r.get('ok') and r.get('why') == w) for w in ('no_candidate', 'no_ik_feasible', 'no_path')},
             rejected_total={k: sum(r.get('rejected', {}).get(k, 0) for r in rows) for k in ('ik', 'self_at_goal', 'bounds', 'path', 'stop_sweep')},
             T_s=sorted(r['path']['T'] for r in okr), bounds_max={k: max(r['path']['bounds'][k] for r in okr) for k in ('v_base', 'w_base', 'a_base', 'alpha', 'v_arm', 'a_arm', 'wheel')} if okr else None,
             path_env_min=min((r['path']['path_min']['env'] for r in okr), default=None), stop_sweep_min={k: min(r['path']['stop_min'][k] for r in okr) for k in ('env', 'base', 'self')} if okr else None,
             stop_sweep_n=sum(r['path']['stop_min']['n'] for r in okr), visible_frac=sorted(r['visibility']['visible_frac'] for r in okr), pregrasp_visible=sum(1 for r in okr if r['visibility']['pregrasp_visible']),
             pregrasp_dist=sorted(r['visibility']['pregrasp_dist'] for r in okr), pregrasp_in_bad_detect_zone=sum(1 for r in okr if r['visibility']['pregrasp_in_known_bad_detect_zone']),
             unknown_eval_min=min((r['unknown_eval']['clear_m'] for r in okr), default=None), unknown_eval_lt0=sum(1 for r in okr if r['unknown_eval']['clear_m'] < 0),
             along_z_mm=dict(min=min(az) if az else None, max=max(az) if az else None, abs_max=max(abs(v) for v in az) if az else None), root_margin_mm=ROOT_MARGIN_MM,
             scene_known=[n for n, *_ in SCENE], wall_s=round(time.perf_counter() - t0, 1),
             params=dict(V_BASE=R1.V_BASE, W_BASE=R1.W_BASE, A_LIN=A_LIN, ALPHA=ALPHA, V_ARM=R1.V_ARM, A_ARM=R1.A_ARM, wheel_margin=FP.MARGIN, LAT=R1.LAT, stop_dt=0.04, cam=[CAM_NEAR, CAM_FAR], det_bad_below=DET_BAD_BELOW))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(D, f'door_standoff_online_r2_{aa.group}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in S.items() if k not in ('params', 'scene_known')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
