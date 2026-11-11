#!/usr/bin/env python3
"""門把手途中重瞄 r4：接住宅場景已知幾何＋補 r3 待修三項＋選擇準則與誤差預算列表（離線；不跑模擬、不閉爪、不開門）。
環境：M8_SCENE_REG=20261112_m8_house_scene.json（本檔預設）⇒ 已知幾何＝住宅場景的外牆、內牆延伸、全部家具（門洞牆與門扇同門資產）。未知物不進模型。
r3 待修：
  1 停止取樣保留全部取樣與終點（r3 隔一個取一個）；步長比較分量報告：底盤位置（m）、偏航（rad）、關節（rad），另算工具（TCP）位置差（m）。
  2 重瞄後的行程改沿 FK 軌跡積分（兩段路徑＋插入段），不是端點直線距離。
  3 插入段（預抓→抓取，指定 1.0 s min-jerk）回核 TCP 速度與關節速度／加速度。
選擇：對每個視點列出全部可行的重瞄方案，報「最快」與「重瞄後行程最短」兩種選法（Codex：先滿足接觸誤差預算再選較快者；預算項目尚缺來源 ⇒ 只列表，不放行）。
接觸誤差預算（逐項列來源；缺的標缺）：把手估計沿接近軸誤差——重瞄靜態擷取的樣本值（開發結果，不是上界）；底盤定位、相機外參、手臂追蹤、姿態角差造成的指尖偏移——缺。
    python3 evaluation/m8_door_reaim_online_r4.py m8doorcap_1 --group dev|holdout"""
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_door_reaim_online_r3 as R3                               # noqa: E402
import bottle_v1_cam_visibility as VZ                             # noqa: E402

R2, R1, PG, DOC, E, K, IDX = R3.R2, R3.R1, R3.PG, R3.DOC, R3.E, R3.K, R3.IDX
T_INS = 1.0
print('#EARLY r4 scene', R2.REG['name'], 'known', len(R2.SCENE), flush=True)


def stop_conv():
    qf, qd = np.zeros(9), np.zeros(9)
    qd[0], qd[1], qd[2] = 0.2, 0.1, 0.3
    qd[IDX] = 0.6
    out = {}
    for dt in (0.04, 0.02):
        a_, b_ = R3.stop_traj(qf, qd, 1, dt)[-1], R3.stop_traj(qf, qd, 1, 0.005)[-1]
        out[str(dt)] = dict(base_pos_m=round(float(np.linalg.norm(a_[:2] - b_[:2])), 5), yaw_rad=round(float(abs(a_[2] - b_[2])), 5), joint_rad_max=round(float(np.abs(a_[IDX] - b_[IDX]).max()), 5),
                            tcp_m=round(float(np.linalg.norm(K.fk(a_, 'link_tcp')[:3, 3] - K.fk(b_, 'link_tcp')[:3, 3])), 5))
    return out


def seg_check(base0, q0, b1, q1, T_WH, tag):
    T, _ = R2.bounded_T(base0, q0, b1, q1)
    _, f = R1.schedule('sync', base0, q0, b1, q1, T, T)
    vb = R2.verify_bounds(f, T)
    if not vb['ok']:
        return dict(ok=False, why=tag + ':bounds', T=round(T, 3)), None
    mn = dict(env=np.inf, base=np.inf, self=np.inf, bar_tool=np.inf)
    tcp, trav = None, 0.0
    for t in np.linspace(0.0, T, 41):
        qf = f(t)[0]
        p = K.fk(qf, 'link_tcp')[:3, 3]
        trav += 0.0 if tcp is None else float(np.linalg.norm(p - tcp))
        tcp = p
    for t in np.linspace(0.0, T, 21):
        cc = R3.cfg_clear_full(f(t)[0], T_WH)
        for k_ in mn:
            mn[k_] = min(mn[k_], cc[k_])
        if not cc['ok']:
            return dict(ok=False, why=tag + ':path', T=round(T, 3)), None
    sm, n = dict(env=np.inf, base=np.inf, self=np.inf, bar_tool=np.inf), 0
    for t in np.arange(0.0, T + 1e-9, 0.2):
        qf, qd = f(t)
        for extra in (0, 1):
            for qs in R3.stop_traj(qf, qd, extra):                # 全部取樣含終點
                cc = R3.cfg_clear_full(qs, T_WH)
                n += 1
                for k_ in sm:
                    sm[k_] = min(sm[k_], cc[k_])
                if not cc['ok']:
                    return dict(ok=False, why=tag + ':stop_sweep', T=round(T, 3), at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()}), None
    return dict(ok=True, T=round(T, 3), bounds=vb, tcp_travel_m=round(trav, 4), path_min={k: round(v, 4) for k, v in mn.items()}, stop_min={k: round(v, 4) for k, v in sm.items()}, stop_n=n), f


def insert_check(b1, q_pre, T_WH, T_WE_at):
    """預抓→抓取：1.0 s min-jerk 沿接近軸；逐取樣 IK；回核 TCP 速度、關節速度／加速度、指定接觸段碰撞（只豁免手指對橫桿）。"""
    n = 21
    qs, prev, ok, shell = [], q_pre, True, np.inf
    for k in range(n):
        p, _ = R1.mj(k / (n - 1))
        s = PG.S_PRE * (1.0 - p)
        q = PG.ik(b1, prev[IDX], T_WE_at(s), extra_seeds=False)
        if q is None:
            return dict(ok=False, why='insert_ik')
        cc = R3.cfg_clear_full(q, T_WH, contact=True)
        shell = min(shell, cc['bar_tool'])
        ok = ok and cc['ok']
        qs.append(q)
        prev = q
    Q = np.array(qs)[:, IDX]
    dt = T_INS / (n - 1)
    qd = np.abs(np.diff(Q, axis=0)) / dt
    qdd = np.abs(np.diff(Q, 2, axis=0)) / dt ** 2
    P = np.array([K.fk(q, 'link_tcp')[:3, 3] for q in qs])
    v = np.linalg.norm(np.diff(P, axis=0), axis=1) / dt
    return dict(ok=bool(ok and qd.max() <= R1.V_ARM and qdd.max() <= R1.A_ARM), T=T_INS, tcp_travel_m=round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 4), v_tcp_max=round(float(v.max()), 4),
                qd_max=round(float(qd.max()), 4), qdd_max=round(float(qdd.max()), 3), shell_bar_min=round(float(shell), 4))


def plan4(T_WH, T_pre, T_gr, T_WE_at, base0, q_obs, T_mount, Kint):
    t0 = time.perf_counter()
    sel = R2.plan_viewpoint2(T_WH, T_pre, T_gr, T_WE_at, base0, q_obs, T_mount, Kint)
    out = dict(standoff_ok=sel['ok'], t_standoff_s=round(time.perf_counter() - t0, 2), standoff_rejected=sel.get('rejected'))
    if not sel['ok']:
        return dict(out, ok=False, why='standoff:' + str(sel.get('why')))
    b1, q_pre = np.array(sel['chosen']['base']), np.array(sel['_q']['q_pre'])
    out['standoff'] = dict(base=b1.round(4).tolist(), T_direct=sel['path']['T'])
    qf0 = np.zeros(9)
    qf0[:3], qf0[IDX] = base0, q_obs
    T_cam_tcp = np.linalg.inv(np.linalg.inv(K.fk(qf0, 'link_tcp')) @ K.fk(qf0, 'link_eef') @ T_mount)
    ins = insert_check(b1, q_pre, T_WH, T_WE_at)
    out['insert'] = ins
    if not ins['ok']:
        return dict(out, ok=False, why='insert:' + str(ins.get('why', 'bounds_or_collision')))
    db = np.r_[b1[:2] - base0[:2], R1.wrap(b1[2] - base0[2])]
    rej, cands, n_ik = {}, [], 0
    for lam in R3.LAMS:
        bl = base0 + lam * db
        if R1.base_clear(bl) < R1.BASE_M:
            rej['base'] = rej.get('base', 0) + 1
            continue
        for ct in R3.cam_targets(T_WH):
            n_ik += 1
            ql = PG.ik(bl, q_obs if lam < 0.9 else q_pre[IDX], ct['T'] @ T_cam_tcp, extra_seeds=True)
            if ql is None:
                rej['ik'] = rej.get('ik', 0) + 1
                continue
            if DOC.jmargin(ql[IDX])[0] < R1.LIM_M:
                rej['joint'] = rej.get('joint', 0) + 1
                continue
            if not R3.cfg_clear_full(ql, T_WH)['ok']:
                rej['collision'] = rej.get('collision', 0) + 1
                continue
            sv = R3.see(K.fk(ql, 'link_eef') @ T_mount, Kint, T_WH, ql, 'link_eef')
            if not sv['ok']:
                rej['vis'] = rej.get('vis', 0) + 1
                continue
            T1, _ = R2.bounded_T(base0, q_obs, bl, ql[IDX])
            T2, _ = R2.bounded_T(bl, ql[IDX], b1, q_pre[IDX])
            est_blind = float(np.linalg.norm(K.fk(q_pre, 'link_tcp')[:3, 3] - K.fk(ql, 'link_tcp')[:3, 3]))
            cands.append(dict(lam=lam, bl=bl, ql=ql, cam=ct, T_est=T1 + R3.T_LOOK + T2, blind_est=est_blind))
    out.update(n_look_ik=n_ik, n_look_geom_ok=len(cands), look_rejected=rej)

    def verify(cd):
        s1, _ = seg_check(base0, q_obs, cd['bl'], cd['ql'][IDX], T_WH, 'to_look')
        if not s1['ok']:
            return dict(ok=False, why=s1['why'])
        s2, _ = seg_check(cd['bl'], cd['ql'][IDX], b1, q_pre[IDX], T_WH, 'to_pre')
        if not s2['ok']:
            return dict(ok=False, why=s2['why'])
        return dict(ok=True, lam=cd['lam'], cam_dist=cd['cam']['d'], elev_deg=cd['cam']['el'], azi_deg=cd['cam']['az'], T_total=round(s1['T'] + R3.T_LOOK + s2['T'] + T_INS, 3),
                    blind_tcp_travel_m=round(s2['tcp_travel_m'] + ins['tcp_travel_m'], 4), blind_time_s=round(s2['T'] + T_INS, 3), base_move_after_look_m=round(float(np.linalg.norm(b1[:2] - cd['bl'][:2])), 3),
                    stop_min={k: min(s1['stop_min'][k], s2['stop_min'][k]) for k in s1['stop_min']}, stop_n=s1['stop_n'] + s2['stop_n'], bounds2=s2['bounds'])
    picks = {}
    for name, key in (('fastest', lambda z: z['T_est']), ('shortest_blind', lambda z: (round(z['blind_est'], 2), z['T_est']))):
        fails = []
        for cd in sorted(cands, key=key)[:4]:
            r = verify(cd)
            if r['ok']:
                picks[name] = r
                break
            fails.append(r['why'])
        else:
            picks[name] = dict(ok=False, why='no_verified_candidate', fails=fails)
    ok = bool(picks['fastest'].get('ok') and picks['shortest_blind'].get('ok'))
    return dict(out, ok=ok, why=None if ok else 'no_look_path', picks=picks, T_direct_plus_insert=round(sel['path']['T'] + T_INS, 3))


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
    conv = stop_conv()
    print('停止取樣步長比較（對 0.005 s）', json.dumps(conv), flush=True)
    rows, t0 = [], time.perf_counter()
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
        est = E.estimate(dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=Kint, T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time']), 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
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
        T_mount = np.linalg.inv(K.fk(qf0, 'link_eef')) @ np.array(f['cam_T_fk'])
        w0 = time.perf_counter()
        res = plan4(T_WH, T_WE_at(PG.S_PRE), T_WE_at(0.0), T_WE_at, base0, q_obs, T_mount, Kint)
        res['wall_s'] = round(time.perf_counter() - w0, 2)
        row.update(res)
        rows.append(row)
        pk = row.get('picks') or {}
        print(f"{v['viewpoint']}: ok {row['ok']} {row.get('why') or ''} 重瞄幾何可行 {row.get('n_look_geom_ok')} 拒 {row.get('look_rejected')} "
              f"最快 {({k: pk['fastest'].get(k) for k in ('lam', 'cam_dist', 'T_total', 'blind_tcp_travel_m', 'blind_time_s')} if pk.get('fastest', {}).get('ok') else pk.get('fastest'))} "
              f"盲段最短 {({k: pk['shortest_blind'].get(k) for k in ('lam', 'cam_dist', 'T_total', 'blind_tcp_travel_m', 'blind_time_s')} if pk.get('shortest_blind', {}).get('ok') else pk.get('shortest_blind'))} {row['wall_s']} s", flush=True)
    okr = [r for r in rows if r.get('ok')]

    def col(pick, k):
        return sorted(r['picks'][pick][k] for r in okr)
    S = dict(run=aa.run, group=aa.group, scene=R2.REG['name'], scene_known=[n for n, *_ in R2.SCENE], n=len(rows), standoff_ok=sum(1 for r in rows if r.get('standoff_ok')), reaim_ok=len(okr),
             fail_why=sorted({str(r.get('why')) for r in rows if not r.get('ok')}), standoff_rejected_total={k: sum((r.get('standoff_rejected') or {}).get(k, 0) for r in rows) for k in ('ik', 'self_at_goal', 'bounds', 'path', 'stop_sweep')},
             insert=dict(v_tcp_max=max((r['insert']['v_tcp_max'] for r in rows if r.get('insert', {}).get('ok')), default=None), qd_max=max((r['insert']['qd_max'] for r in rows if r.get('insert', {}).get('ok')), default=None),
                         qdd_max=max((r['insert']['qdd_max'] for r in rows if r.get('insert', {}).get('ok')), default=None), shell_bar_min=min((r['insert']['shell_bar_min'] for r in rows if r.get('insert', {}).get('ok')), default=None)),
             stop_conv=conv, wall_s=round(time.perf_counter() - t0, 1), compute_s=sorted(r['wall_s'] for r in rows if 'wall_s' in r))
    for pick in ('fastest', 'shortest_blind'):
        if okr:
            S[pick] = dict(T_total=[col(pick, 'T_total')[0], col(pick, 'T_total')[-1]], blind_tcp_travel_m=[col(pick, 'blind_tcp_travel_m')[0], col(pick, 'blind_tcp_travel_m')[-1]], blind_time_s=[col(pick, 'blind_time_s')[0], col(pick, 'blind_time_s')[-1]],
                           base_move_after_look_m=[col(pick, 'base_move_after_look_m')[0], col(pick, 'base_move_after_look_m')[-1]], lam=sorted(set(col(pick, 'lam'))), cam_dist=sorted(set(col(pick, 'cam_dist'))),
                           stop_min={k: min(r['picks'][pick]['stop_min'][k] for r in okr) for k in ('env', 'base', 'self', 'bar_tool')}, extra_vs_direct=[round(min(r['picks'][pick]['T_total'] - r['T_direct_plus_insert'] for r in okr), 2), round(max(r['picks'][pick]['T_total'] - r['T_direct_plus_insert'] for r in okr), 2)])
    S['error_budget'] = dict(root_margin_mm=2.7, items=[dict(item='把手估計沿接近軸誤差（重瞄視角）', value='重瞄靜態擷取 27 幀：−0.87～+3.79 mm（開發結果樣本值，不是上界；50%／100% 同相機位姿不算獨立）', source='m8doorcap_reaim_1'),
                                                         dict(item='重瞄後底盤移動造成的定位誤差', value=None, source='缺：沒有機器人運動資料'), dict(item='相機外參與關節讀值誤差', value=None, source='缺：模擬中 FK 與算圖一致，實機未量'),
                                                         dict(item='手臂追蹤誤差（盲段）', value=None, source='缺：未經控制器'), dict(item='目標姿態角差造成的指尖偏移', value='姿態角差樣本最大 1.59°', source='m8doorcap_reaim_1；未換算成指尖位移')],
                             admissible=False, why='預算項目缺來源，且樣本中已有一幀 3.79 mm 超過餘裕')
    json.dump(dict(summary=S, rows=rows), open(os.path.join(D, f'door_reaim_online_r4_{aa.group}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in S.items() if k not in ('scene_known', 'compute_s', 'error_budget')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
