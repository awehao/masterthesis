#!/usr/bin/env python3
"""門把手：視覺估計 → 在線底盤停位與全身接近（離線整合；Codex 20261111_155455 下一包）。不新增擷取、不跑物理、不閉爪、不開門。
輸入沿用 m8doorcap_1 靜態 RGB-D 與凍結偵測器輸出（door_detect_<group>.json），估計沿用 dl2_obs_to_target_v2.estimate（N-obs）＋bar26 型抓取先驗。
與 m8_door_pregrasp_check（11/10）的差別：停位不再是「沿當下車頭前進 0–1.0 m、側移 ±0.1、偏航固定」的有限搜尋，也不回填固定 D1 停位或真值把手；
  而是在**估計把手座標**下生成候選、依「由當下全身狀態出發的轉移時間」排序選取，並核整段全身路徑與停止終點（獨立減速近似、只核終點，不是完整停止掃掠；Codex 160626）。
量的來源（明列）：
  視覺量＝把手中心與桿軸（偵測器＋深度）、門面法向（深度）；
  地圖先驗＝門扇（q＝0）與兩側牆（door_local_v3）；類型先驗＝橫桿／立柱尺寸、設計抓取 T_HG、預抓退讓 0.03 m；
  機器人模型＝URDF 運動學與點雲、Lite 6 安全限位；
  模擬真值＝只在最後 D 段做安全網核對（規劃與選擇都不讀）。
步驟（每個有效中心視點，取第一格有效估計）：
  1 停位候選：估計把手座標 H（x 沿桿、y 指向門內）。底盤位置＝c＋d·n＋s·u（n＝水平門外法向、u＝水平桿軸），d 0.40–0.85（0.05）、s −0.45–0.45（0.075）；
    偏航＝正對門 ＋ δ（δ ∈ ±40°，20° 步）。粗篩：底盤圓盤對門扇／牆 ≥ 0.02 m；肩點（link_base）到預抓 TCP 0.15–0.62 m。
  2 依底盤轉移時間 t_b＝max(|Δp|/V_BASE, |Δψ|/W_BASE) 由小到大求 IK（預抓、抓取；種子＝當下手臂＋固定種子），可行（關節裕度、手臂對環境、底盤、自碰撞）者
    算手臂轉移 t_a＝1.875·max|Δq|/v_arm，T＝max(t_b, t_a)。最多求 N_IK 個候選或得到 K_FEAS 個可行解；依 T（再依關節裕度）排序。
  3 全身路徑（依 T 順序逐一核，第一個通過者為所選停位）：排程 sync＝底盤（xy 直線、偏航最短角）與手臂關節同步 min-jerk，時長 T；
    不通過時試 base_then_arm＝手臂停在觀測構型、底盤先走 t_b，再手臂 t_a。每排程 25 取樣核底盤圓盤、手臂對環境、自碰撞。
    預抓 → 抓取沿接近軸（s 0.03 → 0，5 點）IK 延續與餘裕（手指—橫桿為設計接觸）。
  4 停止終點（獨立減速近似；未取樣停止過程）：沿所選排程每 0.1 s 取樣（位置＋速度＝min-jerk 導數），停止構型＝位置＋速度·延遲（D_STEPS·PHYS）＋v|v|/(2a)
    （底盤本體各軸 A_BASE、手臂 a_arm）；「晚一拍停」再加一個控制拍（0.05 s）續行。核同 3 的淨空。
  5 D 事後真值：所選預抓／抓取構型對真值把手幾何的可行性與 TCP 誤差。
不代表整機動態可達、物理夾持成功或未知箱體淨空（本擷取無掃描，未知箱體不在模型內）。
    python3 evaluation/m8_door_standoff_online.py m8doorcap_1 [--group dev|holdout]
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import m8_door_pregrasp_check as PG  # noqa: E402

DOC, E, K, IDX = PG.DOC, PG.E, PG.K, PG.IDX
LIM_M, ENV_M, BASE_M, SELF_M = DOC.LIM_M, DOC.ENV_M, DOC.BASE_M, DOC.SELF_M
V_BASE, W_BASE = 0.25, 0.6                      # 門前接近的底盤速度上限（m/s、rad/s；W 同航向參考 W_MAX）
V_ARM, A_ARM = 1.0, 19.984                      # 手臂關節速度／加速度上限（W_ARM v_arm、C0.A_ARM）
A_BASE = np.array([1.5, 1.0, 2.0])              # 底盤本體 vx、vy、ωz 減速上限（bottle_s0b_ctrl.A_BASE）
LAT, DT_CTRL = 8 * 0.01, 0.05                   # 在途命令延遲（D_STEPS·PHYS）、控制拍
MJ_PEAK = 1.875
D_LIST = np.round(np.arange(0.40, 0.8501, 0.05), 3)
S_LIST = np.round(np.arange(-0.45, 0.4501, 0.075), 3)
DPSI_LIST = np.radians([-40.0, -20.0, 0.0, 20.0, 40.0])
REACH = (0.15, 0.62)
N_IK, K_FEAS = 90, 6


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def mj(s):
    """min-jerk 位置、速度係數（對 s∈[0,1]）。"""
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5, 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4


MOUNT = K.fk(np.zeros(9), 'link_base')[:3, 3]                           # 底盤座標下的肩點（手臂座）


def base_clear(b):
    return min(float(DOC.box_dist(np.array([[b[0], b[1], c[2]]]), c, R, hs)[0]) - DOC.RB for _, c, R, hs in PG.static_shapes())


def cfg_clear(qf, T_WH):
    """單一全身構型：手臂對環境（含立柱、橫桿；手指／夾爪殼不對橫桿）、底盤圓盤、自碰撞。"""
    env, who, _, (bc, bwho) = PG.env_eval(qf, T_WH)
    sc = DOC.self_clear(qf)
    return dict(env=env, env_who=who, base=bc, base_who=bwho, self=sc, ok=bool(env >= ENV_M and bc >= BASE_M and sc >= SELF_M))


def candidates(T_WH, T_pre, base0):
    c, R = T_WH[:3, 3], T_WH[:3, :3]
    n = -R[:2, 1] / max(np.linalg.norm(R[:2, 1]), 1e-9)                # 水平門外法向
    u = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9)                 # 水平桿軸
    psi_face = math.atan2(-n[1], -n[0])                                 # 正對門
    out = []
    for d in D_LIST:
        for s in S_LIST:
            p = c[:2] + d * n + s * u
            for dp in DPSI_LIST:
                b = np.array([p[0], p[1], wrap(psi_face + dp)])
                if base_clear(b) < BASE_M:
                    continue
                cs, sn = math.cos(b[2]), math.sin(b[2])
                sh = np.array([b[0] + cs * MOUNT[0] - sn * MOUNT[1], b[1] + sn * MOUNT[0] + cs * MOUNT[1], MOUNT[2]])
                r = float(np.linalg.norm(T_pre[:3, 3] - sh))
                if not (REACH[0] <= r <= REACH[1]):
                    continue
                tb = max(float(np.linalg.norm(b[:2] - base0[:2])) / V_BASE, abs(wrap(b[2] - base0[2])) / W_BASE)
                out.append(dict(b=b, d=float(d), s=float(s), dpsi_deg=round(math.degrees(dp), 1), t_base=tb))
    return sorted(out, key=lambda z: z['t_base'])


def schedule(kind, base0, q_obs, b1, q_pre, tb, ta):
    """回 (T, f(t)→(qf, qf_dot))。"""
    dpsi = wrap(b1[2] - base0[2])
    db = np.r_[b1[:2] - base0[:2], dpsi]
    dq = q_pre - q_obs
    if kind == 'sync':
        T = max(tb, ta, 1e-3)

        def f(t):
            p, v = mj(t / T)
            qf = np.zeros(9)
            qf[:3] = base0 + p * db
            qf[IDX] = q_obs + p * dq
            qd = np.zeros(9)
            qd[:3] = v / T * db
            qd[IDX] = v / T * dq
            return qf, qd
        return T, f
    T = tb + ta

    def f(t):
        qf, qd = np.zeros(9), np.zeros(9)
        if t <= tb:
            p, v = mj(t / max(tb, 1e-3))
            qf[:3], qd[:3] = base0 + p * db, v / max(tb, 1e-3) * db
            qf[IDX] = q_obs
        else:
            p, v = mj((t - tb) / max(ta, 1e-3))
            qf[:3] = base0 + db
            qf[IDX], qd[IDX] = q_obs + p * dq, v / max(ta, 1e-3) * dq
        return qf, qd
    return T, f


def stop_cfg(qf, qd, extra):
    """停止構型：先續行在途延遲（＋晚一拍則再加一控制拍），再以各軸最大減速停下（底盤在本體座標）。"""
    tl = LAT + (DT_CTRL if extra else 0.0)
    out = qf.copy()
    psi = qf[2]
    cs, sn = math.cos(psi), math.sin(psi)
    vb = np.array([cs * qd[0] + sn * qd[1], -sn * qd[0] + cs * qd[1], qd[2]])   # 世界 → 本體
    db = vb * tl + vb * np.abs(vb) / (2.0 * A_BASE)
    out[0] += cs * db[0] - sn * db[1]
    out[1] += sn * db[0] + cs * db[1]
    out[2] += db[2]
    qa = qd[IDX]
    out[IDX] = qf[IDX] + qa * tl + qa * np.abs(qa) / (2.0 * A_ARM)
    return out


def path_check(kind, base0, q_obs, b1, q_pre, tb, ta, T_WH):
    T, f = schedule(kind, base0, q_obs, b1, q_pre, tb, ta)
    worst = dict(env=np.inf, base=np.inf, self=np.inf)
    for t in np.linspace(0.0, T, 25):
        qf, _ = f(t)
        cc = cfg_clear(qf, T_WH)
        for k_ in worst:
            worst[k_] = min(worst[k_], cc[k_])
        if not cc['ok']:
            return dict(ok=False, kind=kind, T=round(T, 3), t_fail=round(float(t), 3), at=cc)
    stop = dict(env=np.inf, base=np.inf, self=np.inf, n=0)
    for t in np.arange(0.0, T + 1e-9, 0.1):
        qf, qd = f(t)
        for extra in (0, 1):
            cc = cfg_clear(stop_cfg(qf, qd, extra), T_WH)
            stop['n'] += 1
            for k_ in ('env', 'base', 'self'):
                stop[k_] = min(stop[k_], cc[k_])
            if not cc['ok']:
                return dict(ok=False, kind=kind, T=round(T, 3), stop_fail=dict(t=round(float(t), 3), extra=extra, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()}))
    return dict(ok=True, kind=kind, T=round(T, 3), path_min={k: round(v, 4) for k, v in worst.items()}, stop_min={k: (round(v, 4) if isinstance(v, float) else v) for k, v in stop.items()})


def plan_viewpoint(T_WH, T_pre, T_gr, T_WE_at, base0, q_obs):
    cands = candidates(T_WH, T_pre, base0)
    feas, n_ik = [], 0
    for cd in cands:
        if n_ik >= N_IK or len(feas) >= K_FEAS:
            break
        n_ik += 1
        sol, info = PG.plan_at(cd['b'], q_obs, T_WH, T_pre, T_gr)
        if sol is None:
            continue
        q_pre, q_gr = sol
        if DOC.self_clear(q_pre) < SELF_M or DOC.self_clear(q_gr) < SELF_M:
            continue
        ta = MJ_PEAK * float(np.max(np.abs(q_pre[IDX] - q_obs))) / V_ARM
        feas.append(dict(cd, q_pre=q_pre, q_gr=q_gr, t_arm=ta, T=max(cd['t_base'], ta), margin=min(info['pre']['joint_margin'], info['grasp']['joint_margin'])))
    feas.sort(key=lambda z: (round(z['T'], 2), -z['margin']))
    tried = []
    for fz in feas:
        for kind in ('sync', 'base_then_arm'):
            pc = path_check(kind, base0, q_obs, fz['b'], fz['q_pre'][IDX], fz['t_base'], fz['t_arm'], T_WH)
            tried.append(dict(d=fz['d'], s=fz['s'], dpsi_deg=fz['dpsi_deg'], kind=kind, ok=pc['ok'], T=pc['T']))
            if pc['ok']:
                ap = PG.sweeps(fz['b'], q_obs, fz['q_pre'], T_WH, T_WE_at)
                return dict(ok=True, chosen=dict(base=fz['b'].round(4).tolist(), d=fz['d'], s=fz['s'], dpsi_deg=fz['dpsi_deg'], t_base=round(fz['t_base'], 3),
                                                 t_arm=round(fz['t_arm'], 3), margin=round(fz['margin'], 4)), path=pc, approach=dict(ok=ap['approach_ok'], env=ap['approach_env']),
                            n_candidates=len(cands), n_ik=n_ik, n_feasible=len(feas), tried=tried,
                            _q=dict(base=fz['b'].tolist(), q_pre=fz['q_pre'].tolist(), q_gr=fz['q_gr'].tolist()))
    return dict(ok=False, why='no_candidate' if not cands else ('no_ik_feasible' if not feas else 'no_path'), n_candidates=len(cands), n_ik=n_ik, n_feasible=len(feas), tried=tried)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--group', choices=('dev', 'holdout'), default='dev')
    ap.add_argument('--only', default='')
    aa = ap.parse_args()
    D = os.path.join(HERE, 'runs', aa.run, 'wrist_door')
    det = json.load(open(os.path.join(D, f'door_detect_{aa.group}.json')))
    meta = json.load(open(os.path.join(D, 'meta.json')))
    reg = json.load(open(os.path.join(WS, meta['registration'])))
    VP = {v['id']: v for v in reg['viewpoints']}
    FR = {f['stem']: f for f in meta['frames']}
    Kint = np.array(meta['intrinsics_readback']['K'])
    Rg, T_HG, a_H = PG.baseline_reference()
    prevB = {r['viewpoint']: r for r in json.load(open(os.path.join(D, f'door_pregrasp_{aa.group}.json')))['rows']}
    rows, t0 = [], time.perf_counter()
    for v in det['per_viewpoint']:
        if aa.only and v['viewpoint'] not in aa.only.split(','):
            continue
        rr = [r for r in det['rows'] if r['viewpoint'] == v['viewpoint'] and r['cls'] == 'center_valid']
        if v['n_center_valid'] < 2 or not rr:
            continue
        r = rr[0]
        f = FR[r['stem']]
        dep = np.load(os.path.join(D, 'frames', f"{r['stem']}_depth_m.npy"))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        obs = dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=Kint, T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time'])
        est = E.estimate(obs, 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
        row = dict(viewpoint=v['viewpoint'], stem=r['stem'], est_ok=est['ok'])
        if not est['ok']:
            rows.append(row)
            continue
        T_WH = np.asarray(est['T_WH'], float)

        def T_WE_at(s, T_WH=T_WH):
            return np.asarray(E.OT.object_target(T_WH, T_HG, a_H, float(s))['T_WE'], float)
        vp = VP[v['viewpoint']]
        base0, q_obs = np.array(vp['base'], float), np.array(vp['q_arm'], float)
        w0 = time.perf_counter()
        res = plan_viewpoint(T_WH, T_WE_at(PG.S_PRE), T_WE_at(0.0), T_WE_at, base0, q_obs)
        res['wall_s'] = round(time.perf_counter() - w0, 2)
        pb = prevB.get(v['viewpoint'], {})
        res['prev_limited_search'] = dict(A=pb.get('A_current_base', {}).get('ok'), B=pb.get('B_moved_base', {}).get('ok'), B_disp=pb.get('B_moved_base', {}).get('displacement_m'),
                                          C_joint_interp=pb.get('C_sweeps', {}).get('joint_interp_ok'))
        row.update(res)
        row['_T_WH_est'] = T_WH.tolist()
        rows.append(row)
        ch = row.get('chosen') or {}
        print(f"{v['viewpoint']}: ok {row['ok']} {row.get('why', '')} 候選 {row['n_candidates']} IK {row['n_ik']} 可行 {row['n_feasible']} "
              f"選 d {ch.get('d')} s {ch.get('s')} δψ {ch.get('dpsi_deg')} 排程 {row.get('path', {}).get('kind')} T {row.get('path', {}).get('T')} "
              f"接近 {row.get('approach', {}).get('ok')}｜舊 B {row['prev_limited_search']['B']} C {row['prev_limited_search']['C_joint_interp']}  {row['wall_s']} s", flush=True)
    truth = json.load(open(os.path.join(D, 'truth.json')))                # D 事後真值（安全網）：規劃與選擇完成後才讀
    for row in rows:
        if '_q' not in row:
            continue
        Tt = np.array(truth['per_viewpoint'][row['viewpoint']]['M_world_handle_prim'])
        qp, qg = np.array(row['_q']['q_pre']), np.array(row['_q']['q_gr'])
        okp, ip = PG.feasible(qp, Tt)
        okg, ig = PG.feasible(qg, Tt)
        T_tool = K.fk(qg, 'link_tcp')
        T_true = np.asarray(E.OT.object_target(Tt, T_HG, a_H, 0.0)['T_WE'], float)
        e = T_tool[:3, 3] - T_true[:3, 3]
        z = T_true[:3, 2]
        row['D_truth'] = dict(pre_ok=okp, grasp_ok=okg, tcp_err_mm=round(float(np.linalg.norm(e)) * 1e3, 2), tcp_err_along_tool_z_mm=round(float(e @ z) * 1e3, 2),
                              pre=ip, grasp=ig)
    okr = [r for r in rows if r.get('ok')]
    S = dict(run=aa.run, group=aa.group, n_valid_center=len(rows), est_ok=sum(1 for r in rows if r['est_ok']), standoff_and_path_ok=len(okr),
             approach_ok=sum(1 for r in okr if r['approach']['ok']), sync=sum(1 for r in okr if r['path']['kind'] == 'sync'),
             fail_why={w: sum(1 for r in rows if r.get('est_ok') and not r.get('ok') and r.get('why') == w) for w in ('no_candidate', 'no_ik_feasible', 'no_path')},
             T_s=sorted(r['path']['T'] for r in okr), displacement_m=sorted(round(float(np.linalg.norm(np.array(r['chosen']['base'][:2]) - np.array(VP[r['viewpoint']]['base'][:2]))), 3) for r in okr),
             dpsi_deg=sorted(r['chosen']['dpsi_deg'] for r in okr), path_env_min=min((r['path']['path_min']['env'] for r in okr), default=None),
             stop_env_min=min((r['path']['stop_min']['env'] for r in okr), default=None), stop_base_min=min((r['path']['stop_min']['base'] for r in okr), default=None),
             stop_self_min=min((r['path']['stop_min']['self'] for r in okr), default=None),
             prev_B_ok=sum(1 for r in rows if r.get('prev_limited_search', {}).get('B')), prev_B_and_C=sum(1 for r in rows if r.get('prev_limited_search', {}).get('B') and r.get('prev_limited_search', {}).get('C_joint_interp')),
             D_truth_grasp_ok=sum(1 for r in rows if r.get('D_truth', {}).get('grasp_ok')), D_truth_pre_ok=sum(1 for r in rows if r.get('D_truth', {}).get('pre_ok')),
             D_tcp_err_mm=[r['D_truth']['tcp_err_mm'] for r in rows if 'D_truth' in r], D_tcp_err_along_z_mm=[r['D_truth']['tcp_err_along_tool_z_mm'] for r in rows if 'D_truth' in r],
             wall_s=round(time.perf_counter() - t0, 1),
             params=dict(V_BASE=V_BASE, W_BASE=W_BASE, V_ARM=V_ARM, A_ARM=A_ARM, A_BASE=A_BASE.tolist(), LAT=LAT, d=[float(D_LIST[0]), float(D_LIST[-1])], s=[float(S_LIST[0]), float(S_LIST[-1])],
                         dpsi_deg=[-40, 40], reach=REACH, N_IK=N_IK, K_FEAS=K_FEAS, thresholds=dict(joint=LIM_M, env=ENV_M, base=BASE_M, self=SELF_M)),
             sources='視覺：把手中心／桿軸／門面法向；地圖先驗：門扇（q＝0）與牆；類型先驗：橫桿／立柱尺寸、設計抓取 T_HG、預抓退讓；真值只在 D 段安全網',
             note='離線幾何；未知箱體不在模型內（本擷取無掃描）；不代表動態可達或物理夾持成功')
    json.dump(dict(summary=S, rows=rows), open(os.path.join(D, f'door_standoff_online_{aa.group}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in S.items() if k not in ('params',)}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
