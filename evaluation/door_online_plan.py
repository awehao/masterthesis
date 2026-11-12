#!/usr/bin/env python3
"""門把手：由**本趟遠距估計**在線算停位與到位重瞄構型（Codex 20261112_133045）。不讀登錄的停位／視點清單、不讀真值。
輸入：遠距把手估計 T_WH、當下底盤位姿與手臂關節、相機掛載與內參、目標函式。已知先驗只有：靜態地圖幾何（牆、門扇、家具）、把手類型尺寸、機器人模型。
流程：
  1 停位候選（重新生成）：以估計把手為中心的格點——離門距離 D_LIST×沿桿偏移 S_LIST×偏航 DPSI_LIST（R1.candidates；過濾底盤淨空與肩到預抓距離），
    依底盤移動時間排序；逐一做預抓／抓取 IK（上限 N_IK 次、收 K_FEAS 個可行即止——**截斷搜尋，不是全域最優**）。這一步是初篩（既有 r2 模型）。
  2 到位重瞄候選（重新生成）：每個可行停位上，相機目標＝把手中心外 DCAMS×ELEVS×AZIS×影像翻轉（R3.cam_targets，24 個）；
    IK（種子＝該停位的預抓構型）→ 關節餘裕 → 全碰撞（**張開手指模型** DA.CLEAR）→ 幾何可見（把手兩端、中心、深度探測區入鏡、距離、自遮擋）。
  3 最終放行（最新接受檢查，張開手指模型）：依估計總時間排序，逐一核
      a 起點→重瞄構型 整段：速度／輪級界、路徑取樣、沿途「現在停／晚一拍停」停止取樣；
      b 重瞄構型→預抓 整段（底盤固定）：同上；
    第一個全過者即採用（最多核 MAX_VERIFY 個）。執行時控制器仍逐拍做真管線停止掃掠與候選命令核對；近距影格到位後仍須重拍重算。
  無有效估計／無候選／全部不過 ⇒ 回 ok=False 與原因；**不退回任何固定停位**。
選擇規則只用可達、碰撞、時間；不看真值誤差。"""
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_approach_jnom_ctrl as DA                               # noqa: E402

R4 = DA.R4
R3, R2, R1, PG, DOC, K, IDX = DA.R3, DA.R2, DA.R1, DA.PG, DA.DOC, DA.K, DA.IDX
MAX_STANDOFF, MAX_VERIFY = 3, 6                                     # 最多展開幾個可行停位、最多做幾次整段核對（截斷條件，事前固定）
KEYS = ('env', 'base', 'self', 'bar_tool')


def seg_check_open(base0, q0, b1, q1, T_WH, tag):
    """R4.seg_check 的同規則版本，碰撞改用張開手指模型（DA.CLEAR）。"""
    T, _ = R2.bounded_T(base0, q0, b1, q1)
    _, f = R1.schedule('sync', base0, q0, b1, q1, T, T)
    vb = R2.verify_bounds(f, T)
    if not vb['ok']:
        return dict(ok=False, why=tag + ':bounds', T=round(T, 3))
    mn = {k: np.inf for k in KEYS}
    for t in np.linspace(0.0, T, 21):
        cc = DA.CLEAR(f(t)[0], T_WH)
        for k_ in KEYS:
            mn[k_] = min(mn[k_], cc[k_])
        if not cc['ok']:
            return dict(ok=False, why=tag + ':path', T=round(T, 3), at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()})
    sm, n = {k: np.inf for k in KEYS}, 0
    for t in np.arange(0.0, T + 1e-9, 0.2):
        qf, qd = f(t)
        for extra in (0, 1):
            for qs in R3.stop_traj(qf, qd, extra):
                cc = DA.CLEAR(qs, T_WH)
                n += 1
                for k_ in KEYS:
                    sm[k_] = min(sm[k_], cc[k_])
                if not cc['ok']:
                    return dict(ok=False, why=tag + ':stop_sweep', T=round(T, 3), at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()})
    return dict(ok=True, T=round(T, 3), path_min={k: round(v, 4) for k, v in mn.items()}, stop_min={k: round(v, 4) for k, v in sm.items()}, stop_n=n)


def plan(T_WH, base_now, q_now, T_mount, Kint, T_at):
    """回 dict(ok, b1, q_look, …)。T_WH 無效 ⇒ ok=False（no_valid_far_estimate）。"""
    w0 = time.perf_counter()
    out = dict(ok=False, search=dict(D_LIST=R1.D_LIST.tolist(), S_LIST=R1.S_LIST.tolist(), DPSI_deg=np.degrees(R1.DPSI_LIST).round(1).tolist(), N_IK=R1.N_IK, K_FEAS=R1.K_FEAS, MAX_STANDOFF=MAX_STANDOFF, MAX_VERIFY=MAX_VERIFY,
                                     DCAMS=list(R3.DCAMS), ELEVS=list(R3.ELEVS), AZIS=list(R3.AZIS), note='截斷搜尋：依時間排序取第一個通過者，不是全域最優'))
    if T_WH is None or not DA.valid_pose(T_WH):
        return dict(out, why='no_valid_far_estimate', wall_s=round(time.perf_counter() - w0, 3))
    T_WH = np.asarray(T_WH, float)
    base_now, q_now = np.asarray(base_now, float), np.asarray(q_now, float)
    T_pre, T_gr = T_at(T_WH, PG.S_PRE), T_at(T_WH, 0.0)
    # 1 停位初篩
    cands = R1.candidates(T_WH, T_pre, base_now)
    feas, n_ik, rej = [], 0, {}
    for cd in cands:
        if n_ik >= R1.N_IK or len(feas) >= R1.K_FEAS:
            break
        n_ik += 1
        sol, info = PG.plan_at(cd['b'], q_now, T_WH, T_pre, T_gr)
        if sol is None:
            rej['ik_or_clear'] = rej.get('ik_or_clear', 0) + 1
            continue
        q_pre = sol[0]
        cc = DA.CLEAR(q_pre, T_WH)
        if not cc['ok']:
            rej['pregrasp_open_finger'] = rej.get('pregrasp_open_finger', 0) + 1
            continue
        Tb, _ = R2.bounded_T(base_now, q_now, cd['b'], q_pre[IDX])
        feas.append(dict(cd, q_pre=q_pre, T=Tb))
    feas.sort(key=lambda z: round(z['T'], 2))
    w1 = time.perf_counter()
    out.update(n_standoff_candidates=len(cands), n_standoff_ik=n_ik, n_standoff_feasible=len(feas), standoff_rejected=rej)
    if not feas:
        return dict(out, why='no_candidate' if not cands else 'no_standoff_feasible', wall_s=round(time.perf_counter() - w0, 3))
    # 2 到位重瞄候選
    qf0 = np.r_[base_now, q_now]
    T_cam_tcp = np.linalg.inv(np.linalg.inv(K.fk(qf0, 'link_tcp')) @ K.fk(qf0, 'link_eef') @ T_mount)
    looks, lrej, n_lik = [], {}, 0
    for fz in feas[:MAX_STANDOFF]:
        for ct in R3.cam_targets(T_WH):
            n_lik += 1
            ql = PG.ik(fz['b'], fz['q_pre'][IDX], ct['T'] @ T_cam_tcp, extra_seeds=True)
            if ql is None:
                lrej['ik'] = lrej.get('ik', 0) + 1
                continue
            if DOC.jmargin(ql[IDX])[0] < R1.LIM_M:
                lrej['joint'] = lrej.get('joint', 0) + 1
                continue
            if not DA.CLEAR(ql, T_WH)['ok']:
                lrej['collision'] = lrej.get('collision', 0) + 1
                continue
            sv = R3.see(K.fk(ql, 'link_eef') @ T_mount, Kint, T_WH, ql, 'link_eef')
            if not sv['ok']:
                key = 'vis:' + sv['why'].split(':')[1]
                lrej[key] = lrej.get(key, 0) + 1
                continue
            T1, _ = R2.bounded_T(base_now, q_now, fz['b'], ql[IDX])
            T2, _ = R2.bounded_T(fz['b'], ql[IDX], fz['b'], fz['q_pre'][IDX])
            looks.append(dict(b=fz['b'], st=dict(d=fz['d'], s=fz['s'], dpsi_deg=fz['dpsi_deg']), ql=ql, q_pre=fz['q_pre'], cam=dict(d=ct['d'], el=ct['el'], az=ct['az'], flip=bool(ct['flip'])), cam_dist=sv['pts']['center']['dist'], T1=T1, T2=T2, T_est=T1 + R3.T_LOOK + T2))
    looks.sort(key=lambda z: round(z['T_est'], 2))
    w2 = time.perf_counter()
    out.update(n_look_ik=n_lik, n_look_ok=len(looks), look_rejected=lrej)
    if not looks:
        return dict(out, why='no_look_candidate', wall_s=round(time.perf_counter() - w0, 3))
    # 3 最終放行
    tried = []
    for cd in looks[:MAX_VERIFY]:
        tag = dict(standoff=cd['st'], cam=cd['cam'], T_est=round(cd['T_est'], 3))
        s1 = seg_check_open(base_now, q_now, cd['b'], cd['ql'][IDX], T_WH, 'to_look')
        if not s1['ok']:
            tried.append(dict(tag, ok=False, why=s1['why']))
            continue
        s2 = seg_check_open(cd['b'], cd['ql'][IDX], cd['b'], cd['q_pre'][IDX], T_WH, 'to_pre')
        if not s2['ok']:
            tried.append(dict(tag, ok=False, why=s2['why']))
            continue
        tried.append(dict(tag, ok=True))
        w3 = time.perf_counter()
        return dict(out, ok=True, b1=cd['b'].copy(), q_look=cd['ql'][IDX].copy(), q_pre_far=cd['q_pre'][IDX].copy(), standoff=cd['st'], cam=cd['cam'], cam_dist=cd['cam_dist'],
                    plan_motion_s=dict(to_look=s1['T'], look=R3.T_LOOK, to_pre=s2['T'], total=round(s1['T'] + R3.T_LOOK + s2['T'], 3)), seg_to_look=s1, seg_to_pre=s2, tried=tried,
                    wall_s=round(w3 - w0, 3), wall_split_s=dict(standoff=round(w1 - w0, 3), look=round(w2 - w1, 3), verify=round(w3 - w2, 3)))
    return dict(out, why='no_verified_candidate', tried=tried, wall_s=round(time.perf_counter() - w0, 3))
