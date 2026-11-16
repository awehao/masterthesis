#!/usr/bin/env python3
"""完整停止核對（逐字取自 far_dry/merge_node.py 工作程序的 stop_check 與其自碰撞／區間界工具；原檔不改）。
兩種實際會採用的停法都核：now（本拍起送零）與 next（提交候選 ua 一拍後送零）；底盤＝在途 8 步＋平滑器既有減速上限、手臂＝在途命令走完後設定點不動、
關節一階模型（α）加靜態偏置模型 off_fn；限位、停穩、淨空（區間界＋自碰撞逐對、不過則加密到逐物理步）。淨空在地圖座標核（把手估計以 map←odom 轉入）。
與原處唯一差別：把手估計由 get_T_est()（里程座標）取得，不是工作程序的區域變數。
    sc = make(pool, get_T_est, off_fn, AL, LO, HI)(q, pipe, rel, s, ua, T_mo)"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import door_wbopt as W                                               # noqa: E402
import door_clear_door_fast as _F                                    # noqa: E402
W.clear_door = _F.clear_door_fast
DT, PHYS, NPH = 0.05, 0.01, 5
DEC = (6.25, 6.25, 25.51)                                             # 平滑器既有減速上限（同 merge_node）
SWEEP_S = 0.040                                                       # 掃掠截止（自停止核對起算，牆鐘）；原處固定 0.040，這裡可由呼叫端依 RTF 設定並須揭露


def _T3(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = p[:2]
    return T


def make(pool, get_T_est, off_fn, AL, LO, HI):
    def to_map(q, T_mo):
        """里程座標的機器人構型 → 地圖座標（靜態環境幾何所在的座標）。"""
        c, s_ = math.cos(T_mo[2]), math.sin(T_mo[2])
        return np.r_[c * q[0] - s_ * q[1] + T_mo[0], s_ * q[0] + c * q[1] + T_mo[1], q[2] + T_mo[2], q[3:]]
    import door_clear_fast as FAST
    from scipy.spatial import cKDTree
    PAIRS = FAST._pairs()
    CHAIN = dict(base_link=0, link_base=0, link1=1, link2=2, link3=3, link4=4, link5=5, link6=6, link_eef=6, uflite_gripper_link=6, uflite_finger1=6, uflite_finger2=6)
    SNAMES = sorted(set(FAST.DOC.ARM_LINKS) | {n_ for p_ in PAIRS for n_ in p_} | {f'link{k_}' for k_ in range(1, 7)})
    PAIR_CACHE = {}
    # 逐對自碰撞區間界用的槓桿長上界（對所有構型成立；URDF 固定平移長度的三角不等式）：
    #   r(k, y) ＝ Σ_{m=k}^{iy−1} |o_{m+1} − o_m| ＋ max_p |p_y − o_iy|
    #   o_m＝關節 m 原點（＝link m 座標原點）；相鄰關節原點距離與構型無關；y 的點雲相對 o_iy 的距離：y 與 link iy 之間只有固定變換（手指另加張開量），與構型無關。
    _T0 = _F.fk_all(np.zeros(9), SNAMES)
    _O = {m_: _T0[f'link{m_}'][:3, 3] for m_ in range(1, 7)}
    _SEG = {m_: float(np.linalg.norm(_O[m_ + 1] - _O[m_])) for m_ in range(1, 6)}
    _RY = {}
    for n_ in SNAMES:
        if n_ in FAST.DOC.CL and CHAIN.get(n_, 0) >= 1:
            Pn = (_T0[n_][:3, :3] @ FAST.DOC.CL[n_].T).T + _T0[n_][:3, 3]
            _RY[n_] = float(np.linalg.norm(Pn - _O[CHAIN[n_]], axis=1).max()) + (0.012 if n_ in FAST.DOC.FINGERS else 0.0)   # 手指沿滑動軸最多張開約 9 mm，加 12 mm
    LEVER = {(k_, n_): sum(_SEG[m_] for m_ in range(k_, CHAIN[n_])) + _RY[n_] for n_ in _RY for k_ in range(1, CHAIN[n_] + 1)}
    R_ENV = 1.10                                                      # 機器人任一點到底盤原點距離的保守上界（m）。Codex 20261116 以碰撞點雲＋URDF 固定平移長度的三角不等式核得約 0.9936 m
    R_JNT = 0.75                                                      # 任一手臂關節到其遠端任一點距離的保守上界（m）。同上核得最大約 0.6401 m

    def self_pairs(qf):
        """自碰撞逐對資料：回 (T, 各對下界 lb＝中心距−兩半徑, 中心, 需要時算精確距離的函式)。精確距離＝與 door_clear_fast.clear_full 同一算法（點雲對 KD 樹）。"""
        T = _F.fk_all(np.asarray(qf, float), SNAMES)
        fq = W.FINGER_Q
        for n_ in FAST.DOC.FINGERS:
            fqn = (fq[0] if n_.endswith('1') else fq[1]) if isinstance(fq, (tuple, list)) else fq
            if n_ in T and fqn:
                T[n_] = T[n_].copy()
                T[n_][:3, 3] += (1.0 if n_.endswith('1') else -1.0) * fqn * T[n_][:3, 1]
        Cw = {n_: T[n_][:3, :3] @ FAST._C[n_] + T[n_][:3, 3] for n_ in SNAMES if n_ in FAST._C}
        lb = {(x, y): float(np.linalg.norm(Cw[x] - Cw[y])) - FAST._R[x] - FAST._R[y] for x, y in PAIRS}

        def exact(x, y):
            if x not in FAST._TREE:
                FAST._TREE[x] = cKDTree(FAST.DOC.CL[x])
            Py = ((T[y][:3, :3] @ FAST.DOC.CL[y].T).T + T[y][:3, 3] - T[x][:3, 3]) @ T[x][:3, :3]
            return float(FAST._TREE[x].query(Py, k=1)[0].min())
        return T, lb, Cw, exact

    def stop_check(q, pipe, rel, s, ua, T_mo):
        """提交前停止核對，兩種實際會採用的停法都核：
          now ：本拍起就送零（候選不提交）——底盤＝在途 8 步 → 平滑器以既有減速上限逐拍降到零；手臂＝在途命令走完後設定點不再動，關節以一階模型收斂。
          next：本拍提交候選 ua、下一拍起送零。
        通過條件（全部成立）：
          (1) 限位：設定點與關節全程在限位內；
          (2) 停穩：在途管線排清後，底盤命令為零且關節每步變化 <1e-5 rad 連續 0.2 s（在預測時域內達成）；
          (3) 淨空含區間界：相鄰兩個樣本之間，任一點的位移不超過「路徑長上界」L（逐物理步累加，不是端點差）；
              區間內任一構型離較近端點的位移 ≤ L/2，所以要求 兩端淨空的較小值 − L/2 ≥ 門檻。門檻不改。
              環境／底盤：L＝底盤平移路徑長＋|偏航變化|·R_ENV＋Σ|關節變化|·R_JNT（對所有連桿取同一保守值）。
              自碰撞逐對：L＝Σ(兩連桿之間的關節 k)|關節變化|·r(k, 遠端連桿)；r＝對所有構型成立的槓桿長上界（相鄰關節原點距離之和＋遠端連桿點雲到其關節原點的最大距離），不超過 R_JNT；兩連桿之間沒有運動關節 ⇒ L＝0。
              某區間不過且中間還有物理步 ⇒ 加密到逐物理步再核；逐步仍不過 ⇒ 不通過。
        淨空在地圖座標核（機器人構型與把手估計以當下 map←odom 轉入；靜態環境幾何本來就在地圖座標）。"""
        t0 = time.perf_counter()
        T_map = _T3(T_mo) @ get_T_est()
        thr = W.THR
        trajs = {}
        for mode in ('now', 'next'):
            cmds = [np.asarray(u, float) for u in pipe] + ([np.r_[rel, ua]] * NPH if mode == 'next' else [])
            v = np.asarray(rel, float).copy()
            while float(np.abs(v).max()) > 1e-9:
                v = v - np.sign(v) * np.minimum(np.abs(v), np.asarray(DEC) * DT)
                cmds += [np.r_[v, np.zeros(6)]] * NPH
            n_cmd = len(cmds)
            cmds += [np.zeros(9)] * 100
            b, a, sp = np.asarray(q[:3], float).copy(), np.asarray(q[3:], float).copy(), np.asarray(s, float).copy()
            off_c = off_fn(q)                                          # 停止期間的靜態偏置取當下構型的值（停止行程短，重力項變化小；與追蹤器同一模型）
            Q, lim_ok, still, k_still = [np.r_[b, a]], True, 0, None
            for k, u in enumerate(cmds):
                c, s_ = math.cos(b[2]), math.sin(b[2])
                b = b + PHYS * np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1], u[2]])
                sp = sp + PHYS * u[3:]
                a_new = a + AL * (sp + off_c - a)
                still = still + 1 if (k >= n_cmd and float(np.abs(a_new - a).max()) < 1e-5) else 0
                a = a_new
                lim_ok = lim_ok and bool(np.all(sp >= LO) and np.all(sp <= HI) and np.all(a >= LO) and np.all(a <= HI))
                Q.append(np.r_[b, a])
                if still >= 20:
                    k_still = k + 1
                    break
            Q = np.array(Q)
            dQ = np.abs(np.diff(Q, axis=0))
            dxy = np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1)
            idx, last = [0], 0
            for k in range(1, len(Q)):
                if float(np.linalg.norm(Q[k, :2] - Q[last, :2])) > 0.005 or float(np.abs(Q[k, 3:] - Q[last, 3:]).max()) > 0.01 or k == len(Q) - 1:
                    idx.append(k); last = k
            trajs[mode] = dict(Q=Q, dQ=dQ, dxy=dxy, idx=idx, lim_ok=lim_ok, settled=k_still is not None, settle_s=None if k_still is None else round(k_still * PHYS, 2), pipe_drained_s=round(n_cmd * PHYS, 2))
        cache = {}

        def key(x):
            return np.ascontiguousarray(x, dtype=np.float64).tobytes()

        def evaluate(cfgs, self_only=False):
            """回 None 或失敗原因。環境／底盤淨空走平行掃掠池；自碰撞逐對資料在本程序算。
            self_only：加密點只需自碰撞資料（環境／底盤已在初始取樣的區間界下通過時），全體最小自碰撞距離用同一精確算法在本程序算。"""
            todo = [x for x in cfgs if key(x) not in cache]
            uniq = {}
            for x in todo:
                uniq[key(x)] = x
            todo = list(uniq.values())
            if not todo:
                return None
            if self_only:
                for x in todo:
                    spx = self_pairs(x)
                    cache[key(x)] = dict(env=None, base=None, self_min=-1.0, sp=spx, ex={})   # self_min=-1：不提供全體最小值這個下界（逐對只用中心下界或精確值）
                return None
            h = pool.submit(T_map, W.FINGER_Q, W.PHI0, np.array([np.r_[to_map(x, T_mo), 0.0] for x in todo]))
            sp_ = [self_pairs(x) for x in todo]                        # 與池同時進行
            outp, why, _ = pool.collect(h, SWEEP_S - (h['t_submit'] - t0))   # 掃掠截止固定為 t0＋40 ms，時間耗盡即判未驗證、不補時間（collect 的期限以派工時刻起算，所以扣的是派工時刻；far_door_5 前誤扣成「現在」，實際上限只有約 36 ms）；整拍 45 ms 期限不變
            if outp is None:
                return 'sweep_' + str(why)
            for x, o, spx in zip(todo, outp, sp_):
                cache[key(x)] = dict(env=float(o['env']), base=float(o['base']), self_min=float(o['self']), sp=spx, ex={})
            return None

        def pair_d(c, pr, need, x_cfg):
            """該對的距離：下界已足夠就用下界（保守），否則算精確值。
            下界＝max(中心距−兩半徑, 該構型全體自碰撞最小距離)——後者是同一構型所有配對的精確最小值，對任一對都是有效下界。
            精確值依「兩連桿之間的關節角」快取（相對位形只由這些關節決定；鍵為實際位元組，只有完全相同的角度才共用）。"""
            lb = max(c['sp'][1][pr], c['self_min'])
            if lb >= need:
                return lb
            ix, iy = sorted((CHAIN[pr[0]], CHAIN[pr[1]]))
            kk = (pr, np.ascontiguousarray(x_cfg[3 + ix:3 + iy], dtype=np.float64).tobytes())   # 鍵＝實際角度位元組（不四捨五入：不同構型不共用距離）
            v_ = PAIR_CACHE.get(kk)
            if v_ is None:
                v_ = c['sp'][3](*pr)
                if len(PAIR_CACHE) > 200000:
                    PAIR_CACHE.clear()
                PAIR_CACHE[kk] = v_
            return v_

        def certify(mode, parts=('env', 'self')):
            tr = trajs[mode]
            Q, idx = tr['Q'], tr['idx_self'] if parts == ('self',) else tr['idx']
            worst, bad = dict(env=9.9, base=9.9, self=9.9), []
            for i, j in zip(idx[:-1], idx[1:]):
                ci, cj = cache[key(Q[i])], cache[key(Q[j])]
                tvq = tr['dQ'][i:j, 3:].sum(axis=0)
                ok_ij = True
                if 'env' in parts:
                    L_env = float(tr['dxy'][i:j].sum()) + float(tr['dQ'][i:j, 2].sum()) * R_ENV + float(tvq.sum()) * R_JNT
                    for k_ in ('env', 'base'):
                        m_ = min(ci[k_], cj[k_]) - 0.5 * L_env - thr[k_]
                        worst[k_] = min(worst[k_], m_)
                        ok_ij = ok_ij and m_ >= 0.0
                if 'self' in parts:
                    for pr in PAIRS:
                        x, y = pr
                        ix, iy = CHAIN[x], CHAIN[y]
                        if ix > iy:
                            ix, iy, y = iy, ix, x
                        L_p = 0.0
                        for kj in range(ix + 1, iy + 1):              # 只計兩連桿之間的運動關節；槓桿長用對所有構型成立的上界 LEVER（URDF 固定長度三角不等式），不超過 R_JNT
                            L_p += float(tvq[kj - 1]) * min(LEVER[(kj, y)], R_JNT)
                        need = thr['self'] + 0.5 * L_p
                        m_ = min(pair_d(ci, pr, need, Q[i]), pair_d(cj, pr, need, Q[j])) - need
                        worst['self'] = min(worst['self'], m_)
                        ok_ij = ok_ij and m_ >= 0.0
                if not ok_ij:
                    bad.append((i, j))
            return worst, bad

        res, n_ref = {}, 0
        t1 = time.perf_counter()
        why = evaluate([trajs[m]['Q'][k] for m in trajs for k in trajs[m]['idx']])
        t2 = time.perf_counter()
        for mode in ('now', 'next'):
            tr = trajs[mode]
            tr['idx_self'] = list(tr['idx'])
            if why:
                worst, bad_e, bad_s = dict(env=-9.9, base=-9.9, self=-9.9), [], []
            else:
                w_e, bad_e = certify(mode, ('env',))                     # 環境／底盤：初始取樣＋區間界（不過 ⇒ 不通過；這兩項餘裕大，不加密）
                w_s, bad_s = certify(mode, ('self',))
                if bad_s and any(j - i > 1 for i, j in bad_s):          # 自碰撞：不過的區間加密到逐物理步（只算自碰撞，不走池）
                    add = sorted({k for i, j in bad_s for k in range(i + 1, j)})
                    evaluate([tr['Q'][k] for k in add], self_only=True)
                    tr['idx_self'] = sorted(set(tr['idx']) | set(add)); n_ref += 1
                    w_s, bad_s = certify(mode, ('self',))
                worst = dict(env=w_e['env'], base=w_e['base'], self=w_s['self'])
            cov = (not why) and not bad_e and not bad_s
            mc = {k_: (round(min(cache[key(tr['Q'][k])][kk] for k in tr['idx']), 4) if not why else None) for k_, kk in (('env', 'env'), ('base', 'base'), ('self', 'self_min'))}
            res[mode] = dict(ok=bool(tr['lim_ok'] and tr['settled'] and cov), limits_ok=tr['lim_ok'], settled=tr['settled'], settle_s=tr['settle_s'], pipe_drained_s=tr['pipe_drained_s'], coverage_ok=bool(cov), min_clear=mc,
                             margin_after_interval_bound={k_: round(v_, 4) for k_, v_ in worst.items()}, n_samples=len(tr['idx']), n_samples_self=len(tr['idx_self']), n_steps=len(tr['Q']) - 1, all_still_s=tr['settle_s'])
        mcc = {k_: (None if why else min(res['now']['min_clear'][k_], res['next']['min_clear'][k_])) for k_ in ('env', 'base', 'self')}
        mab = {k_: min(res['now']['margin_after_interval_bound'][k_], res['next']['margin_after_interval_bound'][k_]) for k_ in ('env', 'base', 'self')}
        return dict(ok=bool(res['now']['ok'] and res['next']['ok']), why=why, now=res['now'], next=res['next'], min_clear=mcc if not why else {}, margin_after_interval_bound=mab, n_eval=len(cache), n_refine=n_ref,
                    coverage='區間界（路徑長上界 L；兩端淨空較小值 − L/2 ≥ 門檻；自碰撞逐對）＋不過則加密到逐物理步', ms_traj=round((t1 - t0) * 1e3, 1), ms_eval0=round((t2 - t1) * 1e3, 1), ms_cert=round((time.perf_counter() - t2) * 1e3, 1), ms=round((time.perf_counter() - t0) * 1e3, 1))

    return stop_check, to_map


def disposition(sc):
    """同 merge_node.stop_disposition（tag＝'tick'）。"""
    if not isinstance(sc, dict) or not isinstance(sc.get('now'), dict) or not isinstance(sc.get('next'), dict) or sc.get('why'):
        return 'abort'
    if sc['now'].get('ok') is True and sc['next'].get('ok') is True:
        return 'commit'
    if sc['now'].get('ok') is True:
        return 'stop_now'
    return 'abort'
