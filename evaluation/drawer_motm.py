#!/usr/bin/env python3
"""抽屜把手移動中抓取選位（Howard 11/19「門把、抽屜把手都要改 要時間最大化 不要停了在動手臂」）。
底盤面向櫃子、沿櫃面橫移（e＝±橫桿方向）以 v_g 經過抓取底盤位 b0；TCP 世界固定在把手抓取位姿；
時間窗內閉爪＋夾持證據＋理想連接；之後橫移速度以時間常數 TAU 指數衰減（橫向再滑 v_g·TAU），同時沿拉出方向拉開 OPEN_M ⇒ 底盤速度不歸零。
選位條件（同控制器淨空 drawer_reach._clear：env／base／self 執行門檻，底盤多留 BASE_MARGIN）：
  1. 經過路線 s∈[S0, v_g·T_WIN] 每 5 cm：關閉時把手 IK（同分支連續、關節跳 ≤0.3 rad）＋淨空；
  2. 拉開終點：底盤＝b0＋(s_end＋v_g·TAU)·e＋OPEN_M·拉出方向，把手在開度 OPEN_M 的 IK＋淨空（抽屜方盒隨開度）。
評分＝目前位置到路線起點＋w_sig/σ_min。座標＝里程（同 drawer_reach）。"""
import math
import os

import numpy as np

import drawer_reach as DR
import door_fast_kin as DFK

D, M = DR.D, DR.M
S0 = float(os.environ.get('DRM_S0', '-0.05'))
T_WIN = float(os.environ.get('DRM_T_WIN', '1.9'))                    # 閉爪開始容許 0.5＋閉爪 0.6＋確認 0.3＋寬限 0.5 s
TAU = float(os.environ.get('DRM_TAU', '1.0'))                        # 連接後橫移衰減時間常數（s）
T_ARM = float(os.environ.get('DRM_T_ARM', '4.0'))                    # 手臂展開最短時間（同茶几 s73）
DS = 0.05


def _ik_cont(T, b, prev):
    q, r, sm = DFK.ik_closure_fc(T, b, prev)
    q = (np.asarray(q, float) + math.pi) % (2 * math.pi) - math.pi
    lo, hi = np.asarray(M.LIM.lower, float), np.asarray(M.LIM.upper, float)
    if r > 1e-4 or np.any(q < lo + 0.05) or np.any(q > hi - 0.05) or sm < 0.02 or float(np.abs(q - prev).max()) > 0.3:
        return None
    return q, float(sm)


def select_handle_motm(b_now, v_g, w_dist=1.0, w_sig=0.02):
    best, by_dir = None, {}
    p = np.asarray(D.PULL[:2], float); n = np.array([-p[1], p[0]])
    h0 = D.handle_T(0.0)[:3, 3]
    s1 = float(v_g) * T_WIN
    ss = np.round(np.arange(S0, s1 + 1e-9, DS), 4)
    for b in DR._frame_bases(h0, np.arange(0.40, 0.66, 0.025), np.arange(-0.15, 0.151, 0.05), (0.0,)):
        g0 = DR._ik_ok(lambda f: D.handle_T(0.0, f), b, 0.0)
        if g0 is None:
            continue
        flip = bool(g0['flip'])
        T0 = D.handle_T(0.0, flip)
        for sg in (1.0, -1.0):
            e = sg * n
            ok, sig, env, prev = True, g0['sigma'], g0['env'], np.asarray(g0['q'], float)
            for direction in (1, -1):                                # 由 s＝0 往兩端延伸，保持同分支
                prev = np.asarray(g0['q'], float)
                for s_ in (ss[ss > 0] if direction > 0 else ss[ss < 0][::-1]):
                    bb = np.array([b[0] + s_ * e[0], b[1] + s_ * e[1], b[2]])
                    r = _ik_cont(T0, bb, prev)
                    if r is None:
                        ok = False; break
                    c_ok, c = DR._clear(bb, r[0], 0.0)
                    if not c_ok:
                        ok = False; break
                    sig, env, prev = min(sig, r[1]), min(env, float(c['env'])), r[0]
                if not ok:
                    break
            if not ok:
                continue
            s_end = s1 + float(v_g) * TAU                            # 拉開期間橫向再滑
            b1 = np.array([b[0] + s_end * e[0] + DR.OPEN_M * p[0], b[1] + s_end * e[1] + DR.OPEN_M * p[1], b[2]])
            g1 = DR._ik_ok(lambda f, fl=flip: D.handle_T(DR.OPEN_M, fl), b1, DR.OPEN_M, flips=(flip,))
            D.set_open(0.0)
            if g1 is None:
                continue
            s_st = S0 - 0.40 - float(v_g) * T_ARM
            b_st = np.array([b[0] + s_st * e[0], b[1] + s_st * e[1], b[2]])
            if not M.base_free(b_st):
                continue
            sc = w_dist * float(np.hypot(*(b_st[:2] - np.asarray(b_now[:2], float)))) + w_sig / max(min(sig, g1['sigma']), 1e-6)
            cand_ = dict(base=b.tolist(), e=e.tolist(), base_start=b_st.tolist(), base_end=b1.tolist(), flip=flip, v_g=float(v_g), sigma_min=min(sig, g1['sigma']),
                         env_min=min(env, g1['env']), score=sc, T_grasp=T0.tolist(), q_grasp=[float(x) for x in g0['q']], s_range=[float(ss[0]), float(ss[-1])], tau=TAU, t_arm=T_ARM)
            if int(sg) not in by_dir or sc < by_dir[int(sg)]['score']:
                by_dir[int(sg)] = cand_                                  # 每方向留最佳（順路經過時依來向選；Howard 11/20）
            if best is None or sc < best['score']:
                best = dict(cand_)
    D.set_open(0.0)
    if best is not None:
        best['alts'] = [v_ for k_, v_ in by_dir.items() if v_['e'] != best['e']]
    return best


if __name__ == '__main__':
    import time
    for v in (0.08, 0.10):
        w0 = time.perf_counter()
        r = select_handle_motm(np.array([-3.5, 10.3, math.pi]), v)
        print(v, round(time.perf_counter() - w0, 1), 's', None if r is None else {k: (np.round(r[k], 3).tolist() if isinstance(r[k], list) else r[k]) for k in ('base', 'e', 'base_start', 'base_end', 'flip', 'sigma_min', 'env_min', 's_range')})
