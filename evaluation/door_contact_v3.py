#!/usr/bin/env python3
"""雙面把手門的接觸分類與紀錄（Codex r46；門規格 door_local_v3、幾何 door_geom_v3、資產 door_asset_v3）。純邏輯，不依賴 Isaac。
舊介面 door_k2_io.classify_contacts（單一把手座標、bar／ambiguous／panel）不改、原義保留；本模組另立，不自動接入換抓准入或中止規則。

輸入：一個物理步的門扇剛體接觸（世界座標接觸點 P (N,3)、力 F：(N,) 大小或 (N,3) 向量、各點機器人側部位名）、當時門角 q（rad）、門根位姿 T_world_door。
  注意：Isaac 張量通常只給「門扇剛體」層級的接觸；本分類是**以幾何把接觸點歸到形狀**，不是讀到碰撞 prim 身分。
分類（每點恰一類）：
  invalid      點或力非有限、力大小為負（純量）——不計入任何力和，另計數；
  bar_out      外側橫桿（n 面）；bar_in 內側橫桿（−n 面）；
  post_out0／post_out1／post_in0／post_in1  四根立柱；
  panel        門板（leaf_box）；
  ambiguous    無法唯一辨識：同時落在兩個以上形狀的容差帶內（例：桿—立柱接合、立柱—門板接合）、落在橫桿的立柱 u 區段內或桿端外、
               或不在任何已知形狀的容差帶內（unmatched）；附原因，不吸收到最近的橫桿。
  形狀容差帶：橫桿＝到桿軸線段（|s| ≤ 半長）的徑向距離 ≤ 半徑＋tol（同舊介面的 near_axis 定義；桿端外 |s| > 半長 且徑向 ≤ 半徑＋tol 記 ambiguous:bar_end）；
             立柱／門板＝有向方盒有號距離 ≤ tol。
力：向量取範數、純量取原值（負值為 invalid）；各類以**大小**相加，不正負抵銷。
步狀態：status ∈ {'ok', 'no_data', 'query_failed'}；非 ok 時各類力和為 None（不記成零力）。ok 但沒有接觸點 ⇒ 各類 0（真的無接觸）。
對帳：Σ 各有效類力和 ＝ 有效點力大小總和（相對 1e-9）；每點恰計一次（各類點數和＋invalid 數＝N）。reconciled 只表示有效點帳目相符。
r2（Codex r47）：
  - 整步完整旗標 complete＝status ok 且無 invalid 點；有 invalid 點時有效點小計保留，但**不能作完整無接觸證據**（全部點無效也不是「無接觸」）。
  - 門根變換核對：有限、4×4、末列 [0,0,0,1]、旋轉正交（‖RᵀR−I‖∞ ≤ 1e-6）、det ≈ +1（|det−1| ≤ 1e-6）；不符 ⇒ status query_failed＋原因。
  - 紀錄保存實際使用的變換（列主序 16 值）與規格識別 spec_id（呼叫端給，例如規格名＋sha256），可獨立重算。
  - extras：每點附加原始資料（例：fn_signed 有號法向力、normal、sep 分離量）原樣保存；分類只用力大小。
"""
from __future__ import annotations

import math

import numpy as np

import door_geom_v3 as G3

CLASSES = ('bar_out', 'bar_in', 'post_out0', 'post_out1', 'post_in0', 'post_in1', 'panel', 'ambiguous')
STATUS = ('ok', 'no_data', 'query_failed')


def _bar_flags(P, T, H, tol):
    c, a = T[:3, 3], T[:3, 0]
    half, r = H['bar']['length'] / 2, H['bar']['radius']
    s = (P - c) @ a
    sc = np.clip(s, -half, half)
    radial = np.linalg.norm(P - (c + sc[:, None] * a), axis=1)
    near = radial <= r + tol
    on_bar = near & (np.abs(s) <= half)
    end = near & (np.abs(s) > half)
    in_post = np.zeros(len(P), bool)
    for du in H['posts']['along_u']:
        in_post |= np.abs(np.abs(s) - abs(du)) <= H['posts']['size'][0] / 2 + tol
    return on_bar, end, in_post


def check_T(T):
    """門根變換核對；回傳 (ok, why)。"""
    try:
        T = np.asarray(T, float)
    except (TypeError, ValueError):
        return False, '變換無法轉成數值陣列'
    if T.shape != (4, 4):
        return False, f'變換形狀 {T.shape} 不是 4×4'
    if not np.all(np.isfinite(T)):
        return False, '變換含非有限值'
    if np.max(np.abs(T[3] - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-9:
        return False, '變換末列不是 [0,0,0,1]'
    R = T[:3, :3]
    if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-6:
        return False, '旋轉不正交（含縮放或剪切）'
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-6:
        return False, '旋轉行列式不為 +1'
    return True, ''


def classify_step(P, F, q, D, tol, robot_links=None, T_world_door=None, status='ok', step=None, t=None, source='geom_v3',
                  spec_id=None, extras=None):
    """回傳 {'status','step','t','points':[...],'sums':{類:力和}|None,'counts':{類:點數}|None,'n_invalid','total_valid','reconciled'}。"""
    if status not in STATUS:
        raise ValueError(f'status 須為 {STATUS}')
    if not (isinstance(tol, (int, float)) and math.isfinite(tol) and tol >= 0):
        raise ValueError('tol 須為有限非負數')
    Tw = np.eye(4) if T_world_door is None else T_world_door
    tok, twhy = check_T(Tw)
    out = {'status': status, 'step': step, 't': t, 'source': source, 'tol_m': float(tol), 'q_door_rad': None, 'spec_id': spec_id,
           'T_world_door': [float(v) for v in np.asarray(Tw, float).reshape(-1)] if tok else None,
           'reconciled_scope': '只表示有效點帳目相符'}
    none = {'points': [], 'sums': None, 'counts': None, 'n_invalid': None, 'total_valid': None, 'reconciled': None, 'complete': False}
    try:
        qf = float(q)
    except (TypeError, ValueError):
        qf = float('nan')
    out['q_door_rad'] = qf if math.isfinite(qf) else None
    if status != 'ok':
        out.update(none)
        return out
    if not math.isfinite(qf):
        out.update(none, status='query_failed', why='門角非有限')
        return out
    if not tok:
        out.update(none, status='query_failed', why=f'門根變換不合格：{twhy}')
        return out
    q = qf
    P = np.asarray(P, float).reshape(-1, 3) if np.size(P) else np.zeros((0, 3))
    F = np.asarray(F, float)
    n = len(P)
    if F.ndim == 2:
        if F.shape != (n, 3):
            raise ValueError('向量力須為 (N,3)')
        fmag = np.linalg.norm(F, axis=1)
        fbad = ~np.all(np.isfinite(F), axis=1)
    else:
        F = F.reshape(-1)
        if len(F) != n:
            raise ValueError('力與點數不符')
        fmag = F.copy()
        fbad = ~np.isfinite(F) | (F < 0)
    links = list(robot_links) if robot_links is not None else [None] * n
    if len(links) != n:
        raise ValueError('robot_links 與點數不符')
    Tw = np.asarray(Tw, float)
    Ti = np.linalg.inv(Tw)
    ex = {}
    for k_, v_ in (extras or {}).items():
        a_ = np.asarray(v_, float)
        if len(a_) != n:
            raise ValueError(f'extras[{k_}] 長度與點數不符')
        ex[k_] = a_
    pbad = ~np.all(np.isfinite(P), axis=1)
    Pl = np.where(pbad[:, None], 0.0, P) @ Ti[:3, :3].T + Ti[:3, 3]           # 門局部
    H, h = D['handle'], float(D['handle']['height'])
    bo, eo, po = _bar_flags(Pl, G3.handle_pose(D, q, h), H, tol)
    bi, ei, pi_ = _bar_flags(Pl, G3.handle_pose_inner(D, q), H, tol)
    sh = {name: (c, R, hs) for name, c, R, hs in G3.shapes(D, q, h)}
    box = {'post_out0': 'post0', 'post_out1': 'post1', 'post_in0': 'post_in0', 'post_in1': 'post_in1', 'panel': 'leaf'}
    near_box = {k: G3.box_dist(Pl, *sh[v]) <= tol for k, v in box.items()}
    cls, why = [], []
    for i in range(n):
        if pbad[i] or fbad[i]:
            cls.append('invalid')
            why.append('點非有限' if pbad[i] else '力非有限或為負')
            continue
        hits = [k for k in box if near_box[k][i]]
        if bo[i]:
            hits.append('bar_out')
        if bi[i]:
            hits.append('bar_in')
        if eo[i] or ei[i]:
            cls.append('ambiguous')
            why.append('bar_end')
            continue
        if len(hits) == 0:
            cls.append('ambiguous')
            why.append('unmatched')
        elif len(hits) > 1:
            cls.append('ambiguous')
            why.append('multi:' + '+'.join(sorted(hits)))
        elif hits[0] == 'bar_out' and po[i] or hits[0] == 'bar_in' and pi_[i]:
            cls.append('ambiguous')
            why.append(f'{hits[0]}:post_u_range')
        else:
            cls.append(hits[0])
            why.append('')
    valid = np.array([c != 'invalid' for c in cls], bool)
    sums = {k: float(sum(fmag[i] for i in range(n) if cls[i] == k)) for k in CLASSES}
    counts = {k: sum(1 for c in cls if c == k) for k in CLASSES + ('invalid',)}
    tot = float(fmag[valid].sum()) if valid.any() else 0.0
    rec = abs(sum(sums.values()) - tot) <= 1e-9 * max(1.0, tot) and sum(counts.values()) == n
    def fin(x):
        return float(x) if math.isfinite(float(x)) else None

    pts_out = []
    for i in range(n):
        d_ = {'p_world': [fin(v) for v in P[i]], 'f': fin(fmag[i]), 'robot_link': links[i], 'class': cls[i], 'why': why[i]}
        for k_, a_ in ex.items():
            d_[k_] = [fin(v) for v in np.atleast_1d(a_[i])] if np.ndim(a_[i]) else fin(a_[i])
        pts_out.append(d_)
    out.update({'points': pts_out, 'sums': sums, 'counts': counts, 'n_invalid': counts['invalid'], 'total_valid': tot, 'reconciled': bool(rec),
                'complete': bool(counts['invalid'] == 0)})
    return out
