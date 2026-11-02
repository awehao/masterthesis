#!/usr/bin/env python3
"""F1 兩輪的接觸欄位映射（純函式，Isaac 主迴圈與 test_bottle_f1_contact 共用；不依賴 Isaac）。
過濾欄：第 0 欄＝/World/bottle，其餘＝table_paths（內含 /World/bottle2）。第二輪把第 0 欄與 /World/bottle2 那欄對調：作用瓶永遠在第 0 欄，另一瓶留在禁止接觸欄。
作用瓶自己的視圖（↔第一桌、↔第二桌、↔環境）每輪各一組，由 pick 選。正常桌面支撐不進禁止接觸：瓶↔第一桌在離桌確立前記 0，瓶↔第二桌由放置判定處理。
"""
import numpy as np


def filter_perm(table_paths, other='/World/bottle2'):
    n = 1 + len(table_paths)
    i = 1 + list(table_paths).index(other)
    p = list(range(n))
    p[0], p[i] = i, 0
    return p, i


def pick(stage, a, b):
    return a if stage == 0 else b


def permute(stage, M, perm):
    """第二輪：作用瓶換到第 0 欄。"""
    return M if (stage == 0 or M is None) else M[:, perm]


def summarize(stage, Mf, Ma, Me, Mt, Ms, Mb, M2, perm, f_sign, lift_ok, has_env=True):
    """由已讀到的接觸矩陣算控制器要的量（公式同 dynsr34）。Mf／Ma／Me 尚未換欄；Mt／Ms／M2 已是作用瓶那組。回傳 (ff, ft, fp, fp_raw, forb)。"""
    Mf, Ma = permute(stage, Mf, perm), permute(stage, Ma, perm)
    Me = [permute(stage, x, perm) for x in Me]
    ff = None if Mf is None or not np.isfinite(Mf[:, 0]).all() else tuple(float(x) for x in np.linalg.norm(Mf[:, 0], axis=1))
    ft = None if Mt is None or not np.isfinite(Mt).all() else float(abs(Mt[0, 0, 2]))
    forb = {}
    if has_env:
        forb['bottle_env'] = float('nan') if Ms is None or not np.isfinite(Ms).all() else float(np.linalg.norm(Ms, axis=2).max())
    fp = None if M2 is None or not np.isfinite(M2).all() else np.asarray(M2[0, 0], float)
    fp_raw = None if M2 is None else np.asarray(M2[0, 0], float).copy()
    if fp is not None:
        fp = f_sign * fp                                     # 固定方向約定（起始靜置時瓶↔第一桌量得的符號），反向力不會被翻成支撐
    forb['bottle_table1'] = float('nan') if ft is None else (float(ft) if lift_ok else 0.0)   # 抬離成立後瓶不得再碰第一桌
    forb['base_env'] = float('nan') if Mb is None else float(np.linalg.norm(Mb, axis=2).max())
    forb['finger_table'] = float('nan') if Mf is None else float(np.linalg.norm(Mf[:, 1:], axis=2).max())
    forb['arm_bottle'] = float('nan') if Ma is None else float(np.linalg.norm(Ma[:, 0], axis=1).max())
    forb['arm_table'] = float('nan') if Ma is None else float(np.linalg.norm(Ma[:, 1:], axis=2).max())
    for i, M in enumerate(Me):
        forb[f'extra{i}_bottle'] = float('nan') if M is None else float(np.linalg.norm(M[:, 0], axis=1).max())
        forb[f'extra{i}_table'] = float('nan') if M is None else float(np.linalg.norm(M[:, 1:], axis=2).max())
    return ff, ft, fp, fp_raw, forb, Mf


def support_check(fz, fz2, f_t2, f_t2_2, f_env, f_env_2, mg, tol_n=0.05):
    """動作前：兩支瓶都靜置第一桌：對第一桌的法向力都要有限、夠大、同方向；對第二桌與環境都應為零。回傳 (ok, why, F_SIGN)。"""
    vals = (fz, fz2, f_t2, f_t2_2, f_env, f_env_2)
    if not all(v is not None and np.isfinite(v) for v in vals):
        return False, 'missing', None
    if abs(fz) < 0.5 * mg or abs(fz2) < 0.5 * mg:
        return False, 'support_too_small', None
    if (fz > 0.0) != (fz2 > 0.0):
        return False, 'support_sign_mismatch', None
    if max(abs(f_t2), abs(f_t2_2), abs(f_env), abs(f_env_2)) > tol_n:
        return False, 'unexpected_contact', None
    return True, None, 1.0 if fz > 0.0 else -1.0
