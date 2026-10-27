#!/usr/bin/env python3
"""換抓參考包登錄核驗（Codex r60 §1；純邏輯）。任務端（RefRuntime3）、執行端 Gate6、判定器使用參考包前都須先通過 validate(pkg)。
核驗：
  1. 內側抓法：pkg['inner_grasp']['T_HG'] 與 pkg['inner_grasp']['phi_deg'] 一致——T_HG＝Rx(φ)·T_HG_base（T_HG_base＝door_coopen_reach_v1.T_HG()）；
  2. 參考抓取端點與登錄抓法一致：H_in(q_ref)·T_HG 對 grasp_T 位置 ≤ 1 mm、姿態 ≤ 0.01 rad（N1 插入以 solve_w 收斂到 1 mm／0.01 rad）；
     預抓取＝上式·tz(−0.06) 對 pregrasp_T 同界；
  3. B1 與 mode 5 起點底盤一致（≤ 1e-6）；
  4. 外側抓取基準獨立：pkg['outer_grasp_basis'] 的檔案存在且 sha256 相符（不被內側新抓法覆寫）；
  5. 來源：replan 檔與 source（時間化）檔存在且 sha256 相符。
任一不符 ⇒ (False, 原因)；呼叫端拒用（不得以舊抓法或舊 B1 承接）。
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POS_TOL, ROT_TOL = 1e-3, 0.01


def Rx4(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[1:3, 1:3] = [[c, -s], [s, c]]
    return T


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _rot_err(Ra, Rb):
    return math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2)))


def validate(pkg, D3=None):
    import yaml
    import door_coopen_reach_v1 as RV
    import door_geom_v3 as G3
    if D3 is None:
        D3 = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')))
    try:
        ig = pkg['inner_grasp']
        T = np.asarray(ig['T_HG'], float).reshape(4, 4)
        phi = float(ig['phi_deg'])
    except (KeyError, TypeError, ValueError):
        return False, 'inner_grasp_missing'
    if 'T_HG' in pkg:
        return False, 'ambiguous_T_HG_key'
    T_exp = Rx4(math.radians(phi)) @ np.asarray(RV.T_HG(), float)
    if np.max(np.abs(T - T_exp)) > 1e-9:
        return False, 'inner_T_HG_ne_phi'
    q = math.radians(float(pkg['q_ref_deg']))
    Tg = G3.handle_pose_inner(D3, q) @ T
    Tp = Tg.copy()
    Tp[:3, 3] = Tg[:3, 3] - 0.06 * Tg[:3, 2]
    for name, Tref, Texp in (('grasp', pkg['grasp_T'], Tg), ('pregrasp', pkg['pregrasp_T'], Tp)):
        R = np.asarray(Tref, float).reshape(4, 4)
        ep, er = float(np.linalg.norm(R[:3, 3] - Texp[:3, 3])), _rot_err(R[:3, :3], Texp[:3, :3])
        if ep > POS_TOL or er > ROT_TOL:
            return False, f'{name}_ne_registered_grasp:{ep * 1e3:.2f}mm/{math.degrees(er):.2f}deg'
    b5 = np.asarray(pkg['segments']['5']['base'][0], float)
    if np.max(np.abs(np.asarray(pkg['B1'], float) - b5)) > 1e-6:
        return False, 'B1_ne_mode5_start'
    og = pkg.get('outer_grasp_basis') or {}
    p = os.path.join(os.path.dirname(HERE), og.get('path', ''))
    if not og.get('path') or not os.path.exists(p) or _sha(p) != og.get('sha256'):
        return False, 'outer_grasp_basis_missing_or_changed'
    for key in ('replan', 'source'):
        src = pkg.get(f'{key}_file') or {}
        p = os.path.join(HERE, src.get('path', ''))
        if not src.get('path') or not os.path.exists(p) or _sha(p) != src.get('sha256'):
            return False, f'{key}_file_missing_or_changed'
    return True, None
