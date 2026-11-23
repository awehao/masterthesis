#!/usr/bin/env python3
"""由求解節點的逐輪紀錄重建求解器輸入，離線重解一輪（開迴路，不推進物理）。

需要 `solve_in` 內含 wgmpc_cycle_record.solver_io_record 的欄位（T_cyc、U_warm、
arm_bias、solver_N）以及 `coord`。舊趟次（2026-10-04 之前）沒有這些欄位 ⇒
`cfg_from_record` 會拋 MissingInput，**不以推測補齊**。

    N_override：同一份輸入改用另一個 N 求解（H1／H5 的同狀態比較）。此時暖啟動
    不能沿用另一方法的序列 ⇒ 一律冷啟動（U_warm = None），並在結果中標明。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402

REQUIRED = ('q_pred', 's_pred', 'u_prev', 'T_cyc', 'U_warm', 'arm_bias',
            'solver_N')


class MissingInput(KeyError):
    pass


def base_cfg(args, ident):
    """由節點 args 與辨識檔組出與節點相同的 WGMPCConfigSP（不含逐輪項）。"""
    am = S.ArmSetpointModel(alpha=ident['alpha'], bias=ident['bias_rad'],
                            phys_dt=ident['phys_dt_measured_s'])
    return S.WGMPCConfigSP(N=int(args['N']), dt=1.0 / float(args['rate']),
                           tcp=args['tcp'], arm_model=am,
                           w_s=args['w_s'], w_a=args['w_a'],
                           w_s_base=args.get('w_s_base'),
                           w_s_arm=args.get('w_s_arm'),
                           row_scaling=not args.get('no_row_scaling', False),
                           # PARK_FIXED：舊趟次沒有此欄 ⇒ False（行為不變）
                           base_fixed=bool(args.get('base_fixed', False)))


def cfg_from_record(args, ident, rec, N_override=None):
    si = rec.get('solve_in') or {}
    miss = [k for k in REQUIRED if k not in si]
    if miss:
        raise MissingInput(f'紀錄缺 {miss}（舊趟次無逐輪求解輸入）')
    cfg = base_cfg(args, ident)
    cfg.N = int(N_override if N_override is not None else si['solver_N'])
    cfg.arm_model.bias = np.asarray(si['arm_bias'], float)
    # PARK_HOLD：本輪底盤等式值由逐輪紀錄還原（不重新取 odom）。args.park_hold = True 時紀錄**必須**完整有效，
    # 缺失／形狀錯／非有限 ⇒ 紀錄不足（不得回退成自由底盤）。舊趟次（無 park_hold）照舊。
    ph = rec.get('park_hold')
    if args.get('park_hold') or ph is not None:
        import math as _m
        _need = ('u_hold', 'pose', 'anchor', 'servo')
        if not isinstance(ph, dict) or any(k not in ph for k in _need):
            raise MissingInput(f'PARK_HOLD 紀錄缺 {[k for k in _need if not isinstance(ph, dict) or k not in ph]}')
        for k in ('u_hold', 'pose', 'anchor'):
            v = ph[k]
            if not isinstance(v, (list, tuple)) or len(v) != 3 or not all(
                    isinstance(x, (int, float)) and _m.isfinite(float(x)) for x in v):
                raise MissingInput(f'PARK_HOLD 紀錄 {k} 形狀錯或非有限：{v}')
        if not isinstance(ph['servo'], dict):
            raise MissingInput('PARK_HOLD 紀錄 servo 不是 dict')
        cfg.base_hold = tuple(float(x) for x in ph['u_hold'])
    c = rec.get('coord')
    if c is not None:
        for k in ('w_vref', 'base_vref', 'w_qn', 'arm_q_nom', 'w_a', 'w_p'):
            if k in c:
                v = c[k]
                setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    return cfg


def replay_cycle(K, args, ident, rec, N_override=None):
    """回傳 (WGMPCResultSP, 說明 dict)。"""
    si = rec['solve_in']
    cfg = cfg_from_record(args, ident, rec, N_override)
    z0 = S.make_z(np.asarray(si['q_pred'], float),
                  np.asarray(si['s_pred'], float))
    T = np.asarray(si['T_cyc'], float).reshape(4, 4)
    same_N = N_override is None or int(N_override) == int(si['solver_N'])
    U_warm = (None if (si['U_warm'] is None or not same_N)
              else np.asarray(si['U_warm'], float))
    r = S.solve_sp(K, z0, np.asarray(si['u_prev'], float), T, cfg,
                   U_warm=U_warm)
    return r, {'N': cfg.N, 'warm_start': U_warm is not None,
               'warm_note': ('沿用紀錄的暖啟動' if U_warm is not None else
                             '冷啟動（紀錄無暖啟動或 N 不同）')}


def load_run(run_dir):
    sol = json.load(open(os.path.join(run_dir, 'align_solver.json')))
    ident = json.load(open(sol['args']['arm_ident']))
    return sol, ident
