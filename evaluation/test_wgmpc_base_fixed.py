#!/usr/bin/env python3
"""PARK_FIXED 求解核心測試：base_fixed 的等式與預設路徑不變。

  1 預設（base_fixed=False）：以實錄 mt_b1_02_P（MOTM=0 舊停車組）的逐輪求解輸入重解，與**改動前的核心**
    （scratch 保存的 wgmpc_core_sp_pre_park.py／wgmpc_core_pre_park.py）逐位元相同
  2 base_fixed=True：解出的整個控制序列 |U[:, 0:3]| ≤ 1e-12（OSQP 浮點解不是精確 0.0；實測約 1e-24，
    遠低於執行端模式判準 1e-6）；接受的解殘差 ≤ r_tol
  3 u_prev 底盤為小的非零值（1e-3 m/s，在 amax·dt 內）：仍可行並降到 0（「u_prev 非零 ⇒ 不可行」不成立）
  4 vmax() 不受 base_fixed 影響（成本 1/vmax² 不變）；vbox() 底盤為 0
  5 舊核心（wgmpc_core.solve）遇 base_fixed=True 拒絕

    python3 evaluation/test_wgmpc_base_fixed.py [<舊核心目錄>]
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import horizon_replay as HR                                             # noqa: E402
from ammr_wholebody_mpc import wgmpc_core as C                          # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                       # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def load_old(dirpath):
    """把改動前的兩個核心檔載成獨立套件（ammr_old.wgmpc_core／ammr_old.wgmpc_core_sp）。"""
    import types
    pkg = types.ModuleType('ammr_old')
    pkg.__path__ = []
    sys.modules['ammr_old'] = pkg
    mods = {}
    for name, fn in (('wgmpc_core', 'wgmpc_core_pre_park.py'), ('wgmpc_core_sp', 'wgmpc_core_sp_pre_park.py')):
        src = open(os.path.join(dirpath, fn)).read()
        src = src.replace('from .wgmpc_core import', 'from ammr_old.wgmpc_core import')
        src = src.replace('from .', 'from ammr_wholebody_mpc.')
        m = types.ModuleType(f'ammr_old.{name}')
        sys.modules[f'ammr_old.{name}'] = m
        exec(compile(src, fn, 'exec'), m.__dict__)
        mods[name] = m
    return mods['wgmpc_core_sp']


def main():
    old_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run = os.path.join(HERE, 'runs', 'mt_b1_02_P')
    sol, ident = HR.load_run(run)
    args = sol['args']
    K = WholeBodyKinematics.from_urdf_file(args['urdf'])
    rows = [r for r in sol['log'] if r.get('solve_in') and r['solve_in'].get('U_warm') is not None][:40:4]
    # 4 vmax／vbox
    cfg = HR.base_cfg(args, ident)
    v0 = cfg.vmax().copy()
    cfg.base_fixed = True
    check('vmax_unchanged_by_base_fixed', np.array_equal(cfg.vmax(), v0))
    check('vbox_base_zero', np.all(cfg.vbox()[:3] == 0) and np.array_equal(cfg.vbox()[3:], v0[3:]))
    # 1 預設路徑逐位元
    if old_dir:
        SO = load_old(old_dir)
        same, n = True, 0
        for r in rows:
            cfg_new = HR.cfg_from_record(args, ident, r)
            si = r['solve_in']
            z0 = S.make_z(si['q_pred'], si['s_pred'])
            U_warm = np.asarray(si['U_warm'], float)
            rn = S.solve_sp(K, z0, si['u_prev'], np.asarray(si['T_cyc']).reshape(4, 4), cfg_new, U_warm=U_warm)
            cfg_old = SO.WGMPCConfigSP(**{k: getattr(cfg_new, k) for k in cfg_new.__dataclass_fields__
                                          if k != 'base_fixed'})
            ro = SO.solve_sp(K, SO.make_z(si['q_pred'], si['s_pred']), si['u_prev'],
                             np.asarray(si['T_cyc']).reshape(4, 4), cfg_old, U_warm=U_warm)
            n += 1
            if not (rn.ok == ro.ok and (not rn.ok or np.array_equal(rn.U, ro.U))):
                same = False
        check('default_path_bitwise_vs_pre_change', same and n > 0, f'n={n}')
    else:
        print('SKIP default_path_bitwise_vs_pre_change（未給舊核心目錄）')
    # 2 base_fixed：底盤恆零
    okall, zero, resid = True, True, 0.0
    for r in rows:
        c = HR.cfg_from_record(args, ident, r)
        c.base_fixed = True
        si = r['solve_in']
        up = np.asarray(si['u_prev'], float).copy()
        up[:3] = 0.0
        res = S.solve_sp(K, S.make_z(si['q_pred'], si['s_pred']), up,
                         np.asarray(si['T_cyc']).reshape(4, 4), c, U_warm=None)
        okall &= bool(res.ok)
        if res.ok:
            zero &= bool(np.abs(np.asarray(res.U)[:, :3]).max() <= 1e-12)
            resid = max(resid, float(res.max_residual))
    check('base_fixed_solves', okall, f'n={len(rows)}')
    check('base_fixed_U_base_all_zero', zero)
    check('base_fixed_residual_ok', resid <= HR.base_cfg(args, ident).r_tol, resid)
    # 3 小的非零 u_prev（底盤 1e-3 m/s）
    r = rows[0]
    c = HR.cfg_from_record(args, ident, r)
    c.base_fixed = True
    si = r['solve_in']
    up = np.asarray(si['u_prev'], float).copy()
    up[:3] = [1e-3, -1e-3, 1e-3]
    res = S.solve_sp(K, S.make_z(si['q_pred'], si['s_pred']), up,
                     np.asarray(si['T_cyc']).reshape(4, 4), c, U_warm=None)
    check('small_nonzero_uprev_still_feasible_to_zero',
          bool(res.ok) and np.abs(np.asarray(res.U)[:, :3]).max() <= 1e-12, res.reason)
    # 5 舊核心拒絕
    try:
        cc = C.WGMPCConfig(N=5, dt=0.05)
        cc.base_fixed = True
        q0 = np.r_[0.0, 0.0, 0.0, np.zeros(6)]
        C.solve(K, q0, np.zeros(9), np.eye(4), cc)
        check('ideal_core_rejects_base_fixed', False, '未拒絕')
    except ValueError:
        check('ideal_core_rejects_base_fixed', True)
    print(f'{len(fails)} 失敗')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
