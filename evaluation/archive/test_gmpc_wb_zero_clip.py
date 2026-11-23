#!/usr/bin/env python3
"""gmpc_wb 零速度框夾限容差（GMPCConfig.zero_box_clip_tol，Codex 20261102_184223）的最小核對。
資料：results/bottle/hold_entry_tick.pkl＝R-B2 假世界（carrygov 版）進 HOLD 那一拍的停車求解器外層輸入（底盤上下限 [0, 0]）；運動學用 bottle_s0_ctrl.S0_K()（同控制器）。
1 預設退化相同：zero_box_clip_tol＝None 時，新版與 git HEAD 版 gmpc_wb 對同一拍（停車設定）與改成導航速度框的同一拍，回傳命令、手臂序列與 arm_check 逐位元相同。
2 重現：預設（None）⇒ base_postprocess_changed 退回（同 HEAD）。
3 開容差 (1e-9 m/s, 1e-9 rad/s) ⇒ 不退回、回傳 0、arm_check['seq_tol'] 記原值／回傳值／修正量，標「數值容差內一致」。
4 超出容差仍退回：容差 (1e-12, 1e-12) ⇒ 照原規則退回。
5 非零速度框不能啟用例外：導航速度框（非零）＋容差 ⇒ 與容差 None 逐位元相同。
6 其他後處理不被掩蓋（判定函式 zero_box_clip_ok）：accept_action＝acc_clipped／wheel_scaled／wheel_projected／brake、不可用、未收斂、回傳≠速度框夾限結果、非有限原值、非零上下限 ⇒ 都不適用；
  逐軸：平移 1e-9 內、角速度超出 ⇒ 不適用。
    python3 evaluation/test_gmpc_wb_zero_clip.py
"""
import dataclasses
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402

TICK = os.path.join(HERE, 'results', 'bottle', 'hold_entry_tick.pkl')
OUT = os.path.join(HERE, 'results', 'bottle', 'gmpc_wb_zero_clip_check.json')
NAV_U = (np.array([-0.20, -0.25, -0.8]), np.array([0.35, 0.25, 0.8]))


def head_module():
    """git HEAD 的 ammr_wholebody_mpc 套件複製到暫存目錄、改名 ammr_head 後匯入其 gmpc_wb。"""
    tmp = tempfile.mkdtemp(prefix='gmpc_head_')
    src = os.path.join(tmp, 'x')
    os.makedirs(src)
    tar = subprocess.run(['git', '-C', WS, 'archive', 'HEAD', 'src/ammr_wholebody_mpc/ammr_wholebody_mpc'], capture_output=True, check=True).stdout
    subprocess.run(['tar', '-x', '-C', src], input=tar, check=True)
    shutil.move(os.path.join(src, 'src', 'ammr_wholebody_mpc', 'ammr_wholebody_mpc'), os.path.join(tmp, 'ammr_head'))
    sys.path.insert(0, tmp)
    import importlib
    return importlib.import_module('ammr_head.gmpc_wb')


def solve(mod, d, cfg, K):
    arm = dict(d['arm'], K=K, model=d['arm_model'])
    return mod.GMPC(cfg).solve(d['X_now'], d['X_ref_win'], d['xi_ref_win'], d['xi_prev'], d['obstacles'], arm=arm, **d['kw'])


def same(r1, r2):
    a = np.array_equal(np.asarray(r1.u_opt), np.asarray(r2.u_opt)) and np.array_equal(np.asarray(r1.u_arm_seq), np.asarray(r2.u_arm_seq))
    c1, c2 = dict(r1.arm_check or {}), dict(r2.arm_check or {})
    keys = ('fallback', 'seq_consistent', 'failed', 'finite', 'max_violation', 'base_as_is')
    return bool(a and all(repr(c1.get(k)) == repr(c2.get(k)) for k in keys))


def to_cfg(mod, cfg):
    """把擷取的 GMPCConfig（新版類別）轉成 mod 的 GMPCConfig（HEAD 沒有 zero_box_clip_tol 欄位）。"""
    f = {k: getattr(cfg, k) for k in mod.GMPCConfig.__dataclass_fields__}
    return mod.GMPCConfig(**f)


def main():
    d = pickle.load(open(TICK, 'rb'))
    K = C0.S0_K()
    cfg0 = d['cfg']
    res, ok = {}, {}
    H = head_module()
    cfg_nav = dataclasses.replace(cfg0, u_min=NAV_U[0], u_max=NAV_U[1])
    r_new, r_head = solve(GW, d, cfg0, K), solve(H, d, to_cfg(H, cfg0), K)
    rn_new, rn_head = solve(GW, d, cfg_nav, K), solve(H, d, to_cfg(H, cfg_nav), K)
    ok['1_default_same_park'] = same(r_new, r_head)
    ok['1_default_same_nav'] = same(rn_new, rn_head)
    ok['2_reproduce'] = (r_new.arm_check or {}).get('fallback') == 'base_postprocess_changed' and (r_head.arm_check or {}).get('fallback') == 'base_postprocess_changed'
    r_tol = solve(GW, d, dataclasses.replace(cfg0, zero_box_clip_tol=(1e-9, 1e-9)), K)
    ck = r_tol.arm_check or {}
    ok['3_tol_pass'] = ck.get('fallback') is None and ck.get('seq_consistent') is True and (ck.get('seq_tol') or {}).get('label') == '數值容差內一致' and not np.any(np.asarray(r_tol.u_opt))
    res['3_seq_tol'] = ck.get('seq_tol')
    r_tight = solve(GW, d, dataclasses.replace(cfg0, zero_box_clip_tol=(1e-12, 1e-12)), K)
    ok['4_exceed_fallback'] = (r_tight.arm_check or {}).get('fallback') == 'base_postprocess_changed'
    rn_tol = solve(GW, d, dataclasses.replace(cfg_nav, zero_box_clip_tol=(1e-9, 1e-9)), K)
    ok['5_nonzero_box_same'] = same(rn_tol, rn_new) and 'seq_tol' not in (rn_tol.arm_check or {})
    cz = dataclasses.replace(cfg0, zero_box_clip_tol=(1e-9, 1e-9))
    raw = np.array([7.4e-11, 2.3e-10, -6.4e-12])
    z = np.zeros(3)
    f = GW.zero_box_clip_ok
    cases = dict(as_is=f(cz, z, raw, 'as_is', True, 'solved') is not None,
                 acc_clipped=f(cz, z, raw, 'acc_clipped', True, 'solved') is None, wheel_scaled=f(cz, z, raw, 'wheel_scaled', True, 'solved') is None,
                 wheel_projected=f(cz, z, raw, 'wheel_projected', True, 'solved') is None, brake=f(cz, z, raw, 'brake', False, 'solved') is None,
                 not_usable=f(cz, z, raw, 'as_is', False, 'solved') is None, not_solved=f(cz, z, raw, 'as_is', True, 'solved inaccurate') is None,
                 other_postproc=f(cz, np.array([1e-12, 0.0, 0.0]), raw, 'as_is', True, 'solved') is None, nonfinite=f(cz, z, np.array([np.nan, 0, 0]), 'as_is', True, 'solved') is None,
                 nonzero_box=f(dataclasses.replace(cz, u_min=NAV_U[0], u_max=NAV_U[1]), z, raw, 'as_is', True, 'solved') is None,
                 ang_exceed=f(cz, z, np.array([5e-10, 5e-10, 2e-9]), 'as_is', True, 'solved') is None, lin_exceed=f(cz, z, np.array([2e-9, 0.0, 0.0]), 'as_is', True, 'solved') is None,
                 tol_none=f(cfg0, z, raw, 'as_is', True, 'solved') is None)
    ok['6_helper'] = all(cases.values())
    res['6_cases'] = cases
    res['ok'] = ok
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k, v in ok.items():
        print(('✓' if v else '✗'), k)
    print('seq_tol', res['3_seq_tol'])
    print('全過' if all(ok.values()) else '未過')
    return 0 if all(ok.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
