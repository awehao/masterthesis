"""W-GMPC 設定點版求解的「候選首筆承接約束」（Codex r51 §1）。**wgmpc_core_sp.py 不改**（多份凍結清單登錄其雜湊）。

  候選解另加：|u0 − u_applied| ≤ a_max × dt_phys（只約束首筆；後續預測步仍以 cfg.dt 的加速度框）。
  - u_applied 由呼叫端給：須是有來源步號、時刻與有效性的**實際套用回報**（本模組只檢查形狀與有限值）。
  - 約束放進 QP（build_constraints 多一個區塊 'first_step_accept'）並因此進入 solve_sp 的最終殘差可行性核對；不在解完後裁切 u0。
  - 不改 cfg.dt、權重或任何加速度上限（a_max 取 cfg.amax()）。
做法：solve_sp_fs 與 wgmpc_core_sp.solve_sp **是同一份程式碼物件**，只把它的全域名 build_constraints_sp 換成本檔的包裝；
  包裝先呼叫原 build_constraints_sp，cfg.first_step 為 None 時原樣回傳 ⇒ 行為逐位元相同（test_wgmpc_core_sp_fs F1 核對）。
"""
from __future__ import annotations

import types
from dataclasses import dataclass

import numpy as np

from . import wgmpc_core_sp as S

NU = S.NU


@dataclass
class WGMPCConfigSPFS(S.WGMPCConfigSP):
    """first_step＝(u_applied 9 維, dt_phys) 或 None（不啟用）。"""
    first_step: tuple | None = None


def _check_first_step(fs):
    u, dtp = fs
    u = np.asarray(u, float).reshape(-1)
    if u.shape != (NU,) or not np.all(np.isfinite(u)):
        raise ValueError('first_step 的 u_applied 須為有限 9 維')
    dtp = float(dtp)
    if not (np.isfinite(dtp) and dtp > 0.0):
        raise ValueError('first_step 的 dt_phys 須為有限正數')
    return u, dtp


def build_constraints_sp_fs(z0, U_nom, u_prev, cfg, delta, Phi, Gam, gam):
    A, lo, hi, blocks = S.build_constraints_sp(z0, U_nom, u_prev, cfg, delta, Phi, Gam, gam)
    fs = getattr(cfg, 'first_step', None)
    if fs is None:
        return A, lo, hi, blocks
    ua, dtp = _check_first_step(fs)
    n = cfg.N * NU
    rows = np.zeros((NU, n))
    rows[:, :NU] = np.eye(NU)
    b = np.asarray(cfg.amax(), float).reshape(-1) * dtp
    i0 = A.shape[0]
    blocks = dict(blocks)
    blocks['first_step_accept'] = (i0, i0 + NU)
    return (np.vstack([A, rows]), np.concatenate([lo, ua - b]), np.concatenate([hi, ua + b]), blocks)


_g = dict(S.solve_sp.__globals__)
_g['build_constraints_sp'] = build_constraints_sp_fs
solve_sp_fs = types.FunctionType(S.solve_sp.__code__, _g, 'solve_sp_fs', S.solve_sp.__defaults__, S.solve_sp.__closure__)
solve_sp_fs.__kwdefaults__ = S.solve_sp.__kwdefaults__
solve_sp_fs.__doc__ = '與 wgmpc_core_sp.solve_sp 同一份程式碼；約束建構換成 build_constraints_sp_fs（cfg.first_step 啟用時加首筆承接框）。'
