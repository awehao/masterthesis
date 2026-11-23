#!/usr/bin/env python3
"""wgmpc_core_sp_fr（時域內未來參考）測試（Codex 20261031 瓶子設計審查 §6 列項）：
  R1 單目標回歸：mission_rt_43 四份求解紀錄各取等距 60 輪（含協同 v_ref、PARK_HOLD、PAUSE 等式），solve_sp 與 solve_sp_fr 的 U、J_nl、E_pred 逐位元相同；
  R2 常數序列等價：T 重複成 (N+1,4,4)、v_ref 重複成 (N,3) ⇒ 與單一目標逐位元相同；
  R3 逐步索引：nonlinear_cost_fr、coord_cost_fr 與手算逐步和一致；E_pred[k] 對 T[k]；
  R4 前瞻可追：以已知可行序列 U* 的 rollout FK 當逐步目標。小 R／S（只為驗索引）：逐步 < 1 mm、錯一格明顯較差；
     任務權重：逐步明顯優於「只給末端目標放滿時域」（小權重下末端版 QP 病態失敗，故不在小權重比）；
  R5 末步處理：只改 T[N] ⇒ 成本差＝Qf 加權末項差；
  R6 非法資料：T 形狀 (N,4,4)、含 NaN、v_ref (N+1,3)、v_ref 含 NaN ⇒ ValueError。
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as SP                    # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp_fr as FR                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import k1_resolve_all as RA                                           # noqa: E402

K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
BAD, N_CHK = [], 0


def chk(name, cond, info=''):
    global N_CHK
    N_CHK += 1
    if not cond:
        BAD.append(f'{name} {info}')


def same(a, b):
    return a.ok == b.ok and (not a.ok or (np.array_equal(a.U, b.U) and a.J_nl == b.J_nl and np.array_equal(a.E_pred, b.E_pred)))


def rec_inputs(c0, rec):
    """同 k1_resolve_all.solve_rec 的還原（框倍率 1）；回傳 (z0, u_prev, T, cfg, U_warm)。"""
    c = copy.deepcopy(c0)
    co, si = rec['coord'], rec['solve_in']
    if co is not None:
        for f in ('w_vref', 'w_qn', 'w_a', 'w_p'):
            if co.get(f) is not None:
                setattr(c, f, co[f])
        c.base_vref = None if co.get('base_vref') is None else tuple(co['base_vref'])
        c.arm_q_nom = None if co.get('arm_q_nom') is None else tuple(co['arm_q_nom'])
    c.arm_model.bias = np.asarray(si['arm_bias'], float)
    ph = rec.get('park_hold')
    if isinstance(ph, dict) and ph.get('u_hold') is not None:
        c.base_hold = tuple(float(x) for x in ph['u_hold'])
    elif rec.get('pause_state') in RA.PAUSE_HOLD_STATES or (rec.get('rg') or {}).get('k_free') is False:
        c.base_hold = tuple(float(x) for x in np.asarray(si['U_sol'], float)[0, :3])
    z0 = SP.make_z(np.asarray(si['q_pred'], float), np.asarray(si['s_pred'], float))
    Uw = None if si['U_warm'] is None else np.asarray(si['U_warm'], float)
    return z0, np.asarray(si['u_prev'], float), np.asarray(si['T_cyc'], float).reshape(4, 4), c, Uw


# ---- R1／R2
samples = []
for fn in RA.FILES:
    d = json.load(open(os.path.join(RA.RUN, fn)))
    recs = [x for x in d['log'] if x.get('solve_in')]
    c0 = RA.base_cfg(d['args'])
    for i in np.linspace(0, len(recs) - 1, 60).astype(int):
        samples.append((fn, c0, recs[i]))
n_vref = n_hold = 0
for fn, c0, rec in samples:
    z0, up, T, c, Uw = rec_inputs(c0, rec)
    a = SP.solve_sp(K, z0, up, T, c, U_warm=Uw)
    b = FR.solve_sp_fr(K, z0, up, T, c, U_warm=Uw)
    chk(f'R1 單目標 {fn}@{rec["sim_t"]:.2f}', same(a, b))
    N = c.N
    c2 = copy.deepcopy(c)
    if c2.base_vref is not None:
        c2.base_vref = np.tile(np.asarray(c2.base_vref, float), (N, 1))
        n_vref += 1
    n_hold += c.base_hold is not None
    b2 = FR.solve_sp_fr(K, z0, up, np.tile(T, (N + 1, 1, 1)), c2, U_warm=Uw)
    chk(f'R2 常數序列 {fn}@{rec["sim_t"]:.2f}', same(a, b2))

# ---- R3／R5（取一筆有 v_ref 的紀錄當底）
fn, c0, rec = next(s for s in samples if s[2]['coord'] and s[2]['coord'].get('w_vref', 0) > 0 and s[2].get('park_hold') is None
                   and s[2].get('pause_state') not in RA.PAUSE_HOLD_STATES)
z0, up, T, c, Uw = rec_inputs(c0, rec)
N = c.N
rng = np.random.default_rng(3)
Tseq = np.tile(T, (N + 1, 1, 1)).copy()
for k in range(N + 1):
    Tseq[k, :3, 3] += rng.normal(0, 0.01, 3)
vseq = np.asarray(c.base_vref, float) + rng.normal(0, 0.005, (N, 3))
c3 = copy.deepcopy(c)
c3.base_vref = vseq
U = rng.normal(0, 0.01, (N, SP.NU))
Z = SP.rollout_sp(z0, U, c3)
Qw, Rw, Sw = c3.Q(), c3.R(), c3.S()
J = 0.0
for k in range(1, N + 1):
    e = SP.task_error_sp(K, Z[k], Tseq[k], c3.tcp, c3.pi_band)
    J += float(e @ (Qw * c3.Qf_scale if k == N else Qw) @ e)
for k in range(N):
    J += float(U[k] @ Rw @ U[k])
    dd = U[k] - (up if k == 0 else U[k - 1])
    J += float(dd @ Sw @ dd)
vb = c3.vmax()[:3]
Jc = sum(c3.w_vref * float(((U[k, :3] - vseq[k]) / vb) @ ((U[k, :3] - vseq[k]) / vb)) for k in range(N))
if c3.w_qn > 0:
    Jc += sum(c3.w_qn * float((Z[k, 3:9] - np.asarray(c3.arm_q_nom)) @ (Z[k, 3:9] - np.asarray(c3.arm_q_nom))) for k in range(1, N + 1))
Jfr = FR.nonlinear_cost_fr(K, z0, U, up, FR.target_seq(Tseq, N), c3)
chk('R3 成本＝手算逐步和', abs(Jfr - (J + Jc)) <= 1e-9 * max(1.0, abs(J + Jc)), f'{Jfr} vs {J + Jc}')
r3 = FR.solve_sp_fr(K, z0, up, Tseq, c3, U_warm=Uw)
chk('R3 求解成功', r3.ok, r3.reason)
if r3.ok:
    E = np.array([SP.task_error_sp(K, r3.Z_pred[k], Tseq[k], c3.tcp, c3.pi_band) for k in range(N + 1)])
    chk('R3 E_pred[k] 對 T[k]', np.allclose(E, r3.E_pred, atol=0, rtol=0))
T5 = Tseq.copy()
T5[N, :3, 3] += np.array([0.02, -0.01, 0.015])
eN0 = SP.task_error_sp(K, Z[N], Tseq[N], c3.tcp, c3.pi_band)
eN1 = SP.task_error_sp(K, Z[N], T5[N], c3.tcp, c3.pi_band)
dJ_exp = float(eN1 @ (Qw * c3.Qf_scale) @ eN1) - float(eN0 @ (Qw * c3.Qf_scale) @ eN0)
dJ = FR.nonlinear_cost_fr(K, z0, U, up, T5, c3) - Jfr
chk('R5 只改末步 ⇒ 成本差＝Qf 末項差', abs(dJ - dJ_exp) <= 1e-9 * max(1.0, abs(dJ_exp)), f'{dJ} vs {dJ_exp}')

# ---- R4 前瞻可追（離線構造：U* 為框內平滑序列，目標＝其 rollout 的 FK）
c4 = copy.deepcopy(c)
c4.w_vref, c4.base_vref, c4.w_qn, c4.arm_q_nom = 0.0, None, 0.0, None
c4.base_hold = None
c4w = copy.deepcopy(c4)                                             # 任務權重版（末端 vs 逐步比較用）
c4.w_a = c4.w_bt = c4.w_br = 1e-8                                  # 構造案例：讓追蹤項主導（只為驗索引，不代表任務權重）
c4.w_s = c4.w_s_arm = c4.w_s_base = 1e-8
c4.w_p = 500.0
u0 = np.asarray(up, float)
Ustar = np.array([u0 + (k + 1) * np.r_[0.004, -0.002, 0.01, 0.03, -0.02, 0.02, 0.01, -0.03, 0.02] for k in range(N)])
Ustar = np.clip(Ustar, -c4.vbox() * 0.8, c4.vbox() * 0.8)
Zs = SP.rollout_sp(z0, Ustar, c4)
Tfr = np.array([K.fk(Zs[k, :SP.NQ], c4.tcp) for k in range(N + 1)])
r_fr = FR.solve_sp_fr(K, z0, u0, Tfr, c4)
r_fw = FR.solve_sp_fr(K, z0, u0, Tfr, c4w)                          # 任務權重：逐步目標
r_end = FR.solve_sp_fr(K, z0, u0, Tfr[N], c4w)                      # 任務權重：只給末端目標放滿時域
Tshift = np.concatenate([Tfr[1:], Tfr[N:]], axis=0)               # 錯一格（k 用到 k+1 的目標）
r_sh = FR.solve_sp_fr(K, z0, u0, Tshift, c4)


def track_err(r):
    return float(max(np.linalg.norm(SP.task_error_sp(K, r.Z_pred[k], Tfr[k], c4.tcp, c4.pi_band)[:3]) for k in range(1, N + 1))) if r.ok else float('inf')


ef, ee, es, efw = track_err(r_fr), track_err(r_end), track_err(r_sh), track_err(r_fw)
print('R4 求解狀態：逐步(小權重)', r_fr.ok, '／錯格(小權重)', r_sh.ok, r_sh.reason, '／逐步(任務權重)', r_fw.ok, '／末端(任務權重)', r_end.ok, r_end.reason)
chk('R4 逐步目標可追（各步位置誤差 < 1 mm）', ef < 1e-3, f'{ef * 1e3:.3f} mm')
chk('R4b 任務權重下兩組皆可解', r_fw.ok and r_end.ok, f'{r_fw.reason}／{r_end.reason}')
chk('R4b 任務權重下只給末端較差', ee > 2 * efw, f'末端 {ee * 1e3:.2f} vs 逐步 {efw * 1e3:.2f} mm')
chk('R4 錯一格明顯較差', es > 3 * ef + 1e-4, f'錯格 {es * 1e3:.2f} vs 逐步 {ef * 1e3:.3f} mm')

# ---- R6 非法資料
for name, fnc in (('T (N,4,4)', lambda: FR.solve_sp_fr(K, z0, up, np.tile(T, (N, 1, 1)), c)),
                  ('T 含 NaN', lambda: FR.solve_sp_fr(K, z0, up, np.where(np.eye(4) > 0, np.nan, T), c)),
                  ('v_ref (N+1,3)', lambda: FR.solve_sp_fr(K, z0, up, T, (lambda cc: (setattr(cc, 'base_vref', np.zeros((N + 1, 3))), cc)[1])(copy.deepcopy(c3)))),
                  ('v_ref 含 NaN', lambda: FR.solve_sp_fr(K, z0, up, T, (lambda cc: (setattr(cc, 'base_vref', np.full((N, 3), np.nan)), cc)[1])(copy.deepcopy(c3))))):
    try:
        fnc()
        chk(f'R6 {name} 應拒絕', False)
    except ValueError:
        chk(f'R6 {name} 拒絕', True)

print(f'wgmpc_core_sp_fr 測試 {N_CHK} 項（紀錄樣本 {len(samples)} 輪，其中含 v_ref {n_vref}、底盤等式 {n_hold}；R4 小權重 逐步 {ef * 1e3:.3f}／錯格 {es * 1e3:.2f} mm；任務權重 逐步 {efw * 1e3:.2f}／末端 {ee * 1e3:.2f} mm）：'
      + ('全過' if not BAD else '失敗 ' + '；'.join(BAD[:6])))
sys.exit(0 if not BAD else 1)
