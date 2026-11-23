#!/usr/bin/env python3
"""完整接近候選的完整停止核對（Codex 20261117_021349 §2）：沿 door_b_plan 在 Smac 實際路徑上算出的候選，
每 0.1 s 取樣一點，以 BPlanner.stop_sweep（在途 8 步管線＋底盤／手臂依加速度上限減速＋一階手臂、逐 20 ms 核淨空直到停住）前推。
三段：導航姿態→觀察（底盤沿實際路徑以導航速度 v_nav）、觀察族追蹤（底盤沿門前線，速度由 v_nav 線性降到 V）、觀察→預抓位過渡（V）。運動學層離線。
    python3 door_b_stopcheck.py RUN_DIR PROBE_JSON OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
import door_b_plan as BP
Q_NAV = np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018])
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); AL = np.asarray(ident['alpha'], float)
V_NAV = float(os.environ.get('V_NAV', '0.22'))                      # 原 GMPC v_nominal（launch 讀回 0.22）
P = json.load(open(sys.argv[2]))
path = np.array([p[:2] for p in P[0]['plan']], float)
bp = BP.BPlanner(TS.T_H, TS.d, TS.n_, TS.yaw, TS.Kc)
pl = bp.plan(path, Q_NAV)
sb, a_w, a0, T = pl['door_line_sb'], pl['a_w'], pl['a_t0'], pl['T_tr']
fam = {round(f['a'], 3): np.array(f['q']) for f in pl['obs']}
q_obs0, a_obs0 = fam[max(fam)], max(fam)
q_pre = np.array(pl['q_pre'])
c, s = math.cos(bp.yaw), math.sin(bp.yaw)
body = lambda vw: np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], 0.0])   # noqa: E731
res = {}
# 1 導航姿態→觀察：最小急動 T0，底盤沿實際路徑以 V_NAV 走到觀察起點
T0 = max(float(np.abs(q_obs0 - Q_NAV).max()) * 1.875 / 1.0, 1.0) * 1.05
seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
AD = np.array([bp.to_door(p) for p in path])
i_obs = int(np.argmin(np.abs(AD[:, 0] - a_obs0) + 10 * (np.abs(AD[:, 1] - sb) > 0.03)))
s_obs = seg[i_obs]
fails, worst = [], None
for t in np.arange(0.0, T0 + 1e-9, 0.1):
    u = t / T0; sj = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5; sd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T0
    s_now = s_obs - V_NAV * (T0 - t)
    if s_now < 0:
        continue
    j = min(int(np.searchsorted(seg, s_now)), len(path) - 1)
    tan = path[min(j + 1, len(path) - 1)] - path[max(j - 1, 0)]; tan /= max(np.linalg.norm(tan), 1e-9)
    b = np.r_[np.interp(s_now, seg, path[:, 0]), np.interp(s_now, seg, path[:, 1]), bp.yaw]
    q = Q_NAV + sj * (q_obs0 - Q_NAV)
    ok, k, why = bp.stop_sweep(b, q, body(tan * V_NAV), sd * (q_obs0 - Q_NAV), AL)
    if not ok:
        fails.append(dict(t=round(t, 2), why=why))
res['nav_to_obs'] = dict(T0=round(T0, 2), v_nav=V_NAV, n_fail=len(fails), fails=fails[:5])
# 2 觀察族追蹤（a_obs0 → a0），速度由 V_NAV 降到 V
A_obs = sorted([a for a in fam if a >= a0 - 1e-9], reverse=True)
fails = []
for i, a in enumerate(A_obs[:-1]):
    a_next = A_obs[i + 1]
    v = V_NAV + (bp.V - V_NAV) * (a_obs0 - a) / max(a_obs0 - a0, 1e-9)
    dtt = (a - a_next) / v
    ok, k, why = bp.stop_sweep(bp.base(a, sb), fam[a], body(-bp.d * v), (fam[a_next] - fam[a]) / dtt, AL)
    if not ok:
        fails.append(dict(a=a, why=why))
res['obs_track'] = dict(n=len(A_obs) - 1, n_fail=len(fails), fails=fails[:5])
# 3 觀察→預抓位過渡（V）
fails = []
q_a0 = fam[round(a0, 3)]
for t in np.arange(0.0, T + 1e-9, 0.1):
    u = t / T; sj = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5; sd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
    a = a0 + (a_w - a0) * u
    ok, k, why = bp.stop_sweep(bp.base(a, sb), q_a0 + sj * (q_pre - q_a0), body(-bp.d * bp.V), sd * (q_pre - q_a0), AL)
    if not ok:
        fails.append(dict(t=round(t, 2), a=round(a, 3), why=why))
res['transition'] = dict(T=T, n_fail=len(fails), fails=fails[:5])
out = dict(note='完整停止前推（在途管線＋加速度上限減速＋一階手臂，逐 20 ms 核淨空到停住）；運動學層離線', plan={k: pl[k] for k in ('door_line_sb', 'a_w', 'a_t0', 'T_tr', 'blind_s', 'goal_xy')}, res=res)
json.dump(out, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str)[:1500])
