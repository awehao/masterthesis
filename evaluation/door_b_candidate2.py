#!/usr/bin/env python3
"""主線 B 候選 2（Codex 20261117_023209：抓持位置／姿態與底盤進入區一起選；維持手腕相機）：
「觀察 → 越帶關節空間過渡到同分支『看得到』構型（後退量 r_v≈0.22–0.31）→ 沿門前線可見追蹤 → 接手前最後快速伸到預抓位（r_v→S_PRE）→ 抓持」。
手腕相機在 r≥~0.23 看得到，所以盲段只剩最後伸手＋抓持排程（目前假設下的估計，不是下限；插入只算在抓持排程內，不重複）。
沿 Smac 實際門前線（door_b_plan.BPlanner；把手＝far_door_9 近距鎖定參考），底盤 V＝0.05 m/s、朝向面門：
  1 越帶落點 a_x（把手後方）× r_v：門旁分支解（由接手預抓位沿 r、a 延拓）——σ、限位、淨空（規劃門檻）、可見；
  2 觀察族構型（a_t0）→ 落點構型：關節空間最小急動，逐點核 σ、淨空、限位、速率／加速度、停止前推（BPlanner.stop_sweep 模型）、記可見；
  3 a_x → a_f 沿分支 IK 追蹤（r＝r_v）：逐點 σ、淨空、可見；
  4 伸手：a_f → a_w，r 由 r_v 線性降到 S_PRE（T_r＝(a_f−a_w)/V），逐點 IK、σ、淨空、關節速率，記最後可見時刻；
  5 盲段估計＝T_r − 最後可見時刻 ＋ 抓持排程 2.05 s ＋ 交接 1 拍。
取盲段最短且全部成立者；全部列出失敗原因。運動學層離線。
    python3 door_b_candidate2.py RUN_DIR PROBE_JSON OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
import door_b_plan as BP
Q_NAV = np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018])
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); AL = np.asarray(ident['alpha'], float)
V, T_READY, V_ARM, A_ARM = 0.05, 2.05, 1.0, 19.984
P = json.load(open(sys.argv[2]))
path = np.array([p[:2] for p in P[0]['plan']], float)
bp = BP.BPlanner(TS.T_H, TS.d, TS.n_, TS.yaw, TS.Kc)
pl = bp.plan(path, Q_NAV)
assert pl['why'] is None, pl['why']
SB, A_W = pl['door_line_sb'], pl['a_w']
q_pre = np.array(pl['q_pre'])
fam = {round(f['a'], 3): np.array(f['q']) for f in pl['obs']}
c_, s_ = math.cos(bp.yaw), math.sin(bp.yaw)
VB = np.array([c_ * (-bp.d[0] * V) + s_ * (-bp.d[1] * V), -s_ * (-bp.d[0] * V) + c_ * (-bp.d[1] * V), 0.0])


def okq(b, q):
    okc, lim, cl = bp.ok_cfg(b, q)
    return okc and bp.sigma6(np.r_[b, q]) >= TS.SIG, cl


def branch_at(a, r, seed):
    q, sm = bp.solve(bp.base(a, SB), bp.target(r), seed)
    if q is None or np.abs(q - seed).max() > 0.3 or sm < TS.SIG:
        return None
    return q


def landing(a_x, r_v):
    """接手預抓位 → 先在 a_w 把 r 升到 r_v → 再沿 a 往回到 a_x（同分支延拓）。"""
    sd = q_pre.copy()
    for r in np.linspace(TS.S_PRE, r_v, 15)[1:]:
        sd = branch_at(A_W, float(r), sd)
        if sd is None:
            return None
    for a in np.linspace(A_W, a_x, 15)[1:]:
        sd = branch_at(float(a), r_v, sd)
        if sd is None:
            return None
    return sd


def track(a0, a1, r0, r1, seed, n=None):
    """a0→a1 沿分支追蹤、r 線性 r0→r1；回 (構型列, 可見列) 或 None。"""
    n = n or max(2, int(round(abs(a0 - a1) / 0.005)))
    Q, vis, sd = [], [], seed.copy()
    for k in range(n + 1):
        u = k / n
        a, r = a0 + (a1 - a0) * u, r0 + (r1 - r0) * u
        q = branch_at(a, r, sd)
        if q is None:
            return None, dict(at=round(a, 3), r=round(r, 3), why='ik_or_sigma')
        ok, cl = okq(bp.base(a, SB), q)
        if not ok:
            return None, dict(at=round(a, 3), r=round(r, 3), why='clear_or_sigma6')
        Q.append(q); vis.append(bool(bp.visible(np.r_[bp.base(a, SB), q]))); sd = q
    return (np.array(Q), vis), None


res, best = [], None
for a_t0 in sorted(fam):
    q_o = fam[a_t0]
    for a_x in (-0.03, -0.04, -0.05):
        for r_v in (0.22, 0.25, 0.28, 0.31):
            rec = dict(a_t0=a_t0, a_x=a_x, r_v=r_v)
            R_LAND = float(os.environ.get('R_LAND', '0'))           # >0：過渡先落在近距（r＝R_LAND），再沿分支退到 r_v 重新看到把手
            q_x = landing(a_x, R_LAND if R_LAND > 0 else r_v)
            if q_x is None:
                rec['why'] = 'landing_ik'; res.append(rec); continue
            if R_LAND <= 0 and not bp.visible(np.r_[bp.base(a_x, SB), q_x]):
                rec['why'] = 'landing_not_visible'; res.append(rec); continue
            # 2 過渡
            T = (a_t0 - a_x) / V
            dq = q_x - q_o
            if float(np.abs(dq).max()) * 1.875 / T > V_ARM:
                rec['why'] = 'transition_rate'; res.append(rec); continue
            bad, vis_tr = None, []
            for k in range(61):
                u = k / 60; sj = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5; sd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
                q = q_o + sj * dq; b = bp.base(a_t0 + (a_x - a_t0) * u, SB)
                ok, cl = okq(b, q)
                if not ok:
                    bad = 'transition_clear_or_sigma@%.2f' % u; break
                okst, _, why = bp.stop_sweep(b, q, VB, sd * dq, AL)
                if not okst:
                    bad = 'transition_stop@%.2f' % u; break
                vis_tr.append(bool(bp.visible(np.r_[b, q])))
            if bad:
                rec['why'] = bad; res.append(rec); continue
            # 4 伸手：找最晚的 a_f（使 r_v→S_PRE 可行且關節速率 ≤V_ARM），先試 T_r 由短到長
            reach_ok = None
            q_x0, a_x0 = q_x, a_x
            if R_LAND > 0:                                            # 退回 r_v：沿分支 a_x→a_x−0.02、r R_LAND→r_v
                trb, fb = track(a_x, a_x - 0.02, R_LAND, r_v, q_x, n=30)
                if trb is None:
                    rec['why'] = 'retreat_' + fb['why']; res.append(rec); continue
                q_x, a_x0 = trb[0][-1], a_x - 0.02
                if not trb[1][-1]:
                    rec['why'] = 'retreat_not_visible'; res.append(rec); continue
            for T_r in (0.6, 0.8, 1.0, 1.4, 2.0):
                a_f = A_W + V * T_r
                if a_f > a_x0 - 1e-9:
                    continue
                tr3, f3 = track(a_x0, a_f, r_v, r_v, q_x)                # 3 可見追蹤
                if tr3 is None:
                    continue
                Q3, vis3 = tr3
                n4 = max(4, int(round(T_r / 0.05)))
                tr4, f4 = track(a_f, A_W, r_v, TS.S_PRE, Q3[-1], n=n4)
                if tr4 is None:
                    continue
                Q4, vis4 = tr4
                rate = float(np.abs(np.diff(Q4, axis=0)).max()) / (T_r / n4)
                if rate > V_ARM:
                    continue
                if not np.allclose(Q4[-1], q_pre, atol=0.05):
                    continue                                          # 伸手終點須回到同一接手預抓位（同分支）
                lv = [k * T_r / n4 for k, v in enumerate(vis4) if v]
                t_lv = max(lv) if lv else 0.0
                reach_ok = dict(T_r=T_r, a_f=round(a_f, 3), track_visible_frac=round(float(np.mean(vis3)), 3), reach_rate=round(rate, 3), t_last_vis_in_reach=round(t_lv, 3),
                                blind_est_s=round(T_r - t_lv + T_READY + 0.05, 3), sigma_min=round(min(bp.sigma6(np.r_[bp.base(A_W, SB), q]) for q in Q4), 4))
                break
            if reach_ok is None:
                rec['why'] = 'reach_or_track'; res.append(rec); continue
            rec.update(ok=True, T_transition=round(T, 2), transition_visible_frac=round(float(np.mean(vis_tr)), 3), transition_last_invisible_u=None if all(vis_tr) else round(max(k for k, v in enumerate(vis_tr) if not v) / 60, 3), **reach_ok)
            res.append(rec)
            if best is None or rec['blind_est_s'] < best['blind_est_s'] or (rec['blind_est_s'] == best['blind_est_s'] and rec['transition_visible_frac'] > best['transition_visible_frac']):
                best = rec
            print('OK', rec, flush=True)
fails = {}
for r in res:
    if not r.get('ok'):
        k = r['why'].split('@')[0]; fails[k] = fails.get(k, 0) + 1
out = dict(note='候選 2：越帶後落在看得到的同分支構型、最後快速伸手；運動學層離線；盲段＝目前假設（V 0.05、抓持排程 2.05 s）下的估計', plan={k: pl[k] for k in ('door_line_sb', 'a_w', 'goal_xy')}, best=best, fails=fails, all=res)
json.dump(out, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=str)
print('n', len(res), 'ok', sum(1 for r in res if r.get('ok')), fails)
print('best', best)
