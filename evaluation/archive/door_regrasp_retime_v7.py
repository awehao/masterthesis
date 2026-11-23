#!/usr/bin/env python3
"""v7（Codex r62 §1；v6 停在 it 0、紀錄 results/door/door_regrasp_retime_v6_stopped_it0.log，無下游）：v6 之外加**參考偏航率上限**
YAW_REF_MAX＝0.82（受控體偏航增益，名目模型）× 0.10（coord 引導 w_cap）× 0.9（餘裕）＝0.0738 rad/s：
  (1) 時間化時底盤偏航速度上限取 min(VMAX_yaw·比例, 0.0738)；(2) 三門角**實際發布序列**（銜接、門角修正、保持後）逐樣本偏航率 ≤ 0.0738，
  超出的來源樣本所在段與前後各一段 ×0.7（同淨空違規處理）。只改參考生成；引導限幅、權重、受控體增益、保護皆不改。0.0738 是名目模型餘裕，不是實機能力保證。
以下為 v6 說明：
v6（Codex r61 §1–§2；v5 保留）：**一張時間表同時對門角 名目、−0.7°、+0.7° 三情境成立**。每輪：時間化 → 20 Hz 取樣 →
以 door_regrasp_ref_split 於記憶體切段建包（同參考包建置規則）→ 對三門角以 door_regrasp_ref_monitor_check_v4.sequence 產生**發布參考序列**
（mode 3 夾持隨門剛體轉、交界保持 1 s、mode 4 RefRuntime3 起始銜接、mode 5 門角修正）→ check_v4.scan(kind='plan')：
固定 T_plan＝0.29 s 的規劃壓力核對（不用離線放寬年齡後的監看器 T_pred；r61 §1 年齡核對分離）；任一情境違規的來源取樣 ⇒ 所在段與前後各一段 ×0.7。
另每點核 PI.check（名目）。速度比例下限 0.005 沿用 v5（看過 v5 r1 結果後由 0.02 改小，屬參考生成變更，單列揭露；報告實際最小比例）。
線上相容核對（age_max 0.2、bench_6 實測來源年齡）不驅動降速（與速度無關），由 check_v4 對建好的參考包另報。
以下為 v5 說明：
v5（Codex r60 §3–§4；v4 保留）：來源改 replan v8（M 段門角包絡）；預測淨空判定改用**與線上相同**的 door_k3m_geom_monitor.GeometryMonitor
＋door_regrasp_geom_switch.SwitchAwareGeom（目標切換重算前態），不再自算公式：每輪新建監看器（登錄 v_floor／ds_err、d_min、warn 1 mm、t_cycle 0.05、t_stop 0.02、
age_max 放寬為 10 s 僅供離線），每個 20 Hz 取樣以 state_age＝0.2 s 呼叫 ⇒ T_pred＝0.27＋實測計算耗時（對齊有效結果上界）；
判定 'stop' 或 'warn' 皆計違規並降速（＝要求 d_pred ≥ d_min＋warn）。其餘同 v4。以下為 v4 說明：
換抓參考時間化 v4（Codex r54 §2、§4）：依既有預測淨空判定逐段降速，直到 20 Hz 取樣全部滿足（不改監看門檻）。
來源：door_regrasp_replan_v7_phi+30.json 的串接列（抓取 φ＋30°、B1 (0.65,−0.05,270°)）（release_outer＋M＋A＋I＋G），最前補 bench_6 110.0 s 實測初態列（同 retime_v2）。
時間參數化同 door_coopen_pass_close_inner_retime_v3（經過點版 LSPB、有號速度、接點混合、底盤車體座標取偏航區間精確最大、規劃 0.995 倍加速度），
差異：**速度比例逐段可調**（初值 0.5＝v3 自由段），反覆：
  取樣 20 Hz → 每點以 door_k3m_geom_monitor_v3 後端（目標：release_outer 的 W1 段 out、其後 None；A 未到預抓取 None、到後與 I、G in）算四類距離，
  接近率＝相鄰取樣差分（**跨模式交界不重置**），預測 d − max(接近率, v_floor)·T_plan − ds_err，T_plan＝0.29 s（規劃壓力值；有效結果上界 0.27）；
  不足 d_min＋warn 的取樣，所在段與前後各一段的速度比例 ×0.7，重新時間化；另每點核 door_coopen_pass_close_inner_v1.check（分階段規則、關節餘裕、底盤 20 mm）。
  任何段速度比例 < 0.005（r2；原 0.02）仍不足 ⇒ 報阻擋（類別、位置），不改其他參數。
報：違規取樣點數與各類違規數分開；各類最小 d 與最小預測值；總時長；各段速度比例分佈；20 Hz 參考（供參考包 v2）。

    python3 -u evaluation/door_regrasp_retime_v4.py   # 寫 results/door/door_regrasp_retime_v4.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_retime_v3 as RT3   # noqa: E402
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_ref_monitor_check_v4 as CK4       # noqa: E402
import door_regrasp_ref_split as RS                    # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_regrasp_replan_v8_phi+30.json')
REL = os.path.join(HERE, 'results', 'door', 'door_coopen_release_bench6_v4.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_retime_v7.json')
YAW_REF_MAX = 0.82 * 0.10 * 0.9
DQS = (0.0, -0.7, 0.7)
BRIDGE = {'base_m': 0.020, 'base_deg': 2.0, 'arm_rad': 0.05, 'door_deg': 0.7}
C = PI.C
G = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
VF, DS = G['v_floor_registered'], G['downsampling']['ds_err_registered']
DMIN = {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}
WARN, T_PLAN, DT_CTRL = 0.001, 0.29, 0.05
CL = ('nonfinger', 'finger', 'base', 'self')
KIN = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
RATIO_FLOOR = 0.005                  # r2：速度比例下限 0.02 → 0.005（r1 在插入末／閉爪交界剩 1 點：殘餘接近率 0.0033 m/s＋實測 T_pred 0.295 s；近橫桿處再放慢屬合理，r1 結果改名 *_r1_floor002.json）
F0 = (0.001686, 0.001704)


def seg_dt(P, Q, F, k, ratio):
    d = P[k + 1] - P[k]
    bx, by = RT3.body_xy_max(d, float(P[k][2]), float(P[k + 1][2]))
    db = np.abs(d)
    db[0], db[1] = bx, by
    vm = np.asarray(RT3.VMAX, float) * ratio[k]
    vm[2] = min(vm[2], YAW_REF_MAX)
    return max(float(np.max(db / vm)), abs(Q[k + 1] - Q[k]) / (RT3.VQ * RT3.TIGHT), abs(F[k + 1] - F[k]) / RT3.VF, 1e-3)


def plan(P, Q, F, ratio):
    n = len(P)
    dt = np.array([seg_dt(P, Q, F, k, ratio) for k in range(n - 1)])
    for _ in range(5000):
        v = [(P[k + 1] - P[k]) / dt[k] for k in range(n - 1)]
        vq = [(Q[k + 1] - Q[k]) / dt[k] for k in range(n - 1)]
        vext = [np.zeros(9)] + v + [np.zeros(9)]
        tau = np.zeros(n)
        for j in range(n):
            dv = RT3.body(vext[j + 1] - vext[j], float(P[j][2]))
            tau[j] = float(np.max(np.abs(dv) / (RT3.AMAX * RT3.TIGHT)))
        bad = [k for k in range(n - 1) if 0.5 * (tau[k] + tau[k + 1]) > dt[k] * (1 - 1e-9)]
        if not bad:
            return dt, tau, v, vq
        for k in bad:
            dt[k] *= 1.1
    raise RuntimeError('時間化未收斂')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    raw = open(SRC, 'rb').read()
    src = json.loads(raw)
    if not src.get('ok'):
        raise SystemExit('重規劃結果不成立，無串接可時間化')
    rel = json.load(open(REL))['start']
    rows = [{'stage': 'release_outer', 'x': list(rel['base']) + list(rel['qa']), 'q': rel['q_deg'], 'qf': float(np.mean(F0))}] + src['rows']
    P = [np.array(r['x'], float) for r in rows]
    Q = [float(r['q']) for r in rows]
    F = [float(r['qf']) for r in rows]
    stages = [r['stage'] for r in rows]
    for k in range(1, len(P)):
        P[k][2] = P[k - 1][2] + math.remainder(P[k][2] - P[k - 1][2], 2 * math.pi)
    fn = GM3.backend_v3(self_mode='full')
    _rows_rp = src['rows']
    T_pre_exact = np.asarray(C.K.fk([float(v) for v in [r for r in _rows_rp if r['stage'] == 'A'][-1]['x']], 'link_tcp'), float)
    ratio = np.full(len(P) - 1, 0.5)
    it_log = []
    for it in range(60):
        TJ.cool()
        dt, tau, v, vq = plan(P, Q, F, ratio)
        S, tw = RT3.sample(P, Q, F, dt, tau, v, vq, stages)
        R = [[float(x['t']), [float(z) for z in x['x']], float(x['q']), float(x['qf']), x['stage']] for x in S]
        sg_, info = RS.split(R, lambda x: C.K.fk(x, 'link_tcp'))
        pkg = {'dt': DT_CTRL, 'q_ref_deg': float(R[0][2]), 'bridge': BRIDGE, 'segments': RS.pack(R, sg_, info),
               'door_bridge_check': {'dq_ok_deg': [-0.7, 0.0, 0.7]}, 'pregrasp_T': T_pre_exact.reshape(-1).tolist()}
        viol_pts, viol_cls, bad_segs, first = 0, {}, set(), None
        mins, per_dq = {}, {}
        for dq in DQS:
            seq, rr = CK4.sequence(pkg, KIN, None, dq)
            if seq is None:
                per_dq[dq] = {'rejected': rr}
                viol_pts += 1
                first = first or {'dq': dq, 'rejected': rr}
                continue
            r, bad_src = CK4.scan(pkg, KIN, seq, dq, fn, 'plan')
            ymax, bad_y = CK4.yaw_rate(seq, YAW_REF_MAX)
            r['yaw_rate_max'], r['n_yaw_bad'] = ymax, len(bad_y)
            viol_cls[f'{dq:+.1f}/yaw_rate'] = len(bad_y)
            viol_pts += len(bad_y)
            bad_src = set(bad_src) | bad_y
            per_dq[dq] = {kk: r[kk] for kk in ('viol_points', 'viol_by_class', 'first', 'min_mm', 'v_ratio_max', 'a_ratio_max', 'yaw_rate_max', 'n_yaw_bad')}
            viol_pts += r['viol_points']
            for c, v in r['viol_by_class'].items():
                viol_cls[f'{dq:+.1f}/{c}'] = v
            if r['first'] is not None and first is None:
                first = dict(r['first'], dq=dq)
            for si in bad_src:
                sk = S[si]['seg']
                for kk in (sk - 1, sk, sk + 1):
                    if 0 <= kk < len(ratio):
                        bad_segs.add(kk)
            for c in CK4.CL:
                m = mins.setdefault(c, {'d': math.inf, 'd_pred': math.inf})
                m['d'], m['d_pred'] = min(m['d'], r['min_mm'][c]['d']), min(m['d_pred'], r['min_mm'][c]['d_pred'])
        pi_fail = []
        for x_ in S:
            ok_pi, why_pi, _ = PI.check(np.asarray(x_['x'], float), x_['q'], x_['qf'], 'release_outer_c' if x_['stage'] == 'release_outer' else x_['stage'])
            if not ok_pi:
                pi_fail.append({'t': x_['t'], 'stage': x_['stage'], 'why': why_pi})
        it_log.append({'it': it, 't_end': S[-1]['t'], 'viol_points': viol_pts, 'viol_by_class': viol_cls, 'first': first, 'n_pi_fail': len(pi_fail),
                       'ratio_min': float(ratio.min()), 'per_dq': {f'{k:+.1f}': v for k, v in per_dq.items()}})
        print(f"it {it}：總時長 {S[-1]['t']:.1f} s、違規點 {viol_pts}（{viol_cls}）、PI 失敗 {len(pi_fail)}、比例最小 {ratio.min():.3f}；首 {first}", flush=True)
        if viol_pts == 0 and not pi_fail:
            break
        if pi_fail:
            break
        for kk in bad_segs:
            ratio[kk] *= 0.7
        if ratio.min() < RATIO_FLOOR:
            break
    ok = viol_pts == 0 and not pi_fail
    out = {'schema': 'door_regrasp_retime/7', 'yaw_ref_max': YAW_REF_MAX, 'yaw_ref_max_basis': '0.82 增益×0.10 w_cap×0.9（名目模型餘裕）', 'dq_deg': list(DQS), 'per_dq_final': {f'{k:+.1f}': v for k, v in per_dq.items()},
           'ratio_floor': RATIO_FLOOR, 'ratio_min_actual': float(ratio.min()), 'ratio_floor_note': '0.02→0.005 為看過 v5 r1 結果後的參考生成變更', 'source': os.path.relpath(SRC, HERE), 'ok': ok, 'T_plan_s': T_PLAN, 'warn_m': WARN,
           'iterations': it_log, 'pi_fail_head': pi_fail[:20], 'min_m': mins, 't_end_s': S[-1]['t'], 'n_samples': len(S),
           'ratio_hist': np.histogram(ratio, bins=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.51])[0].tolist(),
           'stage_t_start': {st: next((float(tw[i - 1]) for i, s in enumerate(stages) if s == st and i > 0), None) for st in ('M', 'A', 'I', 'G')},
           'reference_20hz': {'cols': ['t', 'x9', 'q_door_deg', 'q_finger', 'stage'],
                              'rows': [[s['t'], [float(z) for z in s['x']], s['q'], s['qf'], s['stage']] for s in S]},
           'wall_s': time.time() - t0w}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('ok', 't_end_s', 'min_m', 'ratio_hist')}, ensure_ascii=False, default=float), flush=True)


if __name__ == '__main__':
    main()
