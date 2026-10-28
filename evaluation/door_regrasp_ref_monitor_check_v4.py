#!/usr/bin/env python3
"""換抓「發布參考」核對 v4（Codex r61 §1–§3；v3 保留）。函式參數化（參考包、門角、偏差、監看設定），供時間化 v6 逐輪共用同一套序列與判定。
序列（同 v3）：mode 3（夾持構型隨門繞鉸鏈剛體轉；B 情境底盤偏移＋IK 保外側 TCP）→ 保持 1 s → mode 4（RefRuntime3 起始銜接）→ 1 s → mode 5（門角修正）→ 1 s → mode 6。
每個序列元素帶來源索引（20 Hz 參考列；保持＝前一元素的索引），供時間化把違規對回所在段。
兩種核對分開（r61 §1：離線放寬年齡上限的結果不得稱為線上時效判準通過）：
  plan  規劃壓力：GeometryMonitor（age_max 10 s、僅取距離與接近率；目標切換走 SwitchAwareGeom）＋**固定** T_plan＝0.29 s 扣減：
        d − max(接近率, v_floor)·0.29 − ds_err < d_min＋warn(1 mm) 計違規。不用監看器本身的 T_pred（含實測計算耗時）。
  online 線上相容：GeometryMonitor 原設定（age_max 0.2 s、t_cycle 0.05、t_stop 0.02），來源年齡 sa 取 bench_6 實測 state_age_wall
        （p99 0.0251 s、最大 0.1448 s；runs/door_coopen_core_bench_6/task.json）；切換重算與本輪計算耗時計入；
        stop／warn 計違規，過期（geometry_result_stale／geometry_state_stale）另計。
情境：N；Q±0.7°；B 小偏移（底盤 2 mm × 偏航 0.2°，8 角點；掃描顯示多數方向 ≤ 2–4 mm 才在 0.05 rad 內）；B 0.8 倍（16 mm／1.6°，預期被起始拒）。
被拒（起始銜接或 IK 無解）、核對通過、違規三者分報。仍只是參考構型核對，不是 GMPC 追蹤通過。

    python3 -u evaluation/door_regrasp_ref_monitor_check_v4.py results/door/door_regrasp_ref_v4.json   # 寫 *_monitor_check_v4.json（不覆寫）
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402
import door_regrasp_policy3 as P3                      # noqa: E402

G = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
VF, DS = G['v_floor_registered'], G['downsampling']['ds_err_registered']
DMIN = {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}
CL = ('nonfinger', 'finger', 'base', 'self')
T_PLAN, WARN, DT, HOLD_S = 0.29, 0.001, 0.05, 1.0
SA_ONLINE = {'p99': 0.0251, 'max': 0.1448}
VLIM = np.array([0.035255, 0.035255, 0.1999] + [0.9999] * 6)
ALIM = np.array([0.5, 0.5, 2.0] + [19.984] * 6)
STALE = ('geometry_result_stale', 'geometry_state_stale')


def rot_cfg(x, a):
    """構型繞鉸鏈軸剛體轉 a（rad）：底盤位置旋轉、偏航加 a、手臂不變。"""
    c, s_ = math.cos(a), math.sin(a)
    y = x.copy()
    y[0], y[1], y[2] = c * x[0] - s_ * x[1], s_ * x[0] + c * x[1], x[2] + a
    return y


def sequence(pkg, kin, off=None, dq_deg=0.0, hold_s=HOLD_S):
    """回傳 (seq, rr) 或 (None, why)。seq 元素 (mode, x9, qf, src_index)。"""
    rr = P3.RefRuntime3(json.loads(json.dumps(pkg)), kin)
    seq = []
    s3 = pkg['segments']['3']
    qa_prev = None
    for i in range(len(s3['t'])):
        xr = rot_cfg(np.r_[s3['base'][i], s3['arm'][i]], -math.radians(dq_deg))
        if off is None:
            seq.append((3, xr, float(s3['qf'][i]), s3['src_index'][i]))
            continue
        b = xr[:3] + off
        q9 = PI.C.ik(b, kin.fk(xr, 'link_tcp'), xr[3:] if qa_prev is None else qa_prev, only_prev=True)
        if q9 is None:
            return None, f'ik_fail_mode3_i{i}'
        qa = np.asarray(q9, float)[PI.C.IDX]
        if qa_prev is not None and float(np.max(np.abs(qa - qa_prev))) > 0.10:
            return None, f'ik_branch_jump_i{i}'
        qa_prev = qa
        seq.append((3, np.r_[b, qa], float(s3['qf'][i]), s3['src_index'][i]))
    nh = int(round(hold_s / DT))

    def hold():
        for _ in range(nh):
            seq.append((seq[-1][0], seq[-1][1].copy(), seq[-1][2], seq[-1][3]))
    hold()
    for k in (4, 5):
        x0 = seq[-1][1]
        ok, why, _ = rr.start(k, x0[:3], x0[3:], pkg['q_ref_deg'] + dq_deg, None)
        if not ok:
            return None, f'mode{k}_start:{why}'
        sg = pkg['segments'][str(k)]
        for i in range(len(sg['t'])):
            seq.append((k, rr.cfg(k, i), float(sg['qf'][i]), sg['src_index'][i]))
        hold()
    s6 = pkg['segments']['6']
    for i in range(len(s6['t'])):
        seq.append((6, seq[-1][1].copy(), float(s6['qf'][i]), s6['src_index'][i]))
    return seq, rr


def scan(pkg, kin, seq, dq_deg, fn, kind='plan', sa=0.2):
    """kind 'plan'（固定 T_plan）或 'online'（原設定 age_max 0.2、來源年齡 sa）。回傳 (summary, bad_src)。"""
    q = math.radians(pkg['q_ref_deg'] + dq_deg)
    pre = P3.P1.Rz4(-math.radians(dq_deg)) @ np.asarray(pkg['pregrasp_T'], float).reshape(4, 4)
    cfg = GM.GeomMonitorConfig(v_floor=dict(VF), ds_err=dict(DS), age_max_s=(10.0 if kind == 'plan' else 0.2))
    mon = GM.GeometryMonitor(cfg, fn)
    gsw = GS.SwitchAwareGeom(mon, fn)
    reached = False
    vpts, vcls, n_stale, first, bad_src = 0, {c: 0 for c in CL}, 0, None, set()
    mins = {c: {'d': math.inf, 'd_pred': math.inf} for c in CL}
    tp_max = 0.0
    X = np.array([e[1] for e in seq])
    for j, (k, x, qf, si) in enumerate(seq):
        if k == 5 and not reached:
            reached = float(np.linalg.norm(kin.fk(x, 'link_tcp')[:3, 3] - pre[:3, 3])) <= 0.005
        tgt = {3: 'out', 4: None, 5: ('in' if reached else None), 6: 'in'}[k]
        dec, det = gsw.step(j * DT + 1e-6, x, q, qf, sa, tgt)
        cls_ = det.get('classes') or {}
        bad = False
        if not cls_:
            bad = True
            n_stale += int(det.get('why') in STALE)
            if first is None:
                first = {'j': j, 'mode': k, 'src': si, 'why': det.get('why'), 'state_age': det.get('state_age'), 'compute_s': det.get('compute_s')}
        else:
            tp_max = max(tp_max, float(det['T_pred_s']))
            for c in CL:
                dc, rate = float(cls_[c]['d']), float(cls_[c]['closing_rate'])
                dp = dc - max(rate, VF[c]) * T_PLAN - DS[c] if kind == 'plan' else float(cls_[c]['d_pred'])
                mins[c]['d'], mins[c]['d_pred'] = min(mins[c]['d'], dc), min(mins[c]['d_pred'], dp)
                if dp < DMIN[c] + WARN:
                    vcls[c] += 1
                    bad = True
                    if first is None:
                        first = {'j': j, 'mode': k, 'src': si, 'class': c, 'd_mm': dc * 1e3, 'd_pred_mm': dp * 1e3, 'rate': rate, 'who': cls_[c]['who']}
        if bad:
            vpts += 1
            bad_src.add(int(si))
    V = np.diff(X, axis=0) / DT
    Vb = np.array([np.r_[math.cos(X[i, 2]) * V[i, 0] + math.sin(X[i, 2]) * V[i, 1], -math.sin(X[i, 2]) * V[i, 0] + math.cos(X[i, 2]) * V[i, 1], V[i, 2:]]
                   for i in range(len(V))])
    A = np.diff(Vb, axis=0) / DT
    return ({'kind': kind, 'sa': sa, 'n': len(seq), 'viol_points': vpts, 'viol_by_class': vcls, 'n_stale': n_stale, 'first': first,
             'min_mm': {c: {'d': mins[c]['d'] * 1e3, 'd_pred': mins[c]['d_pred'] * 1e3} for c in CL}, 'T_pred_max_s': tp_max,
             'switch_events': [{kk: e.get(kk) for kk in ('t', 'switch', 'recompute_s', 'why')} for e in gsw.events],
             'v_ratio_max': float(np.max(np.abs(Vb) / VLIM)), 'a_ratio_max': float(np.max(np.abs(A) / ALIM))}, bad_src)


def yaw_rate(seq, lim):
    """發布序列（含銜接、門角修正、保持）相鄰樣本偏航率；回傳 (最大值 rad/s, 超過 lim 的來源索引集合)。（Codex r62 §1：核真正發布值）"""
    mx, bad = 0.0, set()
    for j in range(1, len(seq)):
        w = abs(math.remainder(float(seq[j][1][2] - seq[j - 1][1][2]), 2 * math.pi)) / DT
        mx = max(mx, w)
        if w > lim + 1e-12:
            bad.add(int(seq[j][3]))
    return mx, bad


def cases():
    out = [('N', None, 0.0), ('Q-0.7', None, -0.7), ('Q+0.7', None, 0.7)]
    for tag, r, yaw in (('Bs', 0.002, 0.2), ('B0.8', 0.8 * 0.02, 0.8 * 2.0)):
        r2 = r / math.sqrt(2)
        for sx, sy, sw in itertools.product((1, -1), (1, -1), (1, -1)):
            out.append((f'{tag}{sx:+d}{sy:+d}{sw:+d}', np.r_[sx * r2, sy * r2, sw * math.radians(yaw)], 0.0))
    return out


def main():
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    import door_regrasp_ref_registry as REG
    src = sys.argv[1]
    pkg = json.load(open(os.path.join(HERE, src)))
    ok, why = REG.validate(pkg)
    if not ok:
        raise SystemExit(f'參考包登錄核驗不過：{why}')
    out_p = os.path.join(HERE, src.replace('.json', '_monitor_check_v4.json'))
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    kin = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    fn = GM3.backend_v3(self_mode='full')
    res = {}
    for name, off, dq in cases():
        seq, rr = sequence(pkg, kin, off, dq)
        if seq is None:
            res[name] = {'status': 'rejected', 'why': rr}
            print(name, '被拒', rr, flush=True)
            continue
        r = {'plan': scan(pkg, kin, seq, dq, fn, 'plan')[0]}
        if not name.startswith('Bs') or name == 'Bs+1+1+1':
            for lab, sa in SA_ONLINE.items():
                r[f'online_{lab}'] = scan(pkg, kin, seq, dq, fn, 'online', sa)[0]
        r['status'] = 'pass' if all(v['viol_points'] == 0 for v in r.values() if isinstance(v, dict)) else 'violation'
        res[name] = r
        print(name, r['status'], {kk: (v['viol_points'], v['n_stale'], round(v['T_pred_max_s'], 3), v['first']) for kk, v in r.items() if isinstance(v, dict)},
              flush=True)
    summ = {s: sorted(k for k, v in res.items() if v['status'] == s) for s in ('pass', 'violation', 'rejected')}
    out = {'schema': 'door_regrasp_ref_monitor_check/4', 'pkg': src, 'T_plan_s': T_PLAN, 'hold_s': HOLD_S, 'sa_online': SA_ONLINE,
           'sa_source': 'runs/door_coopen_core_bench_6/task.json trace[].state_age_wall（n 1766）', 'cases': res, 'summary': summ}
    json.dump(out, open(out_p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
