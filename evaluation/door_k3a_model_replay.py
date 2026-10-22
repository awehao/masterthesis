#!/usr/bin/env python3
"""既有實際設定點條件下的一階模型重播與旋轉殘差分析（純離線、只讀；依 Codex 20261022_131608）。

**不是無負載反事實**：實際設定點來自原本的閉迴路，起始 d̂ 也可能已含接觸負載 ⇒ 殘差不命名為負載、順應或馬達能力不足。
只回答「既有模型能重現多少旋轉趨勢、還剩哪些方向性落差」，不判定物理成因；不調參、不改觀測器、不新增控制介入或模擬。

**模型與初值（固定，不用後續量測重擬）**
  q̂_{k+1} = (I − A) q̂_k + A (s_k + d₀)，A＝diag(α_i)，α 取 results/wgmpc_arm_sp_ident_free4.json 六軸原值（不四捨五入）；
  τ 是 α 與物理步長的等價表示，不另加落後層；辨識檔的 bias 不使用（d₀ 取代偏移項）。
  - 初值 q̂_{s_open}＝OPEN 來源步（task trace 第一列 phase＝OPEN 的 step）的 room q_arm_meas。
  - d₀＝快照時刻 ≤ OPEN 來源步 sim_t 的最後一筆求解輸入 solve_in.offset_d_hat（當時已可取得），之後固定；缺 ⇒ 該趟不重播（NA）。
**設定點索引契約**（沿用辨識檔：(sp_i − act_i) → act_{i+1} − act_i）
  物理步 k 的同一迴圈內：先讀 q_arm_meas（room，world.step 前）、再寫入 arm_sp（evidence，本步寫入 API 的設定點），然後 world.step。
  ⇒ s_k＝evidence[step＝k].arm_sp 作用於物理區間 [k, k＋1)，由 q_k 推到 q_{k+1}。
**連續重播、不逐輪重設**：從 s_open 起逐步推進；任一步缺 room 列、缺 evidence 列、arm_sp 為 None／非有限、或步號不連續 ⇒
  在該處停止並記錄原因，之後不補值、不重設初值續跑。
**比較（隔離手臂的條件比較）**：同一物理步 k，模型與實測都用 room 的實測底盤 base_xyth_k 及門角 door_q_k：
  R_q＝rot(H(q_k)⁻¹·FK(base_k, q̂_k)·G)、R_m＝rot(H(q_k)⁻¹·FK(base_k, q_k)·G)；
  R_c＝rot(H(q_k)⁻¹·T_cyc·G)，T_cyc 取快照時刻 ≤ 本步 sim_t 的最後一筆求解採用目標（零階保持，只作參照）。
  沿用 r2 共同把手座標分解：D_QC＝R_qR_cᵀ、D_MQ＝R_mR_qᵀ、D_MC＝R_mR_cᵀ，逐步閉合 D_MC＝D_MQ·D_QC（存閉合誤差）。
  另報抓取相對轉動本身 rotvec(R_q)、rotvec(R_m)（剛性夾持下機構應有相對轉動＝0），以及關節殘差 q_k − q̂_k。
**窗口（來源物理步）**：主窗＝[s_open, 首次單指 bar < 0.5 N 的物理步)；首次低值步另列；
  [首次低值步＋1, 觸發步) 為後續間歇承載窗另列；觸發步（第一列 phase＝ABORTED 的 step）另列。
  r4 的相鄰區間條件殘差只作參照，不相加成整段偏轉。
r2（Codex 20261022_132631；r1 輸出 door_k3a_model_replay.json 保留，程式見 git 歷史）：
  - d₀ 改為**保守初始化**：OPEN 步 sim_t 之前**已發布**（published 且 publish_sim_t_exact ≤ t_open）的最後一個求解週期的 d̂，
    保存其快照／求解開始／發布時間。r1 的「快照時刻 ≤ OPEN」只表示來源時間，不等於當時已取得；另存為參照 d0_snapshot_rule。
  - 窗口：主窗 [OPEN, first_low)、首次低值一列、間歇窗 (first_low, trigger)、觸發步一列，四者分開。
  - HOLD 參照窗存檔：[HOLD 步＋50, OPEN 步)，同一模型、同一保守 d₀ 規則（以窗起點時刻）、窗起點實測初值；保存逐步與窗末。
    只稱「所選 HOLD 窗末的 x／y 殘差」，不當模型驗證通過；U／X 的此窗含介入斜坡（列出擷取步），不稱靜止設定點。

    python3 evaluation/door_k3a_model_replay.py <out_json> <run_dir> [...]   # 已存在則拒絕覆寫
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
from door_k3a_diag2 import rotvec                     # noqa: E402
from door_k3a_rot_source import decompose            # noqa: E402

FINGERS = KC.FINGERS
BIN = 0.05
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')


def replay(q0, sp_of, d0, alpha, k0, k_end):
    """連續重播 q̂_{k+1}＝(1−α)q̂_k＋α(s_k＋d₀)。sp_of(k) 回傳 s_k 或 (None, 原因)。
    回傳 ({k: q̂_k}, 停止資訊或 None)。遇缺值即停止，不補、不重設。"""
    a = np.asarray(alpha, float)
    d0 = np.asarray(d0, float)
    q = np.asarray(q0, float).copy()
    out = {k0: q.copy()}
    for k in range(k0, k_end):
        s, why = sp_of(k)
        if s is None:
            return out, {'step': k, 'why': why}
        q = (1.0 - a) * q + a * (np.asarray(s, float) + d0)
        out[k + 1] = q.copy()
    return out, None


def d0_at(log, t_open):
    """快照時刻 ≤ t_open 的最後一筆 solve_in.offset_d_hat（當時已可取得）。缺 ⇒ (None, 快照時刻)。"""
    c = [x['solve_in'] for x in log if x.get('solve_in') and x['solve_in']['snap_sim_t_exact'] <= t_open + 1e-9]
    if not c:
        return None, None
    si = max(c, key=lambda s: s['snap_sim_t_exact'])
    return si.get('offset_d_hat'), si['snap_sim_t_exact']


def d0_published(log, t_ref):
    """t_ref 之前已發布的最後一個求解週期的 d̂（保守：發布時刻 ≤ t_ref）。缺 ⇒ d_hat None。"""
    c = [x for x in log if x.get('solve_in') and x.get('published') and x.get('publish_sim_t_exact') is not None
         and x['publish_sim_t_exact'] <= t_ref + 1e-9]
    if not c:
        return {'d_hat': None, 'snap_t': None, 'solve_start_t': None, 'publish_t': None}
    x = max(c, key=lambda x: x['publish_sim_t_exact'])
    si = x['solve_in']
    return {'d_hat': si.get('offset_d_hat'), 'snap_t': si['snap_sim_t_exact'], 'solve_start_t': si.get('solve_start_sim_t_exact'),
            'publish_t': x['publish_sim_t_exact']}


def window_of(k, s_low, s_trig):
    if s_trig is not None and k == s_trig:
        return 'trigger'
    if s_low is None or k < s_low:
        return 'main'
    return 'first_low' if k == s_low else 'intermittent'


def run_one(run, alpha):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    room = {int(s[I['step']]): s for s in r['steps']}
    ev = {}
    for ln in open(os.path.join(run, 'evidence.jsonl')):
        x = json.loads(ln)
        ev[int(x['step'])] = x
    tr = t.get('trace') or []
    s_open = int(next(e['step'] for e in tr if e.get('phase') == 'OPEN'))
    s_trig = next((int(e['step']) for e in tr if e.get('phase') == 'ABORTED'), None)
    t_open = room[s_open][I['sim_t']]
    s_low = next((k for k in sorted(room) if k >= s_open and (s_trig is None or k < s_trig)
                  and min(((room[k][I['leaf_contacts']] or {}).get(f, {}) or {}).get('bar', 0.0) for f in FINGERS) < 0.5), None)
    s_hold = int(next(e['step'] for e in tr if e.get('phase') == 'HOLD'))
    D0 = d0_published(a['log'], t_open)
    d0 = D0['d_hat']
    ref_d, ref_snap = d0_at(a['log'], t_open)
    base = {'run': os.path.basename(os.path.normpath(run)), 'steps': {'HOLD': s_hold, 'OPEN': s_open, 'first_low': s_low, 'trigger': s_trig},
            't_open': t_open, 'd0': d0, 'd0_source': D0, 'd0_snapshot_rule': {'d_hat': ref_d, 'snap_t': ref_snap}}
    if d0 is None:
        return {**base, 'status': 'NA：OPEN 前無已發布的 offset_d_hat，不重播'}
    k_end = s_trig if s_trig is not None else max(room)

    def sp_of(k):
        if k not in room or (k + 1) not in room:
            return None, 'room 列缺'
        x = ev.get(k)
        if x is None:
            return None, 'evidence 列缺'
        sp = x.get('arm_sp')
        if sp is None or not np.all(np.isfinite(np.asarray(sp, float))):
            return None, 'arm_sp 缺或非有限'
        if x.get('t') is not None and abs(float(x['t']) - room[k][I['sim_t']]) > 1e-4:
            return None, 'evidence 與 room 時刻不符'
        return sp, None
    qh, stop = replay(room[s_open][I['q_arm_meas']], sp_of, d0, alpha, s_open, k_end)
    G = np.array(t['G_T_H'])
    snaps = sorted([x['solve_in'] for x in a['log'] if x.get('solve_in') and x['solve_in'].get('T_cyc') is not None],
                   key=lambda s: s['snap_sim_t_exact'])
    snap_t = np.array([s['snap_sim_t_exact'] for s in snaps])
    rows = compare_rows(qh, room, I, G, snaps, snap_t, lambda k: window_of(k, s_low, s_trig))
    # ---- HOLD 參照窗（存檔；不當模型驗證）
    k0h = s_hold + 50
    Dh = d0_published(a['log'], room[k0h][I['sim_t']]) if k0h in room else {'d_hat': None}
    cap = next((e['unload_capture'] for e in t.get('policy_events') or [] if 'unload_capture' in e), None)
    hold = {'step0': k0h, 'step1': s_open, 'd0_source': Dh, 'intervention_capture_step': None if cap is None else cap.get('step'),
            'note': '所選 HOLD 窗末的 x／y 殘差；U／X 此窗含介入斜坡，不稱靜止設定點；不當模型驗證通過。'}
    if Dh['d_hat'] is not None and k0h < s_open:
        qhh, stop_h = replay(room[k0h][I['q_arm_meas']], sp_of, Dh['d_hat'], alpha, k0h, s_open)
        hr = compare_rows(qhh, room, I, G, snaps, snap_t, lambda k: 'hold_ref')
        hold.update({'q0': room[k0h][I['q_arm_meas']], 'replay_stop': stop_h, 'end_row': hr[-1] if hr else None, 'rows': hr})
    else:
        hold['status'] = 'NA'
    main = [x for x in rows if x['window'] == 'main']
    bins = {}
    for x in main:
        kq = round(math.floor(max(x['q_deg'], 0.0) / BIN) * BIN, 3)
        bins.setdefault(f'{kq:.2f}', []).append(x)
    summ = {}
    for kq in sorted(bins):
        L = bins[kq]
        d = {'n': len(L)}
        for key in ('model_rot_deg', 'meas_rot_deg', 'D_MQ_deg', 'D_QC_deg', 'D_MC_deg', 'joint_res_rad'):
            v = [x[key] for x in L if x[key] is not None]
            d[key] = None if not v else np.median(np.array(v), axis=0).tolist()
            d[key + '_n'] = len(v)
        summ[kq] = d
    ce = [x['closure_err'] for x in rows if x['closure_err'] is not None]
    pick = lambda w: next((x for x in rows if x['window'] == w), None)
    inter = [x for x in rows if x['window'] == 'intermittent']
    return {**base, 'status': 'ok', 'replay_stop': stop, 'n_replayed_steps': len(qh) - 1,
            'last_step_covered': max(qh), 'closure_err_max': max(ce) if ce else None,
            'main_window': {'step0': s_open, 'step1': s_low, 'n_rows': len(main), 'last_row': main[-1] if main else None},
            'first_low_row': pick('first_low'),
            'intermittent_window': {'step0_excl': s_low, 'step1_excl': s_trig, 'n_rows': len(inter), 'last_row': inter[-1] if inter else None},
            'trigger_row': pick('trigger'),
            'hold_reference': hold, 'bins_main': summ, 'rows': rows}


def compare_rows(qh, room, I, G, snaps, snap_t, win_of):
    rows = []
    for k in sorted(qh):
        s = room[k]
        ts = s[I['sim_t']]
        H = KC._H_of(s[I['door_q_rad']])
        b = s[I['base_xyth']]
        qm = np.asarray(s[I['q_arm_meas']], float)
        Rq = (np.linalg.inv(H) @ KC._tcp(b, list(qh[k])) @ G)[:3, :3]
        Rm = (np.linalg.inv(H) @ KC._tcp(b, list(qm)) @ G)[:3, :3]
        j = int(np.searchsorted(snap_t, ts + 1e-9, side='right') - 1)
        if j >= 0:
            Rc = (np.linalg.inv(H) @ np.array(snaps[j]['T_cyc']).reshape(4, 4) @ G)[:3, :3]
            dec = decompose(Rc, Rq, Rm)
            dec = {'D_MC_deg': dec['D_MC_deg'], 'D_QC_deg': dec['D_SC_deg'], 'D_MQ_deg': dec['D_MS_deg'], 'closure_err': dec['closure_err']}
        else:
            dec = {'D_MC_deg': None, 'D_QC_deg': None, 'D_MQ_deg': np.degrees(rotvec(Rm @ Rq.T)).tolist(), 'closure_err': None}
        rows.append({'step': k, 't': ts, 'q_deg': math.degrees(s[I['door_q_rad']]), 'window': win_of(k),
                     'model_rot_deg': np.degrees(rotvec(Rq)).tolist(), 'meas_rot_deg': np.degrees(rotvec(Rm)).tolist(),
                     **dec, 'joint_res_rad': (qm - qh[k]).tolist(), 'cmd_snapshot_t': None if j < 0 else float(snap_t[j])})
    return rows


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    ident = json.load(open(IDENT))
    alpha = [float(x) for x in ident['alpha']]
    res = {'version': 2, 'model': {'alpha': alpha, 'phys_dt_s': float(ident['phys_dt_measured_s']), 'tau_s': ident['tau_s'],
                                   'ident_file': os.path.relpath(IDENT, HERE), 'bias_used': False},
           'bin_deg': BIN, 'runs': [run_one(r, alpha) for r in sys.argv[2:]],
           'note': '既有實際設定點條件下的一階模型重播（不是無負載反事實：設定點來自原閉迴路、d₀ 可能已含接觸負載）。'
                   '同一物理步以實測底盤與門角比較模型與實測的抓取相對轉動（隔離手臂的條件比較）；共同把手座標旋轉矩陣分解。'
                   '殘差不命名為負載、順應或馬達能力不足；不判定物理成因。'}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    f = lambda v: 'NA' if v is None else '[' + ', '.join(f'{x:+.3f}' for x in v) + ']'
    for R in res['runs']:
        if R['status'] != 'ok':
            print('==', R['run'], R['status'])
            continue
        D = R['d0_source']
        print(f"== {R['run']} 步 {R['steps']}、重播 {R['n_replayed_steps']} 步、停止 {R['replay_stop']}、閉合誤差最大 {R['closure_err_max']:.2e}、"
              f"d₀ 快照 {D['snap_t']:.3f}／求解 {D['solve_start_t']:.3f}／發布 {D['publish_t']:.3f}（OPEN {R['t_open']:.3f}）、"
              f"窗 main {R['main_window']['n_rows']}、first_low {R['first_low_row']['step'] if R['first_low_row'] else None}、"
              f"intermittent {R['intermittent_window']['n_rows']}、trigger {R['trigger_row']['step'] if R['trigger_row'] else None}")
        h = R['hold_reference']
        if h.get('end_row'):
            print(f"  HOLD 參照 {h['step0']}→{h['step1']}（擷取步 {h['intervention_capture_step']}）窗末 D_MQ {f(h['end_row']['D_MQ_deg'])} "
                  f"joint_res(mrad) {f([x*1e3 for x in h['end_row']['joint_res_rad']])}")
        for kq, d in R['bins_main'].items():
            print(f"  q {kq}° n{d['n']:3d} model {f(d['model_rot_deg'])} meas {f(d['meas_rot_deg'])} D_MQ {f(d['D_MQ_deg'])} D_QC {f(d['D_QC_deg'])}")
        lr = R['main_window']['last_row']
        if lr:
            print(f"  主窗末步 {lr['step']} q {lr['q_deg']:.3f}° model {f(lr['model_rot_deg'])} meas {f(lr['meas_rot_deg'])} D_MQ {f(lr['D_MQ_deg'])} "
                  f"joint_res(mrad) {f([x*1e3 for x in lr['joint_res_rad']])}")


if __name__ == '__main__':
    main()
