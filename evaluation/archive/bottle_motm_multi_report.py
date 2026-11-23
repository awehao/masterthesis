#!/usr/bin/env python3
"""多任務第一批（v2 量測目標相對排程）單趟報告（只讀紀錄；Howard 20261102 多任務交接「每趟最低證據」）：
1 目標取得：模型第一列有效 xy／第一個就緒候選／採用（時刻、序號、V_HIGH 段、來源年齡）；觀測 xy／支撐高度與實際施加目標分開；真值差只作評估。
  誤檢：候選紀錄只有逐幀的單一 known_type 列，沒有逐實例的誤檢產生／被選中欄位 ⇒ 不填 0，標「未逐實例記錄」。
2 v2 平移：Δx、採用當下底盤 x、平移前後排程。
3 INSERT → 離桌成立（t_lift_ok）：實測底盤速度（位置差分，5 點平滑，同 bottle_mainline_table）、近停（<0.01 m/s 連續 ≥0.1 s）段與總時間、位移。
4 抓起／運送／放置三段分開與第一阻點；滑移：停止觸發值（判定器停止原因字串）與全程最大值（含停止尾段，判定器 slip_max）分開；禁止接觸、穿透、停止後保持另列。
5 固定 0.10 s 定義的 TCP（世界）與底盤加速度峰（到停止／完成為止）；求解牆鐘與命令時序（非實時保證）。
    python3 evaluation/bottle_motm_multi_report.py <RUN_ID> [<RUN_ID> ...]
輸出 results/bottle/motm_multi/<RUN_ID>.json。
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '產生程式'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_mainline_table as M                       # noqa: E402
import bottle_s1_geom as G                              # noqa: E402

OUT = os.path.join(HERE, 'results', 'bottle', 'motm_multi')
DT, W10 = 0.01, 10                                      # 0.10 s 差分窗＝10 步


def runs_below(t, v, thr, min_s=0.1):
    """v<thr 的連續段（起、迄、長），只列 ≥min_s。"""
    seg, k0 = [], None
    for k in range(len(v) + 1):
        lo = k < len(v) and v[k] < thr
        if lo and k0 is None:
            k0 = k
        elif not lo and k0 is not None:
            L = (k - k0) * DT
            if L >= min_s - 1e-9:
                seg.append([round(float(t[k0]), 2), round(float(t[k - 1]), 2), round(L, 2)])
            k0 = None
    return seg


def acc10(P):
    """固定 0.10 s 定義：v_k＝(P[k+10]−P[k])/0.1，a_k＝(v[k+10]−v[k])/0.1；回 |a| 序列（對齊 k）。"""
    V = (P[W10:] - P[:-W10]) / (W10 * DT)
    A = (V[W10:] - V[:-W10]) / (W10 * DT)
    return np.linalg.norm(A, axis=1)


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']
    e, _ = M.metrics(run)
    ev = r['events']
    ph = {}
    for x in ev:
        if x['what'] in ('phase', 'place_phase'):
            ph.setdefault(x['to'], x['t'])
    t_end = e['t_done'] if e['t_done'] is not None else (r['stop'] or {}).get('t', float(t[-1]))
    # 1 目標取得
    sg, link = r['segtarget'], r.get('segtarget_link') or {}
    T0 = float(link.get('T0', 0.0))
    rows, cands, checks = [], [], []
    for ln in open(os.path.join(D, 'candidates.jsonl')):
        j = json.loads(ln)
        {'row': rows, 'cand': cands}.get(j.get('kind'), checks).append(j)
    r_xy = next((j for j in rows if j['row'].get('xy_valid') == 1.0), None)
    c_rd = next((j for j in cands if j.get('c_ready') == 1.0), None)
    fx = sg['fix']
    k0 = int(np.argmin(np.abs(t - fx['t'])))
    ro_ = (r.get('region') or {}).get('offset', [0.0, 0.0])                    # R-B1：s1_steps 為世界座標、控制器目標為區域座標 ⇒ 真值換區域再比
    tb = (float(c['bottle_base_x'][k0]) - float(ro_[0]), float(c['bottle_base_y'][k0]) - float(ro_[1]), float(c['bottle_base_z'][k0]))
    vh = r['vhigh_seq']['log']
    vh_at = next((p for (tt, p, _, _) in reversed(vh) if tt <= fx['t'] + 1e-9), 'TUCK')
    acq = dict(T0=T0, first_valid_xy_row=None if r_xy is None else dict(t_task=round(r_xy['t_recv'] - T0, 3), stamp_task=round(r_xy['row']['stamp'] - T0, 3), age_s=r_xy['row']['recv_age_s']),
               first_c_ready_after_check=None if c_rd is None else dict(t_task=round(c_rd['t_eval'] - T0, 3), age_s=c_rd.get('c_age_s'), n=c_rd.get('c_n')),
               n_rows=len(rows), n_rows_xy_valid=sum(1 for j in rows if j['row'].get('xy_valid') == 1.0), n_cand=len(cands), n_cand_ready=sum(1 for j in cands if j.get('c_ready') == 1.0),
               checks_task_time=[dict(seq=k['seq'], kind=k['kind'], t_first=round(k['t_first'] - T0, 3), t_last=round(k['t_last'] - T0, 3), t_publish=round(k['t_publish'] - T0, 3), n=k['n'],
                                      cx=k['cx'], cy=k['cy'], z_support=k['z_support']) for k in (j.get('check') for j in checks) if k][:8],
               time_note='紀錄的 t_recv／t_eval／check 時刻是 ROS 時間，此處一律減 T0 換成任務時刻；row 列沒有來源相機欄位',
               adopted=dict(t=fx['t'], seq=fx['seq'], vhigh_phase=vh_at, xy=fx['xy'], z_support=fx['z_support'], margin=fx['margin']),
               adopt_log=sg.get('adopt'), applied_final=sg.get('applied_final'), link_age_s=link.get('age_s'), link_rejected=link.get('rejected'), intake_rejected=sg.get('rejected'),
               truth_at_adopt_eval=dict(bottle_bottom=[round(v, 4) for v in tb], err_xy_mm=round(math.hypot(fx['xy'][0] - tb[0], fx['xy'][1] - tb[1]) * 1e3, 3),
                                        err_z_support_mm=round((fx['z_support'] - tb[2]) * 1e3, 3)),
               false_detections='未逐實例記錄（候選列為單一 known_type 列；誤檢產生／被選中無欄位，不填 0）', priors=sg.get('priors'), measured=sg.get('measured'))
    # 2 v2 平移
    rw = r.get('relwin') or {}
    # 3 INSERT → 離桌
    x, y = c['base_x'], c['base_y']
    sp = np.convolve(np.hypot(np.gradient(x, t), np.gradient(y, t)), np.ones(5) / 5, mode='same')
    ti, tok = ph.get('INSERT'), r.get('t_lift_ok')
    def speed_seg(ta, tb):
        if ta is None or tb is None or tb <= ta:
            return None
        mk = (t >= ta) & (t <= tb)
        ii = np.flatnonzero(mk)
        vc = np.hypot(c['base_cmd_vx'][mk], c['base_cmd_vy'][mk])
        return dict(t=[ta, tb], dur_s=round(tb - ta, 3), v_meas=dict(min=round(float(sp[mk].min()), 4), mean=round(float(sp[mk].mean()), 4), max=round(float(sp[mk].max()), 4)),
                    v_cmd=dict(min=round(float(vc.min()), 4), max=round(float(vc.max()), 4)),
                    near_stop_segments=runs_below(t[mk], sp[mk], M.NEAR_STOP), near_stop_total_s=round(float((sp[mk] < M.NEAR_STOP).sum()) * DT, 2),
                    disp_net_m=round(float(math.hypot(x[ii[-1]] - x[ii[0]], y[ii[-1]] - y[ii[0]])), 4), disp_path_m=round(float(np.hypot(np.diff(x[ii]), np.diff(y[ii])).sum()), 4))
    grip = None
    if ti is not None and tok is not None:
        grip = dict(t_insert=ti, t_close=r['t_close'], t_grasp=r['t_grasp'], t_lift_start=r['t_lift_start'], t_lift_ok=tok, t_carry=ph.get('CARRY'),
                    insert_to_grasp=speed_seg(ti, r['t_grasp']), lift_to_carry=speed_seg(r['t_lift_start'], ph.get('CARRY')), insert_to_liftoff=speed_seg(ti, tok),
                    note='抓取速度看 insert_to_grasp；lift_to_carry 含抬升後的加速，不算抓取速度')
    # 4 三段結果
    jd = r.get('success_s0_judge') or {}
    pl = r.get('place') or {}
    stop = r['stop'] or {}
    seg = dict(grasp=dict(ok=r['t_grasp'] is not None, t=r['t_grasp']), lift=dict(ok=tok is not None, t=tok),
               carry=dict(t_start=ph.get('CARRY'), t_carry_nav=ph.get('CARRY_NAV'), t_place_approach=ph.get('PLACE_APPROACH'),
                          compact=[(x_['t'], x_['what']) for x_ in ev if x_['what'].startswith('compact_')]),
               place=dict(phase_final=pl.get('phase'), metrics=pl.get('metrics')), mission_done=bool(r.get('mission_done')), success=bool(r.get('success')),
               first_block=dict(t=stop.get('t'), why=stop.get('why'), phase=next((p for (p, tt) in sorted(ph.items(), key=lambda kv: kv[1], reverse=True) if tt <= (stop.get('t') or 1e9)), None)),
               judge=r.get('result'))
    slip = dict(trigger=stop.get('why') if 'slip' in str(stop.get('why')) else None, t_trigger=stop.get('t') if 'slip' in str(stop.get('why')) else None,
                max_whole_run_mm_deg=r['slip_max_mm_deg'], warn_v2w=(r.get('slip_warn') or {}).get('warn'), version='v2w（平移超標只警告）' if r.get('slip_warn') else 'v2（5 mm 即停）',
                task_complete=bool(r.get('mission_done')), original_grip_quality_ok=(r.get('result') or {}).get('hold_ok'),
                note='全程最大值含停止尾段（判定器 slip_max）；觸發值取判定器停止原因字串；v2w 的第一次警告取判定器 slip_warn')
    safety = dict(forbidden_peak_n=e['forb_peak_n'], forbidden_nodata=e['forb_nodata'], pen_max_mm=r.get('pen_max_mm'), stop_effect_ok=(r.get('stop_effect') or {}).get('ok'),
                  stop_effect=r.get('stop_effect'), clear_bottle_mm=e['clear_bottle_mm'], shield=e['shield'])
    # 停止尾段：物件保持與接觸（自算：逐步欄 FK 的 TCP→瓶相對平移，對 t_grasp；與判定器 slip 定義可能不同，另標）
    tail = None
    if stop.get('t') is not None and r['t_grasp'] is not None:
        q9a = np.column_stack([x, y, c['base_yaw']] + [c[f'q{j}'] for j in range(1, 7)])
        Pb = np.column_stack([c['bottle_base_x'], c['bottle_base_y'], c['bottle_base_z']])

        def rel(k):
            Tt = G.C.K.fk(q9a[k], 'link_tcp')
            return Tt[:3, :3].T @ (Pb[k] - Tt[:3, 3])
        kg, ks = int(np.argmin(np.abs(t - r['t_grasp']))), int(np.argmin(np.abs(t - stop['t'])))
        kt = np.arange(ks, len(t))
        r0 = rel(kg)
        dr = np.array([np.linalg.norm(rel(k) - r0) for k in kt[::5]]) * 1e3
        fk_ = [k for k in c if k.startswith('forb_')]
        tail = dict(t_stop=stop['t'], t_end=round(float(t[-1]), 2), dur_s=round(float(t[-1] - stop['t']), 2),
                    rel_shift_mm=dict(at_stop=round(float(dr[0]), 2), end=round(float(dr[-1]), 2), max=round(float(dr.max()), 2), note='自算 FK（瓶底中心，TCP 座標，對 t_grasp），非判定器 slip'),
                    bottle_z=dict(at_stop=round(float(Pb[ks, 2]), 4), end=round(float(Pb[-1, 2]), 4), min=round(float(Pb[kt, 2].min()), 4)),
                    finger_force_n=dict(f1_min=round(float(np.nanmin(c['f_finger1_bottle'][kt])), 3), f2_min=round(float(np.nanmin(c['f_finger2_bottle'][kt])), 3),
                                        f1_end=round(float(c['f_finger1_bottle'][-1]), 3), f2_end=round(float(c['f_finger2_bottle'][-1]), 3)),
                    bottle_table_force_max_n=round(float(np.nanmax(c['f_bottle_table'][kt])), 3),
                    forbidden_tail_peak_n=round(float(max(np.nanmax(c[k][kt]) for k in fk_)), 4) if fk_ else None,
                    held=bool(float(np.nanmin(c['f_finger1_bottle'][kt])) > 0 and float(np.nanmin(c['f_finger2_bottle'][kt])) > 0 and float(Pb[-1, 2]) > float(Pb[ks, 2]) - 0.01))
    # 5 0.10 s 加速度
    mk = t <= t_end
    q = np.column_stack([c[f'q{j}'] for j in range(1, 7)])
    q9 = np.column_stack([x, y, c['base_yaw'], q])
    P = np.array([G.C.K.fk(qq, 'link_tcp')[:3, 3] for qq in q9[mk]])
    a_tcp = acc10(P)
    a_base = acc10(np.column_stack([x[mk], y[mk]]))
    tt = t[mk][W10:-W10]
    acc = dict(window_s=0.10, t_end=t_end, tcp_peak=round(float(a_tcp.max()), 3), tcp_peak_t=round(float(tt[int(np.argmax(a_tcp))]), 2), tcp_p95=round(float(np.percentile(a_tcp, 95)), 3),
               base_peak=round(float(a_base.max()), 3), base_peak_t=round(float(tt[int(np.argmax(a_base))]), 2), base_p95=round(float(np.percentile(a_base, 95)), 3))
    timing = dict(solve_ms=e['ms'], world_step_ms=e['world_ms'], ros=(r.get('nav_chain') or {}).get('ros'), note='牆鐘時序；模擬非實時，不是實時保證')
    out = dict(run=run, case=(r.get('readback') or {}).get('perturb'), table1_z=r.get('table1_z'), t_done=e['t_done'], acquisition=acq, relwin=rw, insert_to_liftoff=grip,
               segments=seg, slip=slip, safety=safety, post_stop=tail, acc_0p10s=acc, v_peak_run=round(e['v_peak'], 4), timing=timing)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f'{run}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return out


def main():
    for run in sys.argv[1:]:
        o = one(run)
        print(json.dumps({k: o[k] for k in ('run', 'case', 't_done', 'segments', 'slip', 'insert_to_liftoff', 'post_stop', 'acc_0p10s')}, ensure_ascii=False, default=float))
        print(json.dumps(dict(acq=o['acquisition'], relwin={k: o['relwin'].get(k) for k in ('dx', 't', 'x_now')}), ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
