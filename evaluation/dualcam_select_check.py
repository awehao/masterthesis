#!/usr/bin/env python3
"""雙相機候選交接的既有影格重放核對（Codex 20261111_050821 (c)；Howard 11/11 雙相機指示）。純離線。
輸入：results/vision/seg3d_dualcam_tpa_r1/rows.json——實錄 v1（底盤）／w1（手腕）影格以同一分割權重重放的逐格估計（bottle_seg_depth3d；size-gated 實例）。
  每格轉成候選列：stamp＝擷取時刻、recv_age_s＝RECV_AGE（假設；實錄 w1 收到年齡 0.13–0.42 s）、valid／xy_valid／support_observed／中心（已知半徑擬合）／支撐 z。
  影格只在 1 Hz（前段）與 5 Hz（後段）保存 ⇒ 重放的候選窗長取 3.2 s（實跑為 2.0 s），K_MIN 同 3；結論只到「交接邏輯在這些影格上的行為」，不代表實跑時序。
情境（每趟）：名目；底盤先見（w1 前 8 s 無列）；手腕失效（任務 8–12 s 無 w1 列）；無底盤相機；手腕偏 +3 cm（不一致）；手腕舊影格（收到年齡 0.8 s > 0.5）；第二物件（v1 另插 0.5 m 外的假物件列，5–7 s；SourceTracker 假設單一物件串流，混入列只會讓該來源 std 不過而不就緒——多物件關聯未實作）。
報：導引候選首次就緒（時刻、來源、年齡）、抓取候選首次就緒、交接紀錄、不一致／拒絕／兩物件狀態的次數與首次時刻，以及中心對真值誤差（評估用）。
    python3 evaluation/dualcam_select_check.py OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_dualcam_select as DS  # noqa: E402

RECV_AGE, WIN = 0.2, 3.2


def rows_of(rep_rows, run, cam):
    out = []
    for r in rep_rows:
        if r['run'] != run or r['cam'] != cam or not r['on_table']:
            continue
        e = r.get('est_gated') or {}
        c = e.get('center_known_r') or [np.nan, np.nan]
        sup = e.get('support') if isinstance(e.get('support'), (int, float)) else (e.get('support') or {}).get('z') if isinstance(e.get('support'), dict) else np.nan
        out.append(dict(stamp=float(r['stamp']), task_t=float(r['task_t']), recv_age_s=RECV_AGE, valid=1.0 if e.get('valid') else 0.0, xy_valid=1.0 if e.get('xy_valid') else 0.0,
                        support_observed=1.0 if e.get('support_observed') else 0.0, cx_known_r=float(c[0]), cy_known_r=float(c[1]),
                        z_support=float(sup) if sup is not None and np.isfinite(sup) else np.nan, truth=r['truth']))
    return out


def run_scn(rv, rw, scn):
    rv = [dict(r) for r in rv]
    rw = [dict(r) for r in rw]
    if scn == '手腕失效（任務 8–12 s）':
        rw = [r for r in rw if not 8.0 <= r['task_t'] <= 12.0]
    elif scn == '底盤先見（w1 前 8 s 無列）':
        rw = [r for r in rw if r['task_t'] > 8.0]
    elif scn == '無底盤相機':
        rv = []
    elif scn == '手腕偏 +3 cm（不一致）':
        for r in rw:
            r['cx_known_r'] += 0.03
    elif scn == '手腕舊影格（收到年齡 0.8 s）':
        for r in rw:
            r['recv_age_s'] = 0.8
    elif scn == '第二物件（v1 5–7 s 另一物件）':
        fake = [dict(r, cx_known_r=r['cx_known_r'] + 0.5, stamp=r['stamp'] + 0.05) for r in rv if 5.0 <= r['task_t'] <= 7.0]
        rv = sorted(rv + fake, key=lambda r: r['stamp'])
    sel = DS.DualSelect(DS.SourceTracker('v1', window_s=WIN), DS.SourceTracker('w1', window_s=WIN))
    ev = sorted([('v1', r) for r in rv] + [('w1', r) for r in rw], key=lambda z: z[1]['stamp'])
    states, errs = {}, []
    off = ev[0][1]['stamp'] - ev[0][1]['task_t'] if ev else 0.0
    rejected = None
    for cam, r in ev:
        sel.add(cam, r)
        res = sel.evaluate(r['stamp'])
        for k in ('guide_state', 'grasp_state'):
            st = res[k]
            if st not in ('none', 'kept', 'selected', 'kept_unchecked', 'selected_unchecked'):
                d = states.setdefault(f'{k}:{st}', dict(n=0, t_first=round(r['stamp'] - off, 2)))
                d['n'] += 1
        if res.get('reject') and rejected is None:
            rejected = round(r['stamp'] - off, 2)
        if res['grasp'] is not None:
            tr = np.array(r['truth'])
            errs.append(float(np.hypot(*(np.array(res['grasp']['center']) - tr[:2]))) * 1e3)
    f = {k: (None if v is None else dict(v, t=round(v['t'] - off, 2))) for k, v in sel.first.items()}
    return dict(scenario=scn, first_guide=f['guide'], first_grasp=f['grasp'], handoffs=[dict(h, t=round(h['t'] - off, 2)) for h in sel.handoffs], states=states,
                reject_t=rejected, grasp_err_mm=None if not errs else dict(median=round(float(np.median(errs)), 2), max=round(float(np.max(errs)), 2)),
                rejected_rows={c: sel.src[c].n_rej for c in sel.src})


def main():
    out = sys.argv[1]
    R = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_dualcam_tpa_r1', 'rows.json')))['rows']
    res = []
    for run in ('bottle_motm_tpa_nom_3', 'bottle_motm_tpa_A_1'):
        rv, rw = rows_of(R, run, 'v1'), rows_of(R, run, 'w1')
        for scn in ('名目', '底盤先見（w1 前 8 s 無列）', '手腕失效（任務 8–12 s）', '無底盤相機', '手腕偏 +3 cm（不一致）', '手腕舊影格（收到年齡 0.8 s）', '第二物件（v1 5–7 s 另一物件）'):
            x = run_scn(rv, rw, scn)
            x['run'] = run
            res.append(x)
            print(run[-6:], {k: x[k] for k in ('scenario', 'first_guide', 'first_grasp', 'handoffs', 'states', 'reject_t', 'grasp_err_mm')}, flush=True)
    json.dump(dict(params=dict(recv_age=RECV_AGE, window_replay=WIN, tol_same=DS.TOL_SAME, tol_other=DS.TOL_OTHER, t_incons=DS.T_INCONS), rows=res,
                   note='影格重放（1／5 Hz 保存）＋窗長 3.2 s 調整；任務時間＝stamp−首列偏移；真值只供評估'),
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
