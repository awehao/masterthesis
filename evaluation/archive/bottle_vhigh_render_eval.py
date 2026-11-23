#!/usr/bin/env python3
"""V_HIGH 只渲染核對的離線評估（Codex 20261102_021353 第 4 點；凍結模型＋後處理不變；不啟動 Isaac）。
輸入 runs/<RENDER>/render_w1.npz（isaac_bottle_render_seq.py）與合成來源 s1_report.json 的序列時刻。逐幀照 bottle_seg3d_node 的處理：
  遮罩模型（凍結權重）→ 每個實例 locate（自身濾除＝該幀設定構型、開閉手指點雲聯集，同節點）→ 尺寸閘 → 取信心最高的合格實例 → 一列（同節點欄位意義）。
  列依擷取時刻送 CandidateTracker（同候選節點規則），收到時刻＝擷取＋假設延遲 LAT（實跑旁路中位約 0.08–0.11 s；**假設值，不是本次量到的資料年齡**）。
另以渲染深度做「考慮遮擋」的真值可見性（只評估用）：瓶身側面取樣點、瓶底周圍桌面環（半徑 3–9 cm）取樣點投影到影像，
  該像素深度與點的相機深度差 ≤ 1 cm ⇒ 可見；缺深度（0）另計。另報：預測遮罩內缺深度比例、遮罩截斷旗標、誤檢（合格實例中心距真值 > 30 mm）、影像邊界餘裕（遮罩外框到邊界像素）。
報：逐幀列、連續有效（xy＋支撐）時段、known_type 首次就緒（含快照可達檢查一次，離線同步算）與期限（離開 V_HIGH、漸變起點）比較、對真值誤差、離線處理耗時。
    .venv-dl0/bin/python evaluation/bottle_vhigh_render_eval.py vhigh_render_z075_1
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image as PILImage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_seg_depth3d as S3                         # noqa: E402
import bottle_seg3d_candidates as CT                    # noqa: E402

WEIGHTS = os.path.join(HERE, 'dl0_ckpt', 'bottle_seg_v1_continue3', 'best.pth')
LAT = 0.10
BOTTLE_XY = (0.0, 0.42)


def vis_truth(dep, K, T_wc, tz):
    """考慮遮擋的可見比例：瓶身側面、支撐環。回 dict(body, ring, body_nodepth, ring_nodepth)。"""
    r, h = S3.G.BOTTLE['r'], S3.G.BOTTLE['h']
    a = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    Ti = np.linalg.inv(T_wc)

    def frac(P):
        Pc = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]
        ok = Pc[:, 2] > 0.05
        u = np.floor(np.where(ok, K[0, 0] * Pc[:, 0] / np.where(ok, Pc[:, 2], 1) + K[0, 2], -1)).astype(int)   # 與反投影 (u＋0.5−cx) 一致：像素 u 涵蓋 [u, u＋1)
        v = np.floor(np.where(ok, K[1, 1] * Pc[:, 1] / np.where(ok, Pc[:, 2], 1) + K[1, 2], -1)).astype(int)
        inb = ok & (u >= 0) & (u < dep.shape[1]) & (v >= 0) & (v < dep.shape[0])
        d = np.zeros(len(P))
        d[inb] = dep[v[inb], u[inb]]
        seen = inb & (d > 0) & (np.abs(d - Pc[:, 2]) <= 0.01)
        return float(seen.mean()), float((inb & (d == 0)).mean()), float(inb.mean())
    cam = T_wc[:3, 3]
    body = []
    for z in np.linspace(0.01, h - 0.01, 8):                  # 只取朝相機半圈（背面本來就看不到）
        for t in a:
            n = np.array([math.cos(t), math.sin(t)])
            if np.dot(n, cam[:2] - np.array(BOTTLE_XY)) > 0:
                body.append([BOTTLE_XY[0] + r * n[0], BOTTLE_XY[1] + r * n[1], tz + z])
    ring = [[BOTTLE_XY[0] + rr * math.cos(t), BOTTLE_XY[1] + rr * math.sin(t), tz] for rr in (0.03, 0.06, 0.09) for t in a]
    fb, nb, ib = frac(np.array(body))
    fr, nr, ir = frac(np.array(ring))
    return dict(body=round(fb, 3), ring=round(fr, 3), body_inimg=round(ib, 3), ring_inimg=round(ir, 3), body_nodepth=round(nb, 3), ring_nodepth=round(nr, 3))


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 'render_w1.npz'))
    src = str(Z['src'])
    rep = json.load(open(os.path.join(HERE, 'runs', src, 's1_report.json')))
    tz, info = float(rep['table1_z']), rep['seq_info']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = S3.load_model(WEIGHTS, dev)
    K = np.asarray(Z['K'], float)
    rows, diag = [], []
    for k in range(len(Z['stamp'])):
        st, rgb, dep, T, q9 = float(Z['stamp'][k]), Z['rgb'][k], Z['depth'][k], Z['T_wc'][k], Z['q9'][k]
        w0 = time.perf_counter()
        masks, recs, infer_ms = S3.SI.predict(model, PILImage.fromarray(rgb), dev)
        P = [np.vstack(list(S3.G.link_world(q9, fq).values())) for fq in (S3.G.Q_OPEN, 0.0)]
        robot = cKDTree(np.vstack(P))
        r, conf, n_false, sel_false, cut, nod, bbox_m = dict(valid=False, why='no_instance'), float('nan'), 0, False, {}, float('nan'), float('nan')
        if len(masks):
            order = np.argsort([-x['confidence'] for x in recs])
            ests = [S3.locate(masks[j], dep, K, T, robot) for j in order]
            ok = [i for i, x in enumerate(ests) if x.get('valid') and x['r_silhouette'] <= S3.SIZE_GATE['r_max'] and x['z_visible_span'] <= S3.SIZE_GATE['h_max']]
            for i in ok:
                c_ = ests[i].get('center_known_r')
                if c_ is not None and math.hypot(c_[0] - BOTTLE_XY[0], c_[1] - BOTTLE_XY[1]) > 0.03:
                    n_false += 1
            if ok:
                r, conf = ests[ok[0]], recs[int(order[ok[0]])]['confidence']
                c0 = r.get('center_known_r')
                sel_false = bool(c0 is not None and math.hypot(c0[0] - BOTTLE_XY[0], c0[1] - BOTTLE_XY[1]) > 0.03)
                m = masks[int(order[ok[0]])]
                nod = float((dep[m] == 0).mean()) if m.any() else float('nan')
                ys, xs = np.nonzero(m)
                bbox_m = float(min(xs.min(), m.shape[1] - 1 - xs.max(), ys.min(), m.shape[0] - 1 - ys.max())) if len(xs) else float('nan')
            else:
                r = dict(valid=False, why=ests[0].get('why') if not ests[0].get('valid') else 'size_gate_rejected_all')
        proc_ms = (time.perf_counter() - w0) * 1e3
        v = r if r.get('valid') else {}
        sup = v.get('support', {})
        ck = v.get('center_known_r', [np.nan, np.nan])
        row = dict(stamp=st, recv_age_s=LAT, t_recv=st + LAT, valid=float(bool(r.get('valid'))), xy_valid=float(v.get('xy_valid', False)), cx_known_r=ck[0], cy_known_r=ck[1],
                   top_observed=float(v.get('top_observed', False)), z_top=np.nan if v.get('z_top') is None else v['z_top'], support_observed=float(bool(sup.get('observable', False))),
                   z_support=sup.get('z', np.nan) if sup.get('observable') else np.nan, height_valid=float(v.get('height_valid', False)))
        rows.append(row)
        vt = vis_truth(dep, K, T, tz)
        diag.append(dict(t=round(st, 3), x=round(float(q9[0]), 4), y=round(float(q9[1]), 4), n_inst=len(masks), conf=None if not np.isfinite(conf) else round(conf, 3), why=r.get('why'),
                         xy=bool(row['xy_valid']), sup=bool(row['support_observed']), top=bool(row['top_observed']), sup_why=sup.get('why'),
                         err_xy_mm=None if not row['xy_valid'] else round(math.hypot(ck[0] - BOTTLE_XY[0], ck[1] - BOTTLE_XY[1]) * 1e3, 2),
                         err_sup_mm=None if not row['support_observed'] else round(abs(row['z_support'] - tz) * 1e3, 2),
                         cut=[bool(v.get(c_, False)) for c_ in ('top_cut', 'bottom_cut', 'side_cut')], mask_nodepth=None if not np.isfinite(nod) else round(nod, 3),
                         bbox_margin_px=None if not np.isfinite(bbox_m) else bbox_m, n_false=n_false, sel_false=sel_false, n_pts=v.get('n_pts', 0), vis=vt, proc_ms=round(proc_ms, 1), infer_ms=round(float(infer_ms), 1),
                         dist=round(float(np.linalg.norm(T[:3, 3] - np.array([BOTTLE_XY[0], BOTTLE_XY[1], tz + 0.10]))), 3)))
    def track(sub):
        tr = CT.CandidateTracker()
        first, tl = None, []
        for row in sub:
            tr.add(row)
            ev = tr.evaluate(row['t_recv'])['known_type']
            tl.append((round(row['t_recv'], 3), ev['status'], ev['why']))
            if ev['status'] == 'ready' and first is None:
                first = ev
        if first is None:
            return None, tl
        rc = CT.reach_check(first, np.array(info['q_ready']))
        return dict(t_recv=round(first['t_last'] + LAT, 3), t_last=first['t_last'], n=first['n'], center=first['center'], z_support=first['z_support'],
                    err_xy_mm=round(math.hypot(first['center'][0] - BOTTLE_XY[0], first['center'][1] - BOTTLE_XY[1]) * 1e3, 2), err_sup_mm=round(abs(first['z_support'] - tz) * 1e3, 2),
                    reach=dict(ik_state=rc['ik_state'], local_clear_state=rc['local_clear_state'], local_clear_min=rc['local_clear_min'])), tl
    first_all, tl = track(rows)
    first_5 = {f'phase{ph}': track(rows[ph::2])[0] for ph in (0, 1)}            # 實跑相機 5 Hz：10 Hz 影格取兩種相位各重放一次，兩者都報
    ok_rows = [d for d in diag if d['xy'] and d['sup']]
    segs, cur = [], None
    for d in diag:
        if d['xy'] and d['sup']:
            cur = [d['t'], d['t']] if cur is None else [cur[0], d['t']]
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    in_vh = [d for d in diag if info['t_v0'] <= d['t'] <= info['t_dep']]
    out = dict(run=run, src=src, table1_z=tz, seq=info, lat_assumed_s=LAT, n_frames=len(diag), weights=WEIGHTS,
               valid_xy_sup_segments=[dict(t0=s[0], t1=s[1], dur=round(s[1] - s[0] + 0.1, 2)) for s in segs],
               known_type_first_ready_10hz=first_all, known_type_first_ready_5hz=first_5,
               deadlines=dict(t_depart_vhigh=info['t_dep'], t_x_blend=info['t_blend']),
               vhigh_hold=dict(n=len(in_vh), xy_sup=sum(d['xy'] and d['sup'] for d in in_vh), xy=sum(d['xy'] for d in in_vh), sup=sum(d['sup'] for d in in_vh),
                               no_instance=sum(d['why'] == 'no_instance' for d in in_vh), sup_why={w: sum(d['sup_why'] == w for d in in_vh if not d['sup']) for w in {d['sup_why'] for d in in_vh if not d['sup']}},
                               cut=[sum(d['cut'][i] for d in in_vh) for i in range(3)], false_generated=sum(d['n_false'] for d in in_vh), false_selected=sum(d['sel_false'] for d in in_vh),
                               vis_body=[min((d['vis']['body'] for d in in_vh), default=None), max((d['vis']['body'] for d in in_vh), default=None)],
                               vis_ring=[min((d['vis']['ring'] for d in in_vh), default=None), max((d['vis']['ring'] for d in in_vh), default=None)],
                               bbox_margin_px_min=min((d['bbox_margin_px'] for d in in_vh if d['bbox_margin_px'] is not None), default=None),
                               dist=[min((d['dist'] for d in in_vh), default=None), max((d['dist'] for d in in_vh), default=None)]),
               false_all=dict(generated=sum(d['n_false'] for d in diag), selected=sum(d['sel_false'] for d in diag), frames_generated=[d['t'] for d in diag if d['n_false']]),
               err_xy_mm=None if not ok_rows else [round(float(np.percentile([d['err_xy_mm'] for d in ok_rows], q)), 2) for q in (50, 95, 100)],
               err_sup_mm=None if not ok_rows else [round(float(np.percentile([d['err_sup_mm'] for d in ok_rows], q)), 2) for q in (50, 95, 100)],
               proc_ms=[round(float(np.percentile([d['proc_ms'] for d in diag], q)), 1) for q in (50, 95, 100)], frames=diag, tracker=tl,
               note='只渲染、合成序列；收到時刻＝擷取＋假設延遲 LAT，不是線上資料年齡；處理耗時是本機離線逐幀，不含 ROS 傳輸與最新槽丟格')
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', f'vhigh_render_eval_{run}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('frames', 'tracker', 'seq')}, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
