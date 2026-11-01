#!/usr/bin/env python3
"""視點策略在既有六趟（V0 固定俯視保持段）的離線核對（Codex 20261101_230248 (b)；不讀桌高或佈置編號）。
對各趟保存的手腕影格（保持段）用同一模型與 bottle_seg_depth3d.locate（同節點：信心排序＋類別尺寸閘）重建 schema 3 欄位，
餵給 bottle_seg3d_view_policy 的 V0 判斷，輸出每個佈置的決策與缺項；只有 V0 的資料，後續視點的結果要靠新採集。
期望（依 V0 的實際缺項，非依桌高）：0.60/P1、0.45/P2 接受；0.45/P1 缺底 → V_LOW；0.60/P2 與 0.75 兩趟缺頂 → V_HIGH。
    .venv-dl0/bin/python evaluation/test_bottle_seg3d_view_policy.py
"""
import json
import os
import sys

import numpy as np
from PIL import Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_seg3d_view_policy as VP                   # noqa: E402
import bottle_seg_depth3d as S3                         # noqa: E402
from bottle_seg_train import acquire_lock               # noqa: E402

RUNS = dict(seg3d_col_z045_p1b='V_LOW', seg3d_col_z045_p2b='accept', seg3d_col_z060_p1c='accept', seg3d_col_z060_p2='V_HIGH', seg3d_col_z075_p1b='V_HIGH', seg3d_col_z075_p2b='V_HIGH')


def rows_for(run, model, dev):
    rep = json.load(open(os.path.join(HERE, 'runs', run, 'collect_report.json')))
    h0, h1 = [s for s in rep['sched'] if s[0] == 'hold'][0][1:]
    F = np.load(os.path.join(HERE, 'runs', run, 'w1_frames.npz'))
    L = np.load(os.path.join(HERE, 'runs', run, 'w1_log.npz'))
    lc = {str(k): L['rows'][:, i] for i, k in enumerate(L['cols'])}
    off = float(np.median(lc['stamp'] - lc['task_t']))
    rows = []
    for i, st in enumerate(F['stamp']):
        tt = float(st) - off
        if not (h0 <= tt < h1):
            continue
        masks, recs, _ = S3.SI.predict(model, Image.fromarray(F['rgb'][i]), dev)
        r = dict(stamp=tt, valid=0.0, xy_valid=0.0, top_observed=0.0, support_observed=0.0, height_valid=0.0, bottom_cut=0.0, top_cut=0.0, edge_top_unknown=0.0)
        if len(masks):
            order = np.argsort([-x['confidence'] for x in recs])
            ests = [S3.locate(masks[j], F['depth'][i], F['K'], F['T_wc'][i], None) for j in order]
            ok = [e for e in ests if e.get('valid') and e['r_silhouette'] <= S3.SIZE_GATE['r_max'] and e['z_visible_span'] <= S3.SIZE_GATE['h_max']]
            if ok:
                e = ok[0]
                r.update(valid=1.0, xy_valid=float(e['xy_valid']), top_observed=float(e['top_observed']), support_observed=float(e['support']['observable']),
                         height_valid=float(e['height_valid']), bottom_cut=float(e['bottom_cut']), top_cut=float(e['top_cut']), edge_top_unknown=float(e['edge_top'][2]))
        rows.append(r)
    return rows, h0


def main():
    lock = acquire_lock()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = S3.load_model(os.path.join(HERE, 'dl0_ckpt', 'bottle_seg_v1_continue3', 'best.pth'), dev)
    out, bad = {}, 0
    for run, want in RUNS.items():
        rows, h0 = rows_for(run, model, dev)
        a = VP.assess(VP.fresh(rows, h0))
        act, arg = VP.next_view(['V0'], a)
        got = act if act != 'move' else arg
        ok = got == want
        bad += not ok
        out[run] = dict(n=a.get('n'), complete=a.get('complete'), deficits=a.get('deficits'), known_type_candidate=a.get('known_type_candidate'), action=act, arg=arg, want=want,
                        frac=None if a.get('frac') is None else {k: round(float(v), 2) for k, v in a['frac'].items()})
        print(('  ✓ ' if ok else '  ✗ ') + f"{run}：新鮮列 {a.get('n')}、完整 {a.get('complete')}、缺項 {a.get('deficits')}、已知瓶型候選 {a.get('known_type_candidate')} ⇒ {act} {arg or ''}（期望 {want}）")
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'seg3d_view_policy_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print('全過' if not bad else f'未過 {bad}')
    lock.close()
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
