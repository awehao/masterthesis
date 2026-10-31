#!/usr/bin/env python3
"""現行參考包離線監看「最小 d_pred」更正紀錄（Codex 20261031_152320 (b) 核可；不實跑、不改包、不改舊工具）。
舊工具以公尺值比較已存的 mm 值，包內 offline.min 為末列值、不可引用。本腳本：
  1. 在 door_regrasp_geom_switch.SwitchAwareGeom.step 外包一層 MinTracker（公尺比較），記每個監看實例的真最小、步數、stop／warn；
  2. 以原工具、原預設參數把參考包重產到暫存處（每支工具各一個子程序），內容須與現行包相同（證明監看到的序列與當初一致）。
     d_pred 含**實測計算時間**（T＝週期＋狀態年齡＋計算＋停止），包內 offline 欄位寫的是四捨五入後的值，重跑可差一位 ⇒ 雜湊可能不同
     （首版同程序連跑時 v3 即不同，原因未逐一證明）；故判定為「雜湊相同」或「除 offline 欄位外內容完全相同」。真最小值重跑差約 0.02 mm；
  3. S8（door_ref_overlap_v1 只在檔案不存在時寫出）改為重算 S8 前綴列、逐列比對 door_return_ref_v2 後，以同參數重跑監看。
輸出 results/door/ref_offline_min_correction_20261031.json。stop／warn 判定獨立於此錯誤；本紀錄不擴稱其他安全核對已驗證。
    python3 evaluation/ref_offline_min_fix.py
"""
import contextlib
import hashlib
import subprocess
import io
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_regrasp_geom_switch as GS                  # noqa: E402
from geom_min_track import MinTracker                  # noqa: E402

DOOR = os.path.join(HERE, 'results', 'door')
OUT = os.path.join(DOOR, 'ref_offline_min_correction_20261031.json')
TRK = []
_orig = GS.SwitchAwareGeom.step


def _step(self, *a, **kw):
    dec, det = _orig(self, *a, **kw)
    tr = getattr(self, '_min_trk', None)
    if tr is None:
        tr = self._min_trk = {'t': MinTracker(), 'n': 0, 'stop': 0, 'warn': 0}
        TRK.append(tr)
    tr['t'].update(det, tr['n'])
    tr['n'] += 1
    tr['stop'] += dec == 'stop'
    tr['warn'] += dec == 'warn'
    return dec, det


GS.SwitchAwareGeom.step = _step


def sha16(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def via_tool(mod_name, pack, meta_key, tmp):
    mod = __import__(mod_name)
    out = os.path.join(tmp, pack)
    n0 = len(TRK)
    old = sys.argv
    sys.argv = [mod_name, '--out', out]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.main()
    finally:
        sys.argv = old
    dep = os.path.join(DOOR, pack)
    P = json.load(open(dep))
    meta = P.get(meta_key) or {}
    Q = json.load(open(out))
    same_ex_offline = all((P[k] if k != meta_key else {kk: vv for kk, vv in P[k].items() if kk != 'offline'})
                          == (Q.get(k) if k != meta_key else {kk: vv for kk, vv in (Q.get(k) or {}).items() if kk != 'offline'}) for k in P) and set(P) == set(Q)
    return {'pack': pack, 'tool': mod_name + '.py', 'deployed_sha16': sha16(dep), 'reproduced_sha16': sha16(out),
            'sha_equal': sha16(out) == sha16(dep), 'equal_except_offline': same_ex_offline, 'reproduced': sha16(out) == sha16(dep) or same_ex_offline,
            'monitors': [{'n_steps': t['n'], 'stop': t['stop'], 'warn': t['warn'], 'true_min_mm': t['t'].as_mm()} for t in TRK[n0:]],
            'old_offline_min_not_citable': (meta.get('offline') or {}).get('min')}


def s8_case():
    import door_ref_overlap_v1 as OV
    R1 = json.load(open(os.path.join(DOOR, 'door_return_ref_v1.json')))
    A = R1['A']
    i8 = [i for i, r in enumerate(A) if r['stage'].startswith('S8')]
    ie = i8[0] + 800
    X = np.array([r['x'] for r in A])
    v_e = float(np.hypot(*(X[ie + 1, :2] - X[ie, :2]) / OV.DT))
    w_e = float((X[ie + 1, 2] - X[ie, 2]) / OV.DT)
    new, T, vc, wc = OV.retime(X[i8[0]:ie + 1, :2], (float(X[i8[0], 2]), float(X[ie, 2])), 0.0, v_e, 0.0, w_e)
    armA = X[i8[0], 3:]
    rows = np.c_[new, np.tile(armA, (len(new), 1))]
    out = {}
    for pk in ('door_return_ref_v2.json', 'door_return_ref_v3.json'):
        P = json.load(open(os.path.join(DOOR, pk)))
        j8 = [i for i, r in enumerate(P['A']) if r['stage'].startswith('S8')][0]
        Xp = np.array([r['x'] for r in P['A'][j8:j8 + len(rows)]])
        out[pk] = {'rows_equal': bool(Xp.shape == rows.shape and np.array_equal(Xp, rows)), 'end_row': P['s8_overlap']['end_row']}
    n0 = len(TRK)
    with contextlib.redirect_stdout(io.StringIO()):
        OV.monitor(rows, 59.9, 'S8 前綴')
    t = TRK[n0]
    return {'pack': 'door_return_ref_v2.json／v3.json（A 段 S8 前綴）', 'tool': 'door_ref_overlap_v1.py（S8 部分，同參數重跑監看）', 'rows_check': out,
            'monitors': [{'n_steps': t['n'], 'stop': t['stop'], 'warn': t['warn'], 'true_min_mm': t['t'].as_mm()}],
            'old_offline_min_not_citable': '工具只印出、未寫入包'}


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '--one':                     # 子程序：只跑一支工具，結果以 JSON 印在最後一行
        r = via_tool(*sys.argv[2:5], tempfile.mkdtemp())
        print('\n@@' + json.dumps(r, ensure_ascii=False))
        return 0
    res = []
    if True:
        for mod, pack, key in (('door_ref_r1_yaw3', 'door_regrasp_ref_pb60m.json', 'r1_yaw3'),
                               ('door_ref_r3_retime2', 'door_regrasp_ref_pb60n.json', 'r3_retime2'),
                               ('door_ref_seg7_retime', 'door_regrasp_ref_pb60o.json', 'seg7_retime'),
                               ('door_return_ref_s4_retime', 'door_return_ref_v3.json', 's4_retime')):
            pr = subprocess.run([sys.executable, os.path.abspath(__file__), '--one', mod, pack, key], capture_output=True, text=True, check=True)
            r = json.loads(pr.stdout.rsplit('\n@@', 1)[1])
            res.append(r)
            print(f"{pack}（{mod}）重產 {'相同' if r['reproduced'] else '**不同**'}（雜湊相同 {r['sha_equal']}、除 offline 外相同 {r['equal_except_offline']}）（{r['reproduced_sha16']}／{r['deployed_sha16']}）", flush=True)
            for m in r['monitors']:
                print(f"   監看 {m['n_steps']} 步 stop {m['stop']} warn {m['warn']}；真最小 {m['true_min_mm']}")
            print(f"   包內舊值（末列、不可引用）{r['old_offline_min_not_citable']}")
    r = s8_case()
    res.append(r)
    print(f"S8 前綴：列比對 {r['rows_check']}")
    for m in r['monitors']:
        print(f"   監看 {m['n_steps']} 步 stop {m['stop']} warn {m['warn']}；真最小 {m['true_min_mm']}")
    doc = {'date': '2026-10-31', 'why': '舊工具以公尺值比較已存 mm 值，offline.min 為末列值；本檔為更正（原包、原雜湊、舊工具不改）',
           'scope': 'stop／warn 計數獨立判定、不受影響；不因此擴稱其他安全核對已驗證', 'test': 'evaluation/test_geom_min_track.py',
           'items': res}
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    json.dump(doc, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(OUT, os.path.dirname(HERE)))
    return 0 if all(x.get('reproduced', True) for x in res) and all(v['rows_equal'] for v in res[-1]['rows_check'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
