#!/usr/bin/env python3
"""第一桌 0.75 m 的離線核對（Codex 20261102_005118；假世界、假候選，非物理）。用法：python3 evaluation/test_bottle_w1_segtarget_z075.py Z1 [CASE]
只把「第一桌」幾何換成 Z1（假世界瓶底與支撐 TOP、table_boxes() 無參數呼叫＝第一桌、compactcarry2 的 TOP_Z／_TOP）；第二桌（放置）維持 0.60、
手臂備妥姿態沿用 0.60 名目分支（T_READY／Q_READY 不重算）；假候選的 z_support＝Z1（量測），xy＝真值 ± 0.5 mm。生成高度只用於場景與評估。
CASE：nominal（ready 窗 9–11 s）、nocand（從未 ready ⇒ 不閉爪）、dz（量測支撐＝Z1−0.15，模擬以 0.60 當支撐的錯誤 ⇒ 目標跟著量測、不跟真值）。
報：結果分類（未取得目標／不可達／抓取失敗／完整抓放／其他停止）、抓取目標 z、各相位手臂連桿對第一桌（桌面＋桌腳）與瓶的最小距離、自碰、持瓶對兩桌、
設定點速率峰與最大設定點加速度（2 階差分）、關節餘裕、收近閘與過渡（事件）。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
Z1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
CASE = sys.argv[2] if len(sys.argv) > 2 else 'nominal'
import bottle_s1_geom as G                              # noqa: E402
_TB = G.table_boxes
_S060 = dict(G.S1_TABLE)


def table1_boxes(t=None, x0=0.0):
    return _TB(dict(_S060, top_z=Z1) if t is None else t, x0)


G.table_boxes = table1_boxes                            # 無參數呼叫＝第一桌（Z1）
import test_bottle_s0b as S0B                           # noqa: E402
S0B.TOP = Z1                                            # 假世界：第一桌瓶底與支撐
import test_bottle_place as TP                          # noqa: E402
TP.table2_boxes = lambda: [(n, c + np.array([TP.DX, TP.DY, 0.0]), sz) for n, c, sz in _TB(_S060)]   # 第二桌維持 0.60
import bottle_w1_compactcarry2_ctrl as C2               # noqa: E402
C2.TOP_Z, C2._TOP = Z1, table1_boxes()[0]
import bottle_s0b_path_check as PCK                     # noqa: E402
import test_bottle_w1_compactcarry as TC                # noqa: E402
import test_bottle_w1_segtarget as TS                   # noqa: E402

CLASS = dict(segtarget_no_fix='未取得目標', segtarget_unreachable='不可達', grasp_evidence_deadline='抓取失敗')


def main():
    kw = dict(nominal={}, nocand=dict(win=None), dz=dict(dz=-0.15))[CASE]
    c, f, r, st = TS.run(**kw)
    o = TC.summarize(f'Z1 {Z1} {CASE}', c, f, r)
    why = (r['stop'] or {}).get('why')
    cls = '完整抓放' if o['done'] and (o['place'] or {}).get('ok') else CLASS.get(why, f'其他停止 {why}')
    mins, sm, ns, rate, marg = PCK.check(f)
    S = np.array([h['s'] for h in f.hist])
    acc = float(np.abs(np.diff(S, 2, axis=0)).max() / 0.01 ** 2) if len(S) > 3 else None
    out = dict(z1=Z1, case=CASE, outcome=cls, stop=r['stop'], t_done=o['t_done'], grasp_z=st['z_at_freeze'], close_req=c.t_close_req, mins={k: (round(v[0] * 1e3, 1) if isinstance(v, tuple) else v) for k, v in mins.items()},
               self_min_mm=round(sm[0] * 1e3, 2), rate_peak=round(rate, 3), acc_peak=None if acc is None else round(acc, 2), joint_margin=round(marg, 3), held=o['held'], tilt=round(o['tilt'], 2),
               forb=o['forb'], finite=o['finite'], slip=o['slip'], cc=o['cc'], segtarget=dict((k, r['segtarget'][k]) for k in ('fix', 'rejected', 'z_support')))
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', f'segtarget_offline_z{int(round(Z1 * 100)):03d}_{CASE}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('segtarget',)}, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
