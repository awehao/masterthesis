#!/usr/bin/env python3
"""W1 同任務 M 移植基線的離線調整（假受控對象＋合成估計，非物理；reviews/20261101_howard_continue_physical_experiments_motm.md §C）。
事前宣告：只調 kp、mu_post 兩個參數，至多 4 次調整，依序第一個通過者凍結；任務參數（t_insert 等）與 G 相同，不改。
起點（不計入 4 次）：舊 S1 M 凍結值 (kp 4.0, mu_post 0.01) ⇒ 本任務 19.2 s clearance_bottle（下一拍預測 4.87 mm）。
調整順序（取值只用既有值：kp 4＝S1 凍結、1＝wg4b B1Params 預設、2＝兩者之間；mu_post 0.01＝S1 凍結、0.03＝B1Params 預設）：
  (2.0, 0.01) → (1.0, 0.01) → (4.0, 0.03) → (1.0, 0.03)
通過：完成且無停止、命令有限；5.5 mm 淨空拒絕線照線上判定（未觸發）；閉爪請求前開爪手指↔瓶最小距離 > 0（舊 M 物理失敗是閉爪前手指碰瓶，
假受控對象沒有手指接觸，只能用幾何近似檢查，照實揭露）。四組都不過 ⇒ 記為「該版本安全拒絕」，不跑物理。
    python3 evaluation/test_bottle_w1_motm.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_motm_ctrl as M                         # noqa: E402
import test_bottle_s0b as T0                            # noqa: E402
import test_bottle_w1 as T                              # noqa: E402

START = (4.0, 0.01)
ORDER = ((2.0, 0.01), (1.0, 0.01), (4.0, 0.03), (1.0, 0.03))


def finger_min_preclose(f, t_close):
    """INSERT 起到閉爪請求前，開爪兩指點雲到真瓶圓柱的最小距離（m）。"""
    b0 = np.array([T0.BX, T0.BY, T0.TOP])
    best = 9.9
    for h in f.hist:
        if h['phase'] not in ('INSERT', 'CLOSE') or (t_close is not None and h['t'] >= t_close):
            continue
        W = G.link_world(h['q'], G.Q_OPEN)
        for n in G.C.FINGERS:
            best = min(best, float(G.cyl_dist(W[n] - b0, 0.0, 0.0, 0.0, G.BOTTLE['h'], G.BOTTLE['r']).min()))
    return best


def one(kp, mu):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = lambda *a, **k: M.W1PlaceController(*a, kp=kp, mu_post=mu, **k)
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    r, _ = T.run(c, f)
    ev = {e['what']: e['t'] for e in r['events']}
    fm = finger_min_preclose(f, ev.get('close_request'))
    cb = r['clear_bottle']
    out = dict(kp=kp, mu_post=mu, done=bool(r['mission_done']), stop=r['stop'], t_done=ev.get('mission_done'), n_m=c.n_m, finite=bool(T.finite_cmds(f)),
               clear_mm=[None if cb[k] is None else round(cb[k] * 1e3, 2) for k in ('meas_min', 'pred_min', 'next_min')], finger_preclose_mm=round(fm * 1e3, 2),
               t_close=ev.get('close_request'), wrist=dict(groups=r['w1']['n_groups'], locked=r['w1']['locked']))
    out['ok'] = bool(out['done'] and out['stop'] is None and out['finite'] and fm > 0.0)
    print(f"   (kp {kp}, mu_post {mu})：完成 {out['done']} 阻點 {out['stop'] and out['stop'].get('why')} DONE {out['t_done']} 淨空 {out['clear_mm']} 閉爪前手指↔瓶 {out['finger_preclose_mm']} mm "
          f"手腕組 {out['wrist']['groups']} ⇒ {'通過' if out['ok'] else '不過'}", flush=True)
    return out


def main():
    res = dict(start=one(*START), tries=[], chosen=None, rule='依序至多 4 次、第一個通過即凍結；參數只用既有值')
    for kp, mu in ORDER:
        o = one(kp, mu)
        res['tries'].append(o)
        if o['ok']:
            res['chosen'] = dict(kp=kp, mu_post=mu)
            break
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_motm_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"\n凍結：{res['chosen']}" if res['chosen'] else '\n四組都不過 ⇒ 該版本安全拒絕，不跑物理')
    return 0 if res['chosen'] else 1


if __name__ == '__main__':
    sys.exit(main())
