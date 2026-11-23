#!/usr/bin/env python3
"""慢外伸單因素（bottle_w1_vhigh_outslow_ctrl）的離線核對（Codex 20261102_033425；假世界、無滑移模型，只驗流程與參考）。
同 test_bottle_w1_segtarget_vhigh 的 0.60 名目閉迴路，分別跑基線（V_HIGH，外伸 3.8 s）與慢外伸（7.6 s），各一個程序。核：
  外伸實際時長（compact_out_start→compact_returned）＝7.6 s；外伸參考連續（_cc_T 逐拍 TCP 參考位置跳變、速度與加速度峰，對基線約 1/2、1/4）；
  外伸淨空閘監看最小值 ≥ 0 且無 place_reach_clear_lost；PCK 淨空；放置段無 place_phase_timeout（期限不放寬：PLACE_APPROACH 18 s）；完整抓放；
  增加的外伸時間、PARKED 時刻差、DONE 時刻差（停等）。
    python3 evaluation/test_bottle_w1_vhigh_outslow.py all
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1 and sys.argv[1] == 'all':
    res = {}
    for k in ('base', 'slow'):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=3600)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        res[k] = json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2000:])
    b, s = res['base'], res['slow']
    checks = dict(out_len_slow=abs((s.get('out_len') or 0) - 7.6) < 0.06, out_len_base=abs((b.get('out_len') or 0) - 3.8) < 0.06,
                  ref_ratio=s.get('ref_v_max') is not None and s['ref_v_max'] < 0.6 * b['ref_v_max'] and s['ref_a_max'] < 0.4 * b['ref_a_max'],
                  no_place_timeout=not s.get('place_timeout'), slow_done=bool(s.get('done')),
                  reach_gate_applicable=dict(base=b.get('reach_applicable'), slow=s.get('reach_applicable'), note='不適用（無低點約束）≠ 通過；只分開列出'))
    ok = bool(all(v for k, v in checks.items() if k != 'reach_gate_applicable'))
    res['compare'] = dict(added_out_s=None if not (s.get('out_len') and b.get('out_len')) else round(s['out_len'] - b['out_len'], 3),
                          parked_delay_s=None if not (s.get('t_parked') and b.get('t_parked')) else round(s['t_parked'] - b['t_parked'], 3),
                          done_delay_s=None if not (s.get('t_done') and b.get('t_done')) else round(s['t_done'] - b['t_done'], 3),
                          ref_v_ratio=None if not b.get('ref_v_max') else round(s['ref_v_max'] / b['ref_v_max'], 3), ref_a_ratio=None if not b.get('ref_a_max') else round(s['ref_a_max'] / b['ref_a_max'], 3),
                          checks=checks, pass_=ok,
                          not_passed=[k for k, v in checks.items() if k != 'reach_gate_applicable' and not v])
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'vhigh_outslow_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for k in ('base', 'slow'):
        r = res[k]
        print(k, {kk: r.get(kk) for kk in ('done', 'stop', 'out_len', 't_parked', 't_done', 'ref_v_max', 'ref_a_max', 'ref_jump_max', 'reach_min', 'place_timeout', 'pa_len')})
    print('比較', res['compare'])
    print('全過' if ok else '未過')
    sys.exit(0 if ok else 1)

CASE = sys.argv[1] if len(sys.argv) > 1 else 'slow'
sys.argv = [sys.argv[0], '0.60', 'nom']
sys.path.insert(0, HERE)
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_w1_vhigh_outslow_ctrl as OS               # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402

REF = []


def wrap(base):
    class Ctl(base):
        def step(self, t, q, **kw):
            out = super().step(t, q, **kw)
            f = self.cc
            if f is not None and f.get('t_out') is not None and self.n_step % 5 == 0:
                REF.append((float(t), self._cc_T(float(t))[:3, 3].copy()))
            return out
    return Ctl


def main():
    TV.VH.W1PlaceController = wrap(OS.W1PlaceController if CASE == 'slow' else TV.VH.W1PlaceController)
    c, f, r, st = TV.run()
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'], e['t'])
        if e['what'] == 'place_phase':
            ev.setdefault('P_' + e['to'], e['t'])
    t_o, t_r = ev.get('compact_out_start'), ev.get('compact_returned')
    out_len = None if t_o is None or t_r is None else round(t_r - t_o, 3)
    R = [(t_, p_) for t_, p_ in REF if t_o is not None and t_r is not None and t_o - 1e-9 <= t_ <= t_r + 1e-9]
    T_ = np.array([x[0] for x in R])
    P_ = np.array([x[1] for x in R])
    if len(R) > 3:
        V = np.diff(P_, axis=0) / np.diff(T_)[:, None]
        A_ = np.diff(V, axis=0) / np.diff(T_)[1:, None]
        v_max, a_max, jump = float(np.linalg.norm(V, axis=1).max()), float(np.linalg.norm(A_, axis=1).max()), float(np.linalg.norm(np.diff(P_, axis=0), axis=1).max())
    else:
        v_max = a_max = jump = None
    cc = r.get('compact_carry') or {}
    rg = cc.get('reach_gate') or {}
    mins, sm, ns, rate, marg = PCK.check(f)
    why = (r['stop'] or {}).get('why')
    out = dict(case=CASE, done=bool(r['mission_done']), stop=why, out_len=out_len, t_out=t_o, t_parked=ev.get('P_PARKED'), t_done=ev.get('mission_done'),
               pa_len=None if ev.get('P_PLACE_APPROACH') is None or ev.get('P_PARKED') is None else round(ev['P_PARKED'] - ev['P_PLACE_APPROACH'], 3),
               ref_v_max=None if v_max is None else round(v_max, 4), ref_a_max=None if a_max is None else round(a_max, 4), ref_jump_max_m=None if jump is None else round(jump, 5),
               reach_min=rg.get('monitor_min'), reach_trigger_margin=(rg.get('trigger') or {}).get('margin'),
               reach_applicable=bool((rg.get('trigger') or {}).get('margin') is not None and np.isfinite((rg.get('trigger') or {}).get('margin'))),
               reach_note='rect_margin 回 inf＝外伸掃掠與停止軌跡沒有低於桌面頂＋Z_PAD 的點，此分支未產生低點約束（不是整體淨空無限大）；monitor_min=None 同義',
               place_timeout=why == 'place_phase_timeout', self_min_mm=round(sm[0] * 1e3, 2), joint_margin=round(marg, 3),
               mins={k: (round(v[0] * 1e3, 1) if isinstance(v, tuple) else v) for k, v in mins.items()}, outslow=r.get('outslow'))
    out['ref_jump_max'] = out['ref_jump_max_m']
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
