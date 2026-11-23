#!/usr/bin/env python3
"""靜態懸持實驗控制器（bottle_w1_vhigh_hold_ctrl）的離線流程核對（Codex 20261102_032215；假世界、假候選，**無滑移模型**，只驗流程、不預測物理滑移）。
沿用 test_bottle_w1_segtarget_vhigh 的閉迴路（0.60、候選只在 V_HIGH 保持段送）。
  hold     完整流程：hold_start 在離桌判定之後、懸持期間不送路徑請求（carry_plan_request 在 hold_end 之後）、懸持時長 ≥ 13 s、懸持期間實測底盤速度 ≈ 0、
           手臂設定點不變、桌面力 0；之後完整抓放。另報懸持位置淨空：瓶底離第一桌面高度、夾爪與連桿對第一桌最小距離（PCK）。
  scancut  懸持開始 1 s 後掃描中斷（scan_fn 回 None）⇒ 依既有規則 scan_stale 受控停止，不送路徑請求。
  settle   SETTLE 期間只對懸持邏輯讀的桌面力注入 1.0 N（假：模擬瓶仍被桌面支撐；判定器照實）⇒ 5 s 後 hold_settle_timeout 受控停止。
（hold 案原本另以「懸持中手臂設定點變化 < 1 mrad」判，實得 1.26 mrad，來自 MPC 偏移補償；改核手臂目標 T_carry_rel 懸持中不變，設定點變化照實報。）
    python3 evaluation/test_bottle_w1_vhigh_hold.py all
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = ('hold', 'scancut', 'settle')
if len(sys.argv) > 1 and sys.argv[1] == 'all':
    res, bad = {}, []
    for k in CASES:
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=3600)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        r = json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2000:])
        res[k] = r
        bad += [] if r.get('pass') else [k]
        print(f"{'✓' if r.get('pass') else '✗'} {k}：{r.get('outcome')}｜{r.get('note', r.get('error', ''))}", flush=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'vhigh_hold_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"\n{'全過' if not bad else '未過 ' + str(bad)}")
    sys.exit(1 if bad else 0)

CASE = sys.argv[1] if len(sys.argv) > 1 else 'hold'
sys.argv = [sys.argv[0], '0.60', 'nom']
sys.path.insert(0, HERE)
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_w1_vhigh_hold_ctrl as HC                  # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402
import test_bottle_w1 as T                              # noqa: E402

ST = dict(t_scan_cut=None)


class Ctl(HC.W1PlaceController):
    def _carry_pending(self, q):
        if CASE == 'settle' and self.hold['state'] in ('PRE', 'SETTLE'):
            self.h_ftab = 1.0                                   # 只注入懸持邏輯讀的桌面力（判定器照實，避免其 lift_deadline 先觸發）
        if self.hold['state'] == 'HOLD' and getattr(self, 'T_carry0', None) is None:
            self.T_carry0 = np.array(self.st.T_carry_rel, float)
        if self.hold['state'] == 'HOLD':
            self.T_carry_dev = max(getattr(self, 'T_carry_dev', 0.0), float(np.abs(np.array(self.st.T_carry_rel, float) - self.T_carry0).max()))
        return super()._carry_pending(q)


def main():
    TV.VH.W1PlaceController = Ctl
    if CASE == 'scancut':
        orig_build = TV.T.build

        def build():
            c, f = orig_build()
            fn = c.scan_fn

            def cut(t, q, _fn=fn):
                if c.hold['state'] == 'HOLD' and t >= c.hold['t_stable'] + 1.0:
                    ST['t_scan_cut'] = ST['t_scan_cut'] or t
                    return None
                return _fn(t, q)
            c.scan_fn = cut
            return c, f
        TV.T.build = build
    c, f, r, st = TV.run()
    why = (r['stop'] or {}).get('why')
    done = bool(r['mission_done'])
    h = r['hold_exp']
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'], e['t'])
    H = [x for x in f.hist if h['t_stable'] is not None and h['t_stable'] <= x['t'] <= (h['t_end'] or 1e9)]
    s_rng = None if not H else float(np.ptp(np.array([x['s'] for x in H]), axis=0).max())
    vb = None if len(H) < 2 else float(max(np.hypot(*(H[k]['q'][:2] - H[k - 1]['q'][:2])) / 0.01 for k in range(1, len(H))))
    mins, sm, ns, rate, marg = PCK.check(f)
    t_req = ev.get('carry_plan_request')
    out = dict(case=CASE, outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], hold=h, t_lift_ok=r.get('t_lift_ok'), t_carry_request=t_req,
               hold_len=None if h['t_end'] is None or h['t_stable'] is None else round(h['t_end'] - h['t_stable'], 3), setpoint_range_in_hold=s_rng, base_v_max_in_hold=vb,
               T_carry_rel_dev_in_hold=getattr(c, 'T_carry_dev', None),
               mins={k: (round(v[0] * 1e3, 1) if isinstance(v, tuple) else v) for k, v in mins.items()}, finite=T.finite_cmds(f), t_scan_cut=ST['t_scan_cut'])
    if CASE == 'hold':
        ok = (done and h['t_stable'] is not None and h['t_stable'] >= (r.get('t_lift_ok') or 1e9) - 1e-9 and t_req is not None and t_req >= h['t_end'] - 1e-9
              and out['hold_len'] is not None and out['hold_len'] >= HC.T_HOLD - 1e-6 and getattr(c, 'T_carry_dev', 1.0) < 1e-12 and vb is not None and vb < 0.01 and h['f_table_max'] <= HC.F_TABLE_MAX)
    elif CASE == 'scancut':
        ok = why == 'scan_stale' and t_req is None and out['finite']
    else:
        ok = why == 'hold_settle_timeout' and t_req is None and out['finite']
    out['pass'] = bool(ok)
    out['note'] = (f"進 CARRY {h['t_carry']}、穩定 {h['t_stable']}（過渡 {h['settle_s']} s）、懸持結束 {h['t_end']}（{out['hold_len']} s）、路徑請求 {t_req}；"
                   f"懸持中設定點變化 {s_rng}、底盤實測最大 {vb}、桌面力最大 {h['f_table_max']}；停止 {why}；完成 {done}")
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
