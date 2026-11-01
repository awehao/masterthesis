#!/usr/bin/env python3
"""OBSERVE＋依量測選抓法（bottle_w1_segtarget_obs_ctrl）的離線閉迴路核對（Codex 20261102_010259、010831；假世界、假候選，非物理）。
用法：python3 evaluation/test_bottle_w1_segtarget_obs.py Z1 CASE   （每個案例一個程序：第一桌幾何在 import 前依 Z1 改，同 test_bottle_w1_segtarget_z075）
      python3 evaluation/test_bottle_w1_segtarget_obs.py all        （依序跑全部案例，結果寫 results/vision/segtarget_obs_offline.json）
背景工作用 sim 模式（啟動時同步算完、模擬時間 LAT 秒後交付，可重現）；實跑用執行緒。
假候選：known_type 欄位 5 Hz；中心＝真值 ± 0.3 mm、z_support＝Z1＋dz；快照 3 s 換序號（同候選節點 SNAP_MAX_AGE），偏移注入時立即換。
  接近窗（approach，任務時間）內送；或只在手臂到觀察視點後（LOOK／PLAN 狀態）送（模擬手腕相機俯視才看得到支撐）。
案例：
  reg     Z1 0.60、接近窗 9–11 s ⇒ 接近途中背景規劃 ⇒ 移動抓取、漸變前提交、完整抓放（不觸發 OBSERVE）
  look60  Z1 0.60、只在俯視時送 ⇒ OBSERVE ⇒ 全窗可行 ⇒ 移動抓取（停後恢復）、完整抓放
  full    Z1 0.75、只在俯視時送 ⇒ OBSERVE ⇒ 停車抓取（停車策略）⇒ 完整抓放
  nolook  Z1 0.75、從不送 ⇒ observe_no_target（未取得目標，有限等待）
  stale   Z1 0.75、俯視時送但觀測年齡 0.8 s ⇒ 拒收 stale ⇒ observe_no_target
  unreach Z1 0.75、目標 y＋0.5 m ⇒ segtarget_unreachable（不可達）
  wrongz  Z1 0.75、量測支撐 0.60（dz −0.15）⇒ 全窗可行的移動計畫被真值網拒絕 ⇒ segtarget_path_unsafe
  changed Z1 0.75、暫存後新快照偏 7 mm ⇒ segtarget_plan_candidate_changed（規劃期間候選改變）
  expired Z1 0.75、規劃交付延遲 35 s（> hold_max 30 s）⇒ segtarget_plan_expired（規劃期間候選逾期）
  noview  Z1 0.75、視點搜尋回無解 ⇒ observe_view_unreachable
每案核：結果分類、停止原因、不閉爪（失敗案）、命令有限、禁止接觸 0；完成案另核持瓶對兩桌、自碰、各相位手臂對第一桌最小距離、停車誤差、抓取目標 z＝量測支撐＋0.10。
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
CASES = {'reg': (0.60, dict(win=(9.0, 11.0))), 'look60': (0.60, dict(look=True)), 'full': (0.75, dict(look=True)), 'nolook': (0.75, dict()),
         'stale': (0.75, dict(look=True, age=0.8)), 'unreach': (0.75, dict(look=True, dy=0.5)), 'wrongz': (0.75, dict(look=True, dz=-0.15)),
         'changed': (0.75, dict(look=True, jump=0.007)), 'expired': (0.75, dict(look=True, lat_plan=35.0)), 'noview': (0.75, dict(look=True, noview=True))}
WANT = dict(reg='完整抓放', look60='完整抓放', full='完整抓放', nolook='observe_no_target', stale='observe_no_target', unreach='segtarget_unreachable',
            wrongz='segtarget_path_unsafe', changed='segtarget_plan_candidate_changed', expired='segtarget_plan_expired', noview='observe_view_unreachable')
OUT = os.path.join(HERE, 'results', 'vision', 'segtarget_obs_offline.json')

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    res, bad = {}, []
    for k, (z1, _) in CASES.items():
        p = subprocess.run([sys.executable, __file__, str(z1), k], capture_output=True, text=True, timeout=3600)
        line = [l for l in p.stdout.splitlines() if l.startswith('{')]
        r = json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2000:])
        res[k] = r
        ok = r.get('pass') is True
        bad += [] if ok else [k]
        print(f"{'✓' if ok else '✗'} {k}（Z1 {z1}）：{r.get('outcome')}｜停止 {r.get('stop_why')}｜{r.get('note', '')}", flush=True)
        if not ok and 'error' in r:
            print(r['error'])
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"\n{'全過' if not bad else '未過 ' + str(bad)}")
    sys.exit(1 if bad else 0)

Z1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
CASE = sys.argv[2] if len(sys.argv) > 2 else 'full'
import bottle_s1_geom as G                              # noqa: E402
_TB = G.table_boxes
_S060 = dict(G.S1_TABLE)


def table1_boxes(t=None, x0=0.0):
    return _TB(dict(_S060, top_z=Z1) if t is None else t, x0)


G.table_boxes = table1_boxes                            # 無參數呼叫＝第一桌（Z1）
import test_bottle_s0b as S0B                           # noqa: E402
S0B.TOP, S0B.HG = Z1, Z1 + 0.10                          # 假世界：第一桌瓶底與支撐、抓持判定的抓持帶高度
import test_bottle_place as TP                          # noqa: E402
TP.table2_boxes = lambda: [(n, c + np.array([TP.DX, TP.DY, 0.0]), sz) for n, c, sz in _TB(_S060)]
import bottle_w1_compactcarry2_ctrl as C2               # noqa: E402
C2.TOP_Z, C2._TOP = Z1, table1_boxes()[0]
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_w1_segtarget_obs_ctrl as OB               # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_compactcarry as TC                # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

CLASS = dict(segtarget_no_fix='未取得目標', observe_no_target='未取得目標', observe_view_unreachable='未取得目標', segtarget_unreachable='不可達',
             segtarget_path_unsafe='不可達（路徑不安全）', grasp_evidence_deadline='抓取失敗')


def run(win=None, look=False, age=0.05, dz=0.0, dy=0.0, jump=None, lat_plan=None, noview=False, seed=0):
    kw = CASES[CASE][1]
    lat = 2.0

    class Ctl(OB.W1PlaceController):
        def __init__(self, *a, **k):
            super().__init__(*a, bg_mode='sim', bg_latency=lat, **k)
            if noview:
                self.view_fn = lambda *a_: dict(ok=False, why='no_view', n_try=0)

        def _launch(self, name, fn, args, timeout):
            self.bg_latency = lat_plan if (name == 'plan' and lat_plan is not None) else lat
            return super()._launch(name, fn, args, timeout)
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = Ctl
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    rng = np.random.default_rng(seed)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, jumped=False, z_at_freeze=None, n_sent=0)
    step0 = c.step

    def step(t, q, **kw_):
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            in_win = (win is not None and win[0] <= t <= win[1]) or (look and c.os in ('LOOK', 'PLAN'))
            if in_win:
                off = np.zeros(2)
                if jump is not None and c.hold is not None:
                    off = np.array([jump, 0.0])
                xy = np.array([T.T.BX, T.T.BY + dy]) + rng.normal(0, 0.0003, 2) + off
                if st['snap'] is None or t - st['t_seq'] >= 3.0 or (jump is not None and c.hold is not None and not st['jumped']):
                    st['jumped'] = st['jumped'] or (jump is not None and c.hold is not None)
                    st['seq'] += 1
                    st['t_seq'] = t
                    st['snap'] = xy.copy()
                cand = dict(ready=1.0, age_s=age, snap_seq=float(st['seq']), snap_t_last=t - age, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                            snap_z_support=Z1 + dz, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0)
                c.candidate_intake(t, cand)
                st['n_sent'] += 1
        out = step0(t, q, **kw_)
        if c.vfix is not None and st['z_at_freeze'] is None:
            st['z_at_freeze'] = float(c.task.T_grasp[2, 3])
        return out
    c.step = step
    r, _ = T.run(c, f, t_end=140.0)
    return c, f, r, st


def main():
    kw = CASES[CASE][1]
    c, f, r, st = run(**kw)
    o = TC.summarize(f'Z1 {Z1} {CASE}', c, f, r)
    why = (r['stop'] or {}).get('why')
    done = bool(o['done'] and (o['place'] or {}).get('ok'))
    cls = '完整抓放' if done else CLASS.get(why, f'其他停止 {why}')
    mins, sm, ns, rate, marg = PCK.check(f)
    ob = r['observe']
    zg = None if st['z_at_freeze'] is None else round(st['z_at_freeze'], 5)
    z_exp = C0.grasp_target(T.T.BX, T.T.BY, Z1 + kw.get('dz', 0.0) + 0.10)[2, 3]
    pe = (ob.get('park_check') or {}).get('err')
    out = dict(case=CASE, z1=Z1, outcome=cls, stop_why=why, stop=r['stop'], t_done=o['t_done'], mode=ob['mode'], x_park=ob['x_park'], grasp_z=zg, grasp_z_expected=round(float(z_exp), 5),
               close_req=c.t_close_req, n_sent=st['n_sent'], finite=T.finite_cmds(f), forb=o['forb'], held=o['held'], self_min_mm=round(sm[0] * 1e3, 2), joint_margin=round(marg, 3),
               mins={k: (round(v[0] * 1e3, 1) if isinstance(v, tuple) else v) for k, v in mins.items()}, park_err=pe, view=ob['view'], plan=ob['plan'], motion=ob['motion'],
               states=[(e['t'], e.get('state') or e.get('stop') or e.get('note')) for e in ob['log']], jobs=ob['jobs'], expire_s=ob['expire_s'],
               rejected=r['segtarget']['rejected'])
    want = WANT[CASE]
    if want == '完整抓放':
        ok = (done and o['forb'] == 0 and out['finite'] and all(v[0] >= TL.REJ * 1e3 for v in o['held'].values()) and sm[0] > 0
              and zg is not None and abs(zg - z_exp) < 1e-6)
        if CASE == 'full':
            ok = ok and ob['mode'] == 'parked' and pe is not None and abs(pe['dx']) <= 0.02
        if CASE == 'look60':
            ok = ok and ob['mode'] == 'moving' and any(s_ == 'RESUME' for _, s_ in out['states'])
        if CASE == 'reg':
            ok = ok and ob['mode'] == 'moving' and not any(s_ == 'STOPPING' for _, s_ in out['states'])
    else:
        ok = why == want and c.t_close_req is None and out['finite'] and o['forb'] == 0 and not r['mission_done']
    out['pass'] = bool(ok)
    out['note'] = f"mode {ob['mode']} x_park {ob['x_park']} 停車誤差 {pe} 抓取 z {zg}（期望 {round(float(z_exp), 5)}）完成 {o['t_done']}"
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
