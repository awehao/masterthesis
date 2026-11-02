#!/usr/bin/env python3
"""抓取段限速來源表（Codex 20261102_192423；只讀既有紀錄，不跑任何東西）。
兩趟：bottle_rb1_gaze_1（R-B1）、bottle_rb2_gaze_1（R-B2）。座標一律換成區域座標（世界 − 區域偏移）。
分段（事件時刻來自 s1_report events）：
  漸變（blend_start → PREGRASP）、預抓＋插入（PREGRASP → CLOSE）、閉爪到抓持證據（CLOSE → LIFT）、抬升到相位切換（LIFT → CARRY）、
  CARRY 相位切換到離桌確立（CARRY → t_lift_ok；兩個事件分列，不混稱）。
每段：時長、底盤沿車道位移、實測車速（10 物理步滑動平均）平均／最低／最高、參考車速 v_b 平均／最低。
限速來源（逐控制拍，對參考車速 v_b）：
  cap     v_b ≥ v_c − 1e-3（抓取接近速度上限 v_c）
  window  bottle_s0b_ctrl.heuristic_speed 的窗口時間項 (x_we − 0.02 − x)／(t_rem＋need) < v_c，且 v_b 與它差 ≤ 0.01 m/s
          t_rem、need 依該函式同式重算：漸變 t_rem＝(t_pre − t)＋0.3＋t_insert（t_pre 取實際 PREGRASP 時刻）、插入 t_rem＝max(t_insert − τ, 0)＋0.2；
          need＝T_CLOSE＋T_EV＋T_LIFT＋T_LIFT_SETTLE（閉爪中 max(T_CLOSE＋T_EV − τ, 0)＋T_LIFT＋T_LIFT_SETTLE；抬升中 max(T_LIFT − τ, 0)＋T_LIFT_SETTLE）。
          PREGRASP 的 t_rem 用 TCP 位姿誤差，紀錄沒有，該段不判（只 0.05 s）。
  slew    其他（斜率限制中、或在兩者之間過渡）
另列：相位時長是否由固定時間常數決定（t_insert、T_CLOSE＋T_EV、T_LIFT）、離開窗口前剩餘距離（CARRY 切換時 x_we − x）。
    python3 evaluation/bottle_grasp_speed_limits.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0b_ctrl as B                             # noqa: E402

RUNS = ('bottle_rb1_gaze_1', 'bottle_rb2_gaze_1')
OUT = os.path.join(HERE, 'results', 'bottle', 'grasp_speed_limits.json')
T_INSERT = 1.5


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    ro = np.array((r.get('region') or {}).get('offset', [0.0, 0.0]), float)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t, x, y, vb = c['t'], c['base_x'] - ro[0], c['base_y'] - ro[1], c['v_b']
    vm = np.r_[0.0, np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-9)]
    vm = np.convolve(vm, np.ones(10) / 10, mode='same')
    ev = {}
    for e in r['events']:
        if e['what'] == 'phase':
            ev.setdefault(e['to'], float(e['t']))
        elif e['what'] == 'blend_start':
            ev.setdefault('blend', float(e['t']))
    ev['t_grasp'], ev['t_lift_ok'] = float(r['t_grasp']), float(r['t_lift_ok'])
    v_c = float(r['v_c'])
    x_ws, x_we = (float(v) for v in r['window'])
    segs = [('漸變', 'blend', 'PREGRASP'), ('預抓＋插入', 'PREGRASP', 'CLOSE'), ('閉爪到抓持證據', 'CLOSE', 'LIFT'), ('抬升到相位切換', 'LIFT', 'CARRY'),
            ('CARRY 切換到離桌確立', 'CARRY', 't_lift_ok')]

    def vwin(tt, xx):
        if ev['blend'] <= tt < ev['PREGRASP']:
            t_rem, need = (ev['PREGRASP'] - tt) + 0.3 + T_INSERT, B.T_CLOSE + B.T_EV + B.T_LIFT + B.T_LIFT_SETTLE
        elif ev['INSERT'] <= tt < ev['CLOSE']:
            t_rem, need = max(T_INSERT - (tt - ev['INSERT']), 0.0) + 0.2, B.T_CLOSE + B.T_EV + B.T_LIFT + B.T_LIFT_SETTLE
        elif ev['CLOSE'] <= tt < ev['LIFT']:
            t_rem, need = 0.0, max(B.T_CLOSE + B.T_EV - (tt - ev['CLOSE']), 0.0) + B.T_LIFT + B.T_LIFT_SETTLE
        elif ev['LIFT'] <= tt < ev['CARRY']:
            t_rem, need = 0.0, max(B.T_LIFT - (tt - ev['LIFT']), 0.0) + B.T_LIFT_SETTLE
        else:
            return None
        return float(np.clip((x_we - 0.02 - xx) / max(t_rem + need, 1e-3), 0.0, v_c))
    rows = []
    for name, a, b in segs:
        m = (t >= ev[a] - 1e-9) & (t < ev[b] - 1e-9)
        idx = np.flatnonzero(m)
        ctl = idx[::5]
        src = dict(cap=0, window=0, slew=0, undetermined=0)
        for i in ctl:
            w = vwin(float(t[i]), float(x[i]))
            if vb[i] >= v_c - 1e-3:
                src['cap'] += 1
            elif w is None:
                src['undetermined'] += 1
            elif w < v_c - 1e-6 and abs(vb[i] - w) <= 0.01:
                src['window'] += 1
            else:
                src['slew'] += 1
        rows.append(dict(seg=name, t0=round(ev[a], 3), t1=round(ev[b], 3), dur=round(ev[b] - ev[a], 3), dx=round(float(x[idx[-1]] - x[idx[0]]), 4) if len(idx) else None,
                         v_meas=dict(mean=round(float(vm[m].mean()), 4), min=round(float(vm[m].min()), 4), max=round(float(vm[m].max()), 4)) if len(idx) else None,
                         v_ref=dict(mean=round(float(vb[m].mean()), 4), min=round(float(vb[m].min()), 4)) if len(idx) else None,
                         x0=round(float(x[idx[0]]), 4) if len(idx) else None, limit_ticks=src))
    k_carry = int(np.argmin(np.abs(t - ev['CARRY'])))
    k_ins = int(np.argmin(np.abs(t - ev['INSERT'])))
    return dict(run=run, v_c=v_c, window=[x_ws, x_we], window_len=round(x_we - x_ws, 4), events={k: round(v, 3) for k, v in ev.items()},
                insert_to_liftok=round(ev['t_lift_ok'] - ev['INSERT'], 3), insert_to_carry=round(ev['CARRY'] - ev['INSERT'], 3),
                x_at_insert=round(float(x[k_ins]), 4), x_at_carry=round(float(x[k_carry]), 4), window_left_at_carry=round(x_we - float(x[k_carry]), 4),
                fixed_times=dict(t_insert=T_INSERT, T_CLOSE=B.T_CLOSE, T_EV=B.T_EV, T_LIFT=B.T_LIFT, T_LIFT_SETTLE=B.T_LIFT_SETTLE),
                close_to_grasp=round(ev['t_grasp'] - ev['CLOSE'], 3), segs=rows)


def main():
    out = {k: one(k) for k in RUNS}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    for k, o in out.items():
        print('==', k, 'v_c', o['v_c'], '窗', o['window'], f"長 {o['window_len']}", 'INSERT→離桌確立', o['insert_to_liftok'], 'INSERT→CARRY', o['insert_to_carry'],
              'x@INSERT', o['x_at_insert'], 'x@CARRY', o['x_at_carry'], '切換時離窗口末端', o['window_left_at_carry'], '閉爪→抓持成立', o['close_to_grasp'])
        for s in o['segs']:
            print(f"   {s['seg']:10s} {s['t0']:6.2f}–{s['t1']:6.2f} ({s['dur']:4.2f} s) Δx {s['dx']} x0 {s['x0']} 實測 {s['v_meas']} 參考 {s['v_ref']} 限速拍 {s['limit_ticks']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
