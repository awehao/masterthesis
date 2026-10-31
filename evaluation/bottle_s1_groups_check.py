#!/usr/bin/env python3
"""P、M 對照組最小驗收：各一次停止／切換路徑，各一次名目離線完整任務。假受控對象，非物理。
    python3 evaluation/bottle_s1_groups_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_groups as GR                           # noqa: E402
import test_bottle_s0b as T                             # noqa: E402


def fake(c, t_end, fault=None):
    f = T.Fake(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
    return f.run(t_end, fault or (lambda t, s: (s.c._trigger_stop('sim_limit') if t >= t_end - 3.0 else None) or {}))


def summary(name, c, f):
    r = c.report()
    q = np.array([h['q'][:3] for h in f.hist])
    t = np.array([h['t'] for h in f.hist])
    sp = np.convolve(np.hypot(np.gradient(q[:, 0], t), np.gradient(q[:, 1], t)), np.ones(5) / 5, mode='same')
    mk = (t >= (r['t_close'] or 1e9)) & (t <= (r['t_lift_start'] or -1.0))
    ms = np.array(r['solve_ms_all'] or [0.0])
    cb = r['clear_bottle']
    ev = [(round(e['t'], 1), e.get('to', e['what'])) for e in r['events'] if e['what'] in ('phase', 'unfold_start', 'blend_start', 'exit_enter', 'mission_done', 'm_fail')]
    print(f"[{name}] 成功 {r['success']} 碼 {S1.exit_code(r)} 阻點 {r['stop']}\n  事件 {ev}\n  總 {t[-1]:.1f} s；抓取期速度 {sp[mk].min() if mk.any() else float('nan'):.4f}–{sp[mk].max() if mk.any() else float('nan'):.4f}、近停 {float((sp[mk] < 0.01).sum()) * 0.01 if mk.any() else float('nan'):.2f}／{float(mk.sum()) * 0.01:.2f} s；末 ({q[-1, 0]:.3f}, {q[-1, 1]:.3f})；瓶 實測 {(cb['meas_min'] or float('nan')) * 1e3:.2f} 預測 {(cb['pred_min'] or float('nan')) * 1e3:.2f} mm；滑移 {r['slip_max_mm_deg'][0]:.3f}；停止有效 {r['stop_effect']}\n  求解 ms p50 {np.median(ms):.1f} p95 {np.percentile(ms, 95):.1f} max {ms.max():.1f} >50 {r['n_over_50ms']}；{r['qp_counts']}；Shield {r['nav_chain']['shield']['active']}；管外 {r['n_gate']}", flush=True)
    return r


def main():
    c = SC.make_lidar(cls=GR.S1Parked, log_cbf=False)
    f = fake(c, 50.0)
    rP = summary('P 名目', c, f)
    h_m = [h for h in f.hist if h['phase'] in GR.MANIP]
    if h_m:
        print(f"  P 操作期底盤命令峰 {max(float(np.abs(h['u']).max()) for h in h_m):.6f}；底盤位移 {np.ptp([h['q'][0] for h in h_m]) * 1e3:.3f} mm")
    c = SC.make_lidar(cls=GR.S1Parked)
    t_s = 21.0
    f = fake(c, t_s + 3.0, lambda t, s: dict(forbidden={T.REQ[0]: 0.2, T.REQ[1]: 0.0}) if t >= t_s else {})
    summary('P 停止路徑', c, f)
    c = SC.make_lidar(cls=GR.S1MotM)
    f = fake(c, 50.0)
    rM = summary('M 名目', c, f)
    print('  M 單步週期', c.n_m)
    c = SC.make_lidar(cls=GR.S1MotM)
    t_s = 18.0
    f = fake(c, t_s + 3.0, lambda t, s: dict(forbidden={T.REQ[0]: 0.2, T.REQ[1]: 0.0}) if t >= t_s else {})
    summary('M 停止路徑', c, f)


if __name__ == '__main__':
    sys.exit(main())
