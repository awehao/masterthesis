#!/usr/bin/env python3
"""完整往返分段報告（既有紀錄）：靜止段、控制權切換、終態、監看、抓持、5 Hz 帶、交棒命令連續性、車頭與運動方向。
    python3 evaluation/mission_segment_report.py mission_rt_32 [mission_rt_28 ...]
"""
import json
import re
import sys

import numpy as np


def b5(y, dt=0.01):
    y = np.asarray(y, float)
    y = y - y.mean()
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) * 4 / len(y)
    f = np.fft.rfftfreq(len(y), dt)
    m = (f > 3.5) & (f < 6.5)
    return round(float(s[m].max()), 4)


def main():
    for R in sys.argv[1:]:
        D = f'evaluation/runs/{R}'
        m = json.load(open(f'{D}/room_run_mission_merged.json'))
        I = {k: i for i, k in enumerate(m['steps_cols'])}
        T = m['steps']
        t = np.array([x[I['sim_t']] for x in T])
        b = np.array([x[I['base_xyth']] for x in T], float)
        a = np.array([x[I['q_arm_meas']] for x in T], float)
        own = np.array([x[I['owner(0=nav,1=wb,2=glide)']] for x in T])
        q = np.degrees([x[I['door_q_rad']] for x in T])
        st = 20
        y = np.unwrap(b[:, 2])
        dts = t[st:] - t[:-st]
        v = np.hypot(*(b[st:, :2] - b[:-st, :2]).T) / dts
        w = np.abs(y[st:] - y[:-st]) / dts
        am = np.abs(a[st:] - a[:-st]).max(1) / dts
        idle = (v < 0.005) & (w < 0.01) & (am < 0.01)
        tt, out, i = t[st:], [], 0
        while i < len(idle):
            if idle[i]:
                j = i
                while j + 1 < len(idle) and idle[j + 1]:
                    j += 1
                if tt[j] - tt[i] >= 1.0:
                    out.append((round(tt[i], 1), round(tt[j] - tt[i], 1), int(own[st + i])))
                i = j + 1
            else:
                i += 1
        print(R, '末 sim', round(t[-1], 1), '靜止≥1 s 總計', round(sum(o[1] for o in out), 1))
        print('  靜止段(起,長,owner)', out)
        print('  owner 切換', [(round(t[k], 1), int(own[k - 1]), int(own[k])) for k in range(1, len(own)) if own[k] != own[k - 1]])
        e = b[-1]
        print('  終點 距 home mm', round(float(np.hypot(e[0] + .6, e[1] + 2.8)) * 1e3, 2), '朝向差', round(float(np.arctan2(np.sin(e[2] - np.pi / 2), np.cos(e[2] - np.pi / 2))), 4),
              '門', round(float(q[-1]), 3), '抽屜 mm', round(T[-1][I['opening_m']] * 1e3, 2))
        print('  門 B 放手後', 'max q(760 後)', round(float(q[t > t[-1] - 40].max()), 3))
        for sg in 'AB':
            d = json.load(open(f'{D}/task_return_{sg}.json'))
            mm = d['monitor']
            print('  ', sg, d['result'], 'n', mm['n'], 'stop', mm['n_stop'], 'warn', mm['n_warn'], {k: round(v_ * 1e3, 2) for k, v_ in mm['pred_min'].items()}, 'age', round(mm['age_max'], 3))
        r = json.load(open(f'{D}/room_run_drawer.json'))
        C = r['steps_cols']
        ix = {k: i for i, k in enumerate(C)}
        good = lambda x, n: isinstance(x, list) and len(x) >= n and all(e_ is not None for e_ in x[:n])
        Sd = [x for x in r['steps'] if good(x[ix['applied_arm']], 6) and good(x[ix['base_cmd_body']], 3)]
        td = np.array([x[1] for x in Sd])
        A = np.array([x[ix['applied_arm']] for x in Sd], float)
        C3 = np.array([x[ix['base_cmd_body']][:3] for x in Sd], float)
        Od = np.array([x[2] for x in Sd])
        g = r['return_gate']
        ev = {f"{e_.get('op')}{e_.get('to')}": e_['t'] for e_ in g['events'] if 'to' in e_}
        print('   閘門', {k: round(v_, 1) for k, v_ in ev.items() if k[1:] in ('MOVING', 'OPENING', 'FREE')}, '漂移', np.round(g['drift_max'], 4).tolist())
        ph = dict((b_, float(c_)) for a_, b_, c_ in re.findall(r'相位 (\S+) → \*\*(\S+)\*\* @ sim ([\d.]+)', open(f'{D}/task_d.log').read()))
        print('   抽屜', {k: ph.get(k) for k in ('ENGAGE_WAIT', 'OPEN', 'OPEN_HOLD', 'CLOSE', 'CLOSE_HOLD', 'RELEASE_WAIT')}, re.findall(r'夾持成立[^\n]*', open(f'{D}/task_d.log').read())[:1])

        def W(x_, y_):
            k = (td >= x_) & (td < y_)
            return b5(A[k, 1]) if k.sum() > 100 else None
        print('   5 Hz 帶 j1：S4', W(ev['1MOVING'], ev['1OPENING']), 'C4', W(ev['2MOVING'], ev['2OPENING']), 'OPEN', W(ph['OPEN'], ph['OPEN_HOLD']), 'CLOSE', W(ph['CLOSE'], ph['CLOSE_HOLD']))
        rows = [x for x in r['steps'] if x[ix['return']] and good(x[ix['return']][1:3], 2)]
        T2 = np.array([x[1] for x in rows])
        F = np.array([x[ix['return']][1:3] for x in rows], float)
        for k in ('1', '2'):
            mk = (T2 >= ev[k + 'MOVING']) & (T2 < ev[k + 'OPENING'])
            print('   抓持 op', k, 'f 均', F[mk].mean(0).round(2).tolist(), 'min', F[mk].min(0).round(2).tolist(), '<0.1 N', (F[mk] < .1).mean(0).round(3).tolist())
        for k in range(1, len(Od)):
            if Od[k] != Od[k - 1]:
                print('   切換', round(td[k], 2), int(Od[k - 1]), '→', int(Od[k]), 'cmd', C3[max(k - 3, 0)].round(3).tolist(), '→', C3[min(k + 6, len(C3) - 1)].round(3).tolist())


if __name__ == '__main__':
    main()
