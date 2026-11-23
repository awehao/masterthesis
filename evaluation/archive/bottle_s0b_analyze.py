#!/usr/bin/env python3
"""S0b 逐步紀錄獨立重算（不用判定程式的中間值，只讀 s0b_steps.npz）：同 S0a 各項，另加底盤進度（實測速度、對命令與參考的誤差、可達窗）、抓持期 TCP 世界漂移、瓶子隨底盤移動量、停止或收尾。
    python3 evaluation/bottle_s0b_analyze.py <RUN_ID>
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 's0b_steps.npz'))
    R, cols = Z['rows'], [str(c) for c in Z['cols']]
    rep = json.load(open(os.path.join(D, 's0b_report.json')))
    c = {k: R[:, i] for i, k in enumerate(cols)}
    t, ph = c['t'], c['phase_index'].astype(int)
    names = [str(x) for x in Z['phases']]
    print(f'步數 {len(t)}，t {t[0]:.2f}–{t[-1]:.2f} s；各相位時長', {names[i]: round(float((ph == i).sum() * 0.01), 2) for i in sorted(set(ph))})
    tc, tg, tl, tok = rep['t_close'], rep['t_grasp'], rep['t_lift_start'], rep['t_lift_ok']
    print(f'閉爪請求 {tc:.2f}、證據成立 {tg:.2f}（{tg - tc:.2f} s）、抬升開始 {tl:.2f}、離桌成立 {tok:.2f}')
    f1, f2 = c['f_finger1_bottle'], c['f_finger2_bottle']
    m = t >= tg
    print(f'手指對瓶力（證據成立後）：指1 均 {np.nanmean(f1[m]):.2f} min {np.nanmin(f1[m]):.2f} max {np.nanmax(f1[m]):.2f}；指2 均 {np.nanmean(f2[m]):.2f} min {np.nanmin(f2[m]):.2f} max {np.nanmax(f2[m]):.2f} N；NaN 步 {int(np.isnan(f1).sum())}')
    pre = t < tc
    print(f'閉爪前手指力峰 {np.nanmax(np.r_[f1[pre], f2[pre]]):.4f} N')
    t_first = t[np.argmax((f1 > 0.5) & (f2 > 0.5))]
    print(f'兩指首次皆 ≥0.5 N：{t_first:.2f} s（閉爪後 {t_first - tc:.2f} s）')
    top = rep['asset']['table']['top_z']
    _qx, _qy = c['bottle_qx'], c['bottle_qy']
    _axz = 1 - 2 * (_qx ** 2 + _qy ** 2)                               # 瓶軸 z 分量
    z = c['bottle_base_z'] - float(rep['asset']['bottle']['radius']) * np.sqrt(np.clip(1 - _axz ** 2, 0, 1))   # 傾斜後瓶底最低點（Codex 20261031_190217）
    print(f'瓶底最低點高度：起 {z[0] - top:.5f}、抬升前 {z[t < tl][-1] - top:.5f}、峰 {z.max() - top:.4f}、末 {z[-1] - top:.4f} m；≥40 mm 自 {t[np.argmax(z - top >= 0.04)]:.2f} s 起')
    ft = c['f_bottle_table']
    print(f'瓶↔桌力：抬升前均 {np.nanmean(ft[t < tc]):.3f} N（瓶重 1.962）；離桌成立後 max {np.nanmax(ft[t >= tok]):.4f} N；NaN 步 {int(np.isnan(ft).sum())}')
    qw, qx, qy, qz = c['bottle_qw'], c['bottle_qx'], c['bottle_qy'], c['bottle_qz']
    ax_z = 1 - 2 * (qx ** 2 + qy ** 2)
    tilt = np.degrees(np.arccos(np.clip(ax_z, -1, 1)))
    print(f'瓶傾角：閉爪前 max {tilt[pre].max():.3f}°、全程 max {tilt.max():.3f}°、末 {tilt[-1]:.3f}°')
    dxy = np.hypot(c['bottle_base_x'] - c['bottle_base_x'][0], c['bottle_base_y'] - c['bottle_base_y'][0])
    print(f'瓶水平位移：閉爪前 max {dxy[pre].max() * 1e3:.2f} mm；閉爪到抬升開始 max {dxy[(t >= tc) & (t < tl)].max() * 1e3:.2f} mm；全程 max {dxy.max() * 1e3:.2f} mm')
    print(f"滑移（判定）{rep['slip_max_mm_deg']}；穿透估計峰 {rep['pen_max_mm']:.3f} mm")
    for k in cols:
        if k.startswith('forb_'):
            v = c[k]
            print(f'  {k:<24} 峰 {np.nanmax(v):.4f} N、>0 步 {int((v > 0).sum())}、≥0.05 N 步 {int((v >= 0.05).sum())}、NaN {int(np.isnan(v).sum())}')
    x, vx_m = c['base_x'], np.gradient(c['base_x'], t)
    print(f"底盤：x {x[0]:.4f} → {x[-1]:.4f}，y 峰 {np.abs(c['base_y']).max() * 1e3:.2f} mm、偏航峰 {np.degrees(np.abs(c['base_yaw']).max()):.3f}°；實測 vx 峰 {vx_m.max():.4f}、最低 {vx_m.min():.4f} m/s（命令峰 {c['base_cmd_vx'].max():.4f}）")
    print(f"底盤實測 vx − 命令 vx：p95 {np.percentile(np.abs(vx_m - c['base_cmd_vx']), 95):.4f}、max {np.abs(vx_m - c['base_cmd_vx']).max():.4f} m/s；x − s_ref：max {np.abs(x - c['s_ref']).max() * 1e3:.2f} mm；v_b 範圍 {c['v_b'].min():.4f}–{c['v_b'].max():.4f}")
    win_lo, win_hi = rep.get('window', (None, None))
    print(f"抓取關鍵時刻底盤 x：閉爪 {np.interp(tc, t, x):.4f}、證據 {np.interp(tg, t, x):.4f}、抬升 {np.interp(tl, t, x):.4f}、離桌 {np.interp(tok, t, x):.4f}；可達窗 {win_lo}–{win_hi}")
    m_g = (t >= tc) & (t <= tl)
    print(f"閉爪至抬升期間底盤移動 {(x[m_g].max() - x[m_g].min()) * 1e3:.2f} mm（TCP 目標世界固定）；瓶子同期水平位移見上")
    # 分段實測速度與近停（Codex 20261031_195504）：速度由位姿差分，5 點移動平均；近停門檻 0.01 m/s（事前沿用 V_MIN，不是判定項）
    vs = np.convolve(np.gradient(x, t), np.ones(5) / 5, mode='same')
    print('分段實測底盤速度（m/s）與近停（< 0.01 m/s）：')
    for nm, t0, t1 in (('接近至閉爪', 0.0, tc), ('閉爪至證據', tc, tg), ('證據至抬升開始', tg, tl), ('抬升開始至離桌成立', tl, tok), ('離桌成立至結束', tok, t[-1])):
        mk = (t >= t0) & (t < t1)
        if mk.any():
            print(f"  {nm:<12} {t0:6.2f}–{t1:6.2f} s：均 {vs[mk].mean():.4f}、最低 {vs[mk].min():.4f}、最高 {vs[mk].max():.4f}；近停 {float((vs[mk] < 0.01).sum()) * 0.01:.2f} s／{float(mk.sum()) * 0.01:.2f} s")
    ms_all = np.array(rep.get('solve_ms_all') or [float('nan')])
    print(f"求解器：{rep.get('qp_counts')}；GI 未採用原因 {rep.get('gi_fail')}；備援耗時峰 {rep.get('backup_ms_max')} ms；端到端 ms p50 {np.nanmedian(ms_all):.1f}、p95 {np.nanpercentile(ms_all, 95):.1f}、p99 {np.nanpercentile(ms_all, 99):.1f}、max {np.nanmax(ms_all):.1f}；>50 ms {int((ms_all > 50.0).sum())}／{ms_all.size} 週期；管外擋 INSERT {rep.get('n_gate')} 週期")
    print(f"實際終點 x {x[-1]:.4f}（參考終點 {rep.get('x_end', 0.15)}）；終點未到不算到達參考終點")
    print(f"手指：指1 末 {c['finger1_q'][-1]:.5f}、指2 末 {c['finger2_q'][-1]:.5f} m；主從差峰 {np.abs(c['finger1_q'] - c['finger2_q']).max() * 1e3:.3f} mm；命令末 {c['finger_cmd'][-1]:.4f}")
    print('d̂', np.round(rep['d_hat'], 4).tolist(), '求解 ms', rep['solve_ms'], '求解失敗', rep['n_solve_fail'], '禁止脈衝', {k: v for k, v in rep['forbidden_pulses'].items() if v['peak'] > 0})


if __name__ == '__main__':
    sys.exit(main())
