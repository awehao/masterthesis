#!/usr/bin/env python3
"""K3-M 前置離線核對：中止後的底盤停止推進（純離線、只讀；依 Codex 20261023_202016 第 2 項）。

不另改減速律或安全門檻，直接用既有 `wb_wheel_limit.limit9_timeout`（E2 逾時路徑：底盤經輪級限制減速、手臂速率立即歸零）
與 `WheelLimitConfig` 預設值，物理步長 0.01 s，逐步推進到底盤命令為零。
起始命令（本體座標）取最壞情形：
  - box：逐軸速度框上限 (0.035255, 0.035255, 0.1999)（W-GMPC 可輸出的最大值）；
  - latch：低速介面閂鎖邊界（合線 0.05 m/s、0.20 rad/s；超過即由 E2 閂鎖，不會被套用）；
  - replay：door_k3m_offline_replay_r2.json free 的最大逐軸套用值。
反應時間（越界到鏈內命令過期之間，底盤以起始命令繼續走）：
  - 執行端包絡／走廊監看：逐物理步檢查 ⇒ 1 步（0.01 s）；
  - 策略端監看：任務週期 0.05 s＋狀態年齡上限 0.2 s ⇒ 0.25 s（最壞）；
  - 另列牆鐘停更：命令鏈逾時 max_cmd_age_s（0.2 s）＋1 步。
底盤位移以本體座標積分（偏航角小，取起始方向近似）；報剩餘前進／側向位移、合位移與偏航；手臂停止後設定點凍結（速率 0）。
界線：Isaac 底盤以速度直接設定（無額外致動延遲假設）；不含接觸力造成的被動移動。

    python3 evaluation/door_k3m_offline_stop.py   # 寫 results/door/door_k3m_offline_stop.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wb_wheel_limit as WL                           # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_stop.json')
DT = 0.01
REACT = {'executor_per_step': 0.01, 'policy_worst': 0.25, 'chain_timeout': 0.21}


def propagate(u0, t_react, cfg):
    u = np.asarray(u0, float).copy()
    x = np.zeros(3)                                    # 本體座標前進、側向、偏航
    n_react = int(round(t_react / DT))
    for _ in range(n_react):
        x += u[:3] * DT
    n = 0
    while np.abs(u[:3]).max() > 1e-12 and n < 10000:
        r = WL.limit9_timeout(u, DT, cfg)
        if r.u_out is None:
            u = WL.stop_command(u)
        else:
            u = r.u_out
        x += u[:3] * DT
        n += 1
    return {'reaction_s': t_react, 'decel_steps': n, 'fwd_mm': x[0] * 1e3, 'lat_mm': x[1] * 1e3,
            'disp_mm': float(math.hypot(x[0], x[1])) * 1e3, 'yaw_deg': math.degrees(x[2]), 'arm_rate_after': float(np.abs(u[3:]).max())}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    cfg = WL.WheelLimitConfig()
    rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
    fs = next(c for c in rp['configs'] if c['name'] == 'free')['summary']
    starts = {'box': [0.035255, 0.035255, 0.1999], 'latch': [0.05 / math.sqrt(2), 0.05 / math.sqrt(2), 0.20],
              'replay': [fs['u_base_vx_absmax'], fs['u_base_vy_absmax'], fs['u_base_wz_absmax']]}
    res = {'schema': 'door_k3m_offline_stop/1', 'dt': DT, 'wheel_cfg': cfg.__dict__, 'starts_body': starts, 'reaction_s': REACT, 'cases': []}
    for sn, ub in starts.items():
        u0 = np.array(list(ub) + [0.5] * 6)                  # 手臂起始速率任意非零：驗證逾時後立即歸零
        for rn, tr in REACT.items():
            c = propagate(u0, tr, cfg)
            c.update(start=sn, reaction=rn)
            res['cases'].append(c)
    res['worst'] = {rn: {k: max(abs(c[k]) for c in res['cases'] if c['reaction'] == rn) for k in ('disp_mm', 'yaw_deg', 'fwd_mm', 'lat_mm')}
                    for rn in REACT}
    res['note'] = '既有 E2 逾時路徑的純離線推進；不改減速律或門檻。用來選內側中止界限（外側核對範圍減去最壞剩餘位移）。'
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for c in res['cases']:
        print(f"{c['start']:6s} {c['reaction']:18s} 反應 {c['reaction_s']:.2f} s、減速 {c['decel_steps']} 步：前進 {c['fwd_mm']:.2f}、側向 {c['lat_mm']:.2f}、"
              f"合 {c['disp_mm']:.2f} mm、偏航 {c['yaw_deg']:.3f}°、手臂後 {c['arm_rate_after']}")
    print('最壞：', {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in res['worst'].items()})


if __name__ == '__main__':
    main()
