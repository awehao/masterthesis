#!/usr/bin/env python3
"""第二批移動物案例（凍結；Codex 20261101_090533 第 3、5 點）：一個圓柱移動物，固定世界軌跡、速度、相對任務起跑（任務時間 t＝0）的時刻，兩組策略相同，不追著機器人觸發。
尺寸與速度沿用既有交通情境 src/ammr_bringup/config/dynamic_trajectories_arena_traffic.yaml 的 dyn_obs_0（0.30 m/s、半徑 0.25 m、高 1.0 m），
速度高於瓶子任務追蹤器的移動判定（min_track_speed 0.20、min_net_speed 0.25 m/s），否則會被當靜止物、兩組無差異。
軌跡：與持瓶斜段平行、反向錯身（迎面走過，不橫越規劃路徑）。由凍結 G 名目 bottle_w1_1 的量測軌跡求得：
  錯身時刻 t_m＝25.0 s，機器人位置 P_m＝(0.4399, −0.5351)，行進方向 d＝(0.6609, −0.7504)；障礙在右側法向 0.80 m 處反向行進。
  0.80 m：中心離規劃路徑 > 0.25＋0.39（代價圖內切）＋0.05（格）＝0.69 ⇒ 不佔用 path_blocked 的內切格；表面離機器人中心 0.55 m < 動態 CBF 邊距 0.69 ⇒ 確實進入動態 CBF 影響範圍。
  起點 S＝錯身點前 2.2 m、終點 E＝錯身點後 2.5 m；t_start 前靜止在 S，之後等速到 E 後靜止。
"""
import numpy as np

CASE = dict(name='w1_dyn_pass1', source='arena_traffic dyn_obs_0', speed=0.30, radius=0.25, height=1.0,
            start=(1.2936, -2.7148), end=(-1.8128, 0.8123), t_start=17.667, t_meet=25.0, lateral=0.80,
            meet_robot=(0.4399, -0.5351), robot_dir=(0.6609, -0.7504))


def state(t, case=CASE):
    """任務時間 t 的 (中心 xy, 速度 xy)。"""
    S, E = np.array(case['start'], float), np.array(case['end'], float)
    L = float(np.linalg.norm(E - S))
    u = (E - S) / L
    s = min(max((float(t) - case['t_start']) * case['speed'], 0.0), L)
    moving = case['t_start'] <= float(t) and s < L
    return S + u * s, (u * case['speed'] if moving else np.zeros(2))


def ray_cyl(origin, angles, center, radius, rmax):
    """2D 射線打圓：回傳每條射線距離（未命中 inf）。"""
    o = np.asarray(origin, float)
    dx, dy = np.cos(angles), np.sin(angles)
    fx, fy = o[0] - center[0], o[1] - center[1]
    b = fx * dx + fy * dy
    cc = fx * fx + fy * fy - radius * radius
    disc = b * b - cc
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (t > 0) & (t <= rmax), t, np.inf)
