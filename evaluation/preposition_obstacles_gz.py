#!/usr/bin/env python3
"""gz：把 scheduled 情境的每個動態障礙物瞬移到相位 0 位置（/case_start 之前）。

為什麼需要：scheduled 模式的驅動在 /case_start 前只送零速度，障礙物停在世界檔的 spawn 位置；
相位 0 位置（phase0_m、direction、waypoints 決定）通常不在那裡。Isaac 端由模擬器放好，gz 原本沒有這一步，
所以 case_start_check.py preposition 必然不通過。障礙物是 kinematic 模型，用 /world/<world>/set_pose 直接放。

相位 0 位置用驅動同一份函式（dynamic_obstacle_driver.schedule_state, t=0）計算，避免兩份公式分歧。
用法：preposition_obstacles_gz.py --traj <yaml> --world bigarena [--out pre.json]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DRV = os.path.join(HERE, '..', 'src', 'ammr_bringup', 'ammr_bringup', 'dynamic_obstacle_driver.py')


def load_schedule():
    """只取驅動檔的純函式（不載 rclpy）。"""
    src = open(DRV).read()
    i0, i1 = src.index('def profile_arclength'), src.index('class DynamicObstacleDriver')
    ns = {'math': math}
    exec(src[i0:i1], ns)                                                # noqa: S102（同 repo 的固定檔）
    return ns['schedule_state']


def obstacle_dict(d):
    wp = d.get('waypoints')
    pts = [[float(x), float(y)] for x, y in wp] if wp else [[float(v) for v in d['start']], [float(v) for v in d['end']]]
    cum = [0.0]
    for k in range(1, len(pts)):
        cum.append(cum[-1] + math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]))
    return dict(pts=pts, cum=cum, speed=float(d['speed']), phase0_m=float(d.get('phase0_m', 0.0)),
                direction=float(d.get('direction', 1.0)), profile=d.get('speed_profile'))


def phase0_positions(traj_file):
    sched = load_schedule()
    out = {}
    for d in (yaml.safe_load(open(traj_file)).get('dynamic_obstacles') or []):
        p, _ = sched(obstacle_dict(d), 0.0)
        out[d['name']] = (p[0], p[1], float(d.get('height', 1.0)) / 2.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', required=True)
    ap.add_argument('--world', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    pos = phase0_positions(a.traj)
    res, ok = {}, True
    for name, (x, y, z) in pos.items():
        req = 'name: "%s" position { x: %.4f y: %.4f z: %.4f } orientation { w: 1 }' % (name, x, y, z)
        r = subprocess.run(['gz', 'service', '-s', '/world/%s/set_pose' % a.world, '--reqtype', 'gz.msgs.Pose',
                            '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req', req],
                           capture_output=True, text=True, timeout=10)
        good = r.returncode == 0 and 'true' in r.stdout
        ok = ok and good
        res[name] = dict(x=round(x, 4), y=round(y, 4), ok=good, reply=r.stdout.strip()[:60])
        print('%-10s -> (%.3f, %.3f)  %s' % (name, x, y, 'ok' if good else 'FAIL ' + r.stdout.strip() + r.stderr.strip()[:120]))
    if a.out:
        json.dump(dict(traj=a.traj, world=a.world, ok=ok, obstacles=res), open(a.out, 'w'), ensure_ascii=False, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
