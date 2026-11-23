#!/usr/bin/env python3
"""WG2 命令追蹤封裝的反例核對（真實安全層，**不開 Isaac**）。

驗的是：**同值不同序號、重複輸出、延遲／遺失、安全層修改、停止命令**
都不會被誤配。

界線：序號只說明「執行了哪筆、被改成什麼」，
**不保證未來安全層不修改命令**。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import wgmpc_cmd_envelope as ENV                                  # noqa: E402

URDF_TF = os.path.join(HERE, 'models', 'omni_bot_manip.urdf')
URDF_WB = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
SC = os.environ.get('WG2E_TMP', '/tmp/wg2_env')
DOMAIN = os.environ.get('WG2E_DOMAIN', '131')
FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:52s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


DRIVER = r'''
import json, sys, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
sys.path.insert(0, %r)
import wgmpc_cmd_envelope as ENV

class D(Node):
    def __init__(self):
        super().__init__('env_driver')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
        self.pub = self.create_publisher(
            Float64MultiArray, ENV.TOPIC[ENV.ST_SOLVER], 10)
        self.got = []
        self.create_subscription(Float64MultiArray,
                                 ENV.TOPIC[ENV.ST_SAFETY], self._cb, 50)
        self.rid = ENV.run_id_num('env_test')

    def _cb(self, m):
        try:
            self.got.append(ENV.decode(m.data))
        except ValueError:
            pass

    def send(self, seq, u):
        t = self.get_clock().now().nanoseconds * 1e-9
        mm = Float64MultiArray()
        mm.data = ENV.encode(self.rid, seq, ENV.ST_SOLVER, seq, seq,
                             True, t, u)
        self.pub.publish(mm)

rclpy.init(); nd = D()
for _ in range(60):
    rclpy.spin_once(nd, timeout_sec=0.05)
U1 = [0.0, 0.0, 0.0, 0.3, -0.2, 0.1, 0.0, 0.0, 0.0]
# 1) 同一個九維值，送兩個**不同**的 source_seq。
#    送出後**連續 spin**，讓安全層在命令仍新鮮時多次輸出
#    —— 這才是「同一請求被多次處理」的實況。
for seq in (101, 102):
    nd.send(seq, U1)
    for _ in range(60):
        rclpy.spin_once(nd, timeout_sec=0.02)
# 2) 停發一段（延遲／遺失），序號留缺口
for _ in range(30):
    rclpy.spin_once(nd, timeout_sec=0.05)
# 3) 再送一筆，序號跳號（模擬遺失 103–109）
nd.send(110, [0.0, 0.0, 0.0, -0.9, 0.9, -0.9, 0.9, -0.9, 0.9])
for _ in range(60):
    rclpy.spin_once(nd, timeout_sec=0.02)
# 4) **強制越界**：手臂速率 5.0 遠超低速框 0.9999 ⇒ 安全層必須修改
nd.send(120, [0.5, 0.5, 2.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
for _ in range(60):
    rclpy.spin_once(nd, timeout_sec=0.02)
json.dump([{k: (v if not isinstance(v, list) else [round(x, 8) for x in v])
            for k, v in g.items()} for g in nd.got],
          open(sys.argv[1], 'w'))
nd.destroy_node(); rclpy.try_shutdown()
'''


def main() -> int:
    os.makedirs(SC, exist_ok=True)
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    ps = []
    print(f'暫存 {SC}　domain {DOMAIN}　（不開 Isaac、不用 GPU）\n')
    try:
        ps.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'coman_solver_sched_world.py'),
             '--duration-s', '240', '--stop-js-at', '-1', '--no-cloud'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(3)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher',
             URDF_TF, '--ros-args', '-p', 'use_sim_time:=true'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(2)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'ammr_wholebody_mpc', 'arm_link_distance',
             '--ros-args', '-p', 'use_sim_time:=true',
             '-p', 'report_frame:=odom', '-p', 'geometry:=links',
             '-p', f'wholebody_urdf:={URDF_WB}'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(10)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'ammr_wholebody_mpc', 'wholebody_safety',
             '--ros-args', '-p', 'use_sim_time:=true',
             '-p', 'report_frame:=odom', '-p', 'base_frame:=base_link',
             '-p', f'wholebody_urdf:={URDF_WB}',
             '-p', 'vmax_base_lin:=0.035255', '-p', 'vmax_base_ang:=0.199900',
             '-p', 'vmax_arm:=0.999900', '-p', 'freespace_confirmed:=true',
             '-p', 'cmd_env:=true'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(8)

        drv = os.path.join(SC, 'driver.py')
        out = os.path.join(SC, 'got.json')
        open(drv, 'w').write(DRIVER % HERE)
        r = subprocess.run([sys.executable, drv, out], cwd=WS, env=env,
                           capture_output=True, text=True, timeout=180)
        if not os.path.exists(out):
            print(r.stdout[-800:], r.stderr[-800:])
            ck('驅動器有產出', False, '**無輸出**')
            return 1
        got = json.load(open(out))
        print(f'收到安全層輸出 {len(got)} 筆\n')

        print('E1  封裝格式與身分')
        ck('每筆都有 output_seq 且**嚴格遞增**',
           all(got[i]['output_seq'] < got[i + 1]['output_seq']
               for i in range(len(got) - 1)) and len(got) > 1,
           f"{got[0]['output_seq']}–{got[-1]['output_seq']}")
        ck('九維值與身分在**同一份訊息**',
           all(len(g['u']) == 9 for g in got), f'{len(got)} 筆')

        print('\nE2  同值不同序號：不可被當成同一筆')
        d1 = [g for g in got if g['source_seq'] == 101 and g['derived']]
        d2 = [g for g in got if g['source_seq'] == 102 and g['derived']]
        ck('兩個 source_seq 都有對應的輸出',
           len(d1) > 0 and len(d2) > 0, f'101:{len(d1)} 筆、102:{len(d2)} 筆')
        if d1 and d2:
            same_val = np.allclose(d1[0]['u'], d2[0]['u'])
            ck('**值相同但 source_seq 不同 ⇒ 仍可分辨**',
               d1[0]['output_seq'] != d2[0]['output_seq'],
               f"值相同={same_val}、output_seq "
               f"{d1[0]['output_seq']} vs {d2[0]['output_seq']}")

        print('\nE3  同一請求被**多次輸出**：每次都有自己的 output_seq')
        ck('同一 source_seq 出現**多筆**輸出（安全層每週期各輸出一次）',
           len(d1) > 1 or len(d2) > 1,
           f'101:{len(d1)} 筆、102:{len(d2)} 筆')
        ck('多筆輸出的 output_seq 互不相同',
           len({g['output_seq'] for g in d1 + d2}) == len(d1 + d2), '')

        print('\nE4  安全層**修改**命令：source_seq 保留、值可不同')
        sent120 = [0.5, 0.5, 2.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0]
        d120 = [g for g in got if g['source_seq'] == 120 and g['derived']]
        ck('越界命令有對應的輸出', len(d120) > 0, f'{len(d120)} 筆')
        if d120:
            mod = [g for g in d120 if not np.allclose(g['u'], sent120)]
            ck('**確實被修改**（否則本項為空過）', len(mod) > 0,
               f'{len(mod)} / {len(d120)} 筆與送出值不同')
            ck('被修改的輸出仍帶**正確的 source_seq**',
               all(g['source_seq'] == 120 for g in mod), '')
            if mod:
                ck('**看得出被改成什麼**（值隨封裝一起回報）',
                   max(abs(x) for x in mod[0]['u'][3:9]) <= 0.9999 + 1e-6,
                   f'送出手臂 ±5.0 → 輸出 '
                   f'{[round(x,4) for x in mod[0]["u"][3:6]]}（框 0.9999）')

        print('\nE5  停止／自行產生的輸出：derived = 0 且不冒認來源')
        nd0 = [g for g in got if not g['derived']]
        ck('存在 derived = 0 的輸出（命令過期後的停止）',
           len(nd0) > 0, f'{len(nd0)} 筆')
        ck('derived = 0 時 **source_seq = −1**（不冒認任何來源）',
           all(g['source_seq'] == -1 and g['src_seq'] == -1 for g in nd0),
           '')

        print('\nE6  序號缺口（延遲／遺失）可被看見，且不誤配')
        seqs = sorted({g['source_seq'] for g in got if g['derived']})
        ck('缺口可見（101,102 之後直接跳到 110）',
           110 in seqs and not any(103 <= x <= 109 for x in seqs),
           f'出現過的 source_seq：{seqs}')
        ck('沒有任何輸出宣稱來自未送出的序號',
           all(x in (101, 102, 110, 120) for x in seqs), f'{seqs}')

        print('\nE7  格式說明可取得且標明界線')
        de = ENV.describe()
        ck('說明含「身分與值同訊息」', 'identity_with_value' in de, '')
        ck('說明含「不保證安全層不修改命令」',
           '不保證' in de.get('not_guaranteed', ''), '')
        ck('說明含「既有路徑不變」', 'existing_path' in de, '')
    finally:
        for p in ps:
            p.terminate()
        time.sleep(2)
        for p in ps:
            if p.poll() is None:
                p.kill()
    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('全部通過。序號讓我們知道「執行了哪筆、被改成什麼」，')
        print('**但不保證未來安全層不修改命令**。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
