"""自由空間管線的最小通路核對（**不開 Isaac、不用 GPU**，用實際節點）。

三件事，對應指導的第 4 點：
  F1 **NODATA 列真的發布**（外部障礙物為空；TF 齊全 ⇒ 可歸因於空場景）
  F2 **被安全層解析**，且在 freespace_confirmed:=true 下**放行非零命令**
  F3 **停止資料供應後仍會停止**（停發距離雲 ⇒ 安全層回到零輸出）

用 coman_solver_sched_world 提供 /clock、/joint_states、/odom 與 TF，
再起**實際的** robot_state_publisher、arm_link_distance（不給 obstacles）
與 wholebody_safety。**不修改**缺資料／過期／溢位的停止規則。
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
SC = os.environ.get('WG2F_TMP', f'/tmp/wg2f_{os.getpid()}')
DOMAIN = os.environ.get('WG2F_DOMAIN', '181')
URDF_TF = os.path.join(HERE, 'models', 'omni_bot_manip.urdf')
URDF_WB = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
_bad = 0


def ck(name, cond, extra=''):
    global _bad
    print(f'  {name:56s} {"ok" if cond else "**錯**"}{extra}')
    _bad += not cond


def probe(env, out, wait=45):
    """跑實際的通路核對腳本。"""
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, 'wgmpc_wg2_freespace_check.py'),
         '--out', out, '--wait-s', str(wait)],
        cwd=WS, env=env, capture_output=True, text=True, timeout=wait + 60)
    return (json.load(open(out)) if os.path.exists(out) else None), r.stdout


def main() -> int:
    os.makedirs(SC, exist_ok=True)
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    ps = []
    print(f'暫存 {SC}　domain {DOMAIN}　（不開 Isaac、不用 GPU）\n')
    try:
        world = subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'coman_solver_sched_world.py'),
             '--duration-s', '240', '--stop-js-at', '-1', '--no-cloud'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True)
        ps.append(world)
        time.sleep(3)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher',
             URDF_TF, '--ros-args', '-p', 'use_sim_time:=true'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(2)
        dist = subprocess.Popen(
            ['ros2', 'run', 'ammr_wholebody_mpc', 'arm_link_distance',
             '--ros-args', '-p', 'use_sim_time:=true',
             '-p', 'report_frame:=odom', '-p', 'geometry:=links',
             '-p', f'wholebody_urdf:={URDF_WB}'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True)
        ps.append(dist)
        time.sleep(10)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'ammr_wholebody_mpc', 'wholebody_safety',
             '--ros-args', '-p', 'use_sim_time:=true',
             '-p', 'report_frame:=odom', '-p', 'base_frame:=base_link',
             '-p', f'wholebody_urdf:={URDF_WB}',
             '-p', 'vmax_base_lin:=0.035255', '-p', 'vmax_base_ang:=0.199900',
             '-p', 'vmax_arm:=0.999900', '-p', 'freespace_confirmed:=true'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(8)

        # ---------- F1 ＋ F2 ----------
        print('F1/F2 NODATA 列發布、TF 齊全、安全層解析（freespace_confirmed=true）')
        rep, out1 = probe(env, os.path.join(SC, 'alive.json'))
        ck('核對腳本有產出', rep is not None)
        if rep is None:
            print(out1[-1500:])
            return 1
        # 把核對腳本自己的輸出一併顯示（狀態分布、frame 樹等診斷）
        for ln in out1.strip().split('\n'):
            if ln.strip().startswith(('狀態分布', '**實際的 TF', '  ')) \
                    and 'ok' not in ln and '**錯**' not in ln:
                print('    ' + ln.strip())
        for c in rep['checks']:
            ck('  ' + c['name'], c['ok'], f"  {c['detail']}")

        # 安全層在 freespace 下是否放行非零命令
        print('\nF2b 送非零命令，看 cmd_out 是否非零（放行）')
        pub = subprocess.Popen(
            ['ros2', 'topic', 'pub', '-r', '20',
             '/wholebody_safety/cmd_in', 'std_msgs/msg/Float64MultiArray',
             '{data: [0.0, 0.02, 0.1, 0.3, -0.2, 0.1, 0.0, 0.2, 0.0]}'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True)
        ps.append(pub)
        time.sleep(6)
        rep2, out2 = probe(env, os.path.join(SC, 'cmding.json'))
        co = (rep2 or {}).get('cmd_out')
        ck('cmd_out **非零**（freespace 下放行）',
           bool(co) and co.get('nonzero'),
           f"  {co['v'][:3]}…" if co else '  **取不到 cmd_out**')
        if rep2 and 'safety_diag' in rep2:
            sd = rep2['safety_diag']
            ck('  安全層 reason 不是 5（缺資料）', sd.get('reason') != 5.0,
               f"  reason={sd.get('reason')} n_rows={sd.get('n_rows')}")

        # ---------- F3 停止供應 ⇒ 仍會停止 ----------
        print('\nF3 停發距離雲 ⇒ 安全層必須回到零輸出（停止規則未被改動）')
        os.killpg(os.getpgid(dist.pid), signal.SIGTERM)
        time.sleep(8)
        rep3, _ = probe(env, os.path.join(SC, 'stopped.json'))
        co3 = (rep3 or {}).get('cmd_out')
        ck('停發後 cmd_out **全為零**',
           bool(co3) and not co3.get('nonzero'),
           f"  {co3['v'][:3]}…" if co3 else '  **取不到**')
        if rep3 and 'safety_diag' in rep3:
            print(f"    停發後 safety reason={rep3['safety_diag'].get('reason')}"
                  f"（5 = 缺資料、7 = 過期）")
    finally:
        for p in ps:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(2)
        for p in ps:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    print()
    print('自由空間管線核對：' + ('全部通過' if _bad == 0 else f'**{_bad} 項失敗**'))
    print('**這是管線核對，不是到達成果**；第四趟才判到達與保持。')
    return 1 if _bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
