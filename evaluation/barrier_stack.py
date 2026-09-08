"""Bring up the link-safety stack against an already-running Gazebo.

Deliberately not a ros2 launch file. Two of the six processes -- the adapter and
the watchdog gate -- have to be independent OS processes with the gate as the
sole publisher on the controller's command topic, and the sim must be able to
keep running while this stack is restarted. Starting it separately makes both
true by construction rather than by configuration.

One expansion, one description
------------------------------
The whole-body xacro is expanded ONCE here and the resulting file is handed to
every node that needs it. The wire between the distance node and the safety node
carries a link INDEX, not a name: if the two ends expanded the description
separately and the file changed in between, every barrier row would attach to
the wrong link, silently, with a residual that still looks plausible. The obstacle
list is generated from the same world SDF the simulator loaded, for the same
reason.

What this configuration does NOT test
-------------------------------------
`require_occlusion_feed` is set false. The obstacle here is a static box read
from the world file -- ground truth, not something the lidar perceived -- so
there is no occlusion question to answer and no self-filter running to answer
it. This run therefore says nothing about the perception path. With a perceived
obstacle the flag must go back to true, because absence of the occlusion feed is
not evidence of no occlusion.

The base is fixed: `fix_base:=true` constrains the three base velocities to zero
INSIDE the solve, so the barrier cannot quietly assume the chassis will move out
of the way and then have that part of the answer thrown away downstream.

    python3 evaluation/barrier_stack.py [--no-foxglove] [--gate-timeout 0.15]
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WHOLEBODY = os.path.join(
    ROOT, 'src/my_omnibot_description/urdf/omni_bot_wholebody.urdf.xacro')
WORLD = os.path.join(ROOT, 'src/ammr_bringup/worlds/arm_barrier_test.sdf')


def specs_from_world(path: str) -> list[str]:
    """Distance-node obstacle specs, from the world the simulator loaded.

    Format is name:model:kind:dims:xyz:rpy, and an empty model field means the
    pose is the world pose of a static body -- which is what every obs_* in this
    world is. Generated rather than typed so the two ends cannot drift apart.
    """
    sdf = open(path).read()
    out = []
    for m in re.finditer(r'<model name="(obs_\d+)">(.*?)</model>', sdf, re.S):
        body = m.group(2)
        pose = re.search(r'<pose>([-\d.eE\s]+)</pose>', body)
        box = re.search(r'<box><size>([^<]+)</size>', body)
        cyl = re.search(r'<cylinder><radius>([\d.]+)</radius>\s*<length>([\d.]+)',
                        body)
        if not pose:
            continue
        v = [float(x) for x in pose.group(1).split()]
        xyz = ','.join(f'{x:g}' for x in v[:3])
        rpy = ','.join(f'{x:g}' for x in v[3:6]) if len(v) >= 6 else '0,0,0'
        if box:
            dims = ','.join(f'{float(x):g}' for x in box.group(1).split())
            out.append(f'{m.group(1)}::box:{dims}:{xyz}:{rpy}')
        elif cyl:
            out.append(f'{m.group(1)}::cylinder:{cyl.group(1)},{cyl.group(2)}'
                       f':{xyz}:{rpy}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default=WORLD)
    ap.add_argument('--urdf', default=WHOLEBODY)
    ap.add_argument('--report-frame', default='world')
    ap.add_argument('--spawn-z', type=float, default=0.05,
                    help='z of the model root in the world, from `gz model -p`')
    # The base is fixed but not necessarily at the origin: 5B parks it at a
    # standoff so the pre-grasp point lands inside the arm's workspace. Read
    # these from `gz model -m omni_bot -p` after moving it, never from the
    # value that was ASKED for -- the two differ once physics has settled, and
    # the distance node would then be looking at a different scene.
    ap.add_argument('--base-x', type=float, default=0.0)
    ap.add_argument('--base-y', type=float, default=0.0)
    ap.add_argument('--base-yaw', type=float, default=0.0)
    ap.add_argument('--fix-base', dest='fix_base', action='store_true',
                    default=True,
                    help='hold the chassis: static world TF, base velocities '
                         'forced to zero INSIDE the solve')
    ap.add_argument('--free-base', dest='fix_base', action='store_false',
                    help='whole-body: base is part of the solution, world TF '
                         'comes live from odometry, /cmd_vel is bridged')
    ap.add_argument('--gate-timeout', type=float, default=0.15)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--max-rows-per-link', type=int, default=60)
    ap.add_argument('--no-foxglove', action='store_true')
    ap.add_argument('--camera-topic', default='/demo_cam',
                    help='bridge this gz camera to ROS for recording; empty to skip')
    ap.add_argument('--indep-n', type=int, default=10000)
    a = ap.parse_args()

    specs = specs_from_world(a.world)
    if not specs:
        print(f'no obs_* in {a.world}', file=sys.stderr)
        return 1
    print('障礙物規格（由世界檔產生）:')
    for s in specs:
        print('   ', s)

    print('展開 whole-body 描述…', flush=True)
    xml = subprocess.check_output(['xacro', a.urdf], text=True)
    fd, urdf = tempfile.mkstemp(prefix='wholebody_', suffix='.urdf')
    with os.fdopen(fd, 'w') as f:
        f.write(xml)
    print(f'   {urdf}  ({len(xml)} bytes) — 三個節點共用同一份')

    procs: list[tuple[str, subprocess.Popen]] = []

    def spawn(label, cmd):
        # Own process group per child. `ros2 run X Y` execs a WRAPPER that
        # spawns the real node as a grandchild: signalling the wrapper alone
        # left the node running, reparented and invisible to this script. Three
        # restarts that way accumulated six orphaned nodes at ~80% CPU each,
        # all still publishing on the same topics as the live stack -- load
        # average 19, /joint_states gaps of 337 ms, and every safety topic
        # carrying messages from four different publishers. Signal the GROUP.
        print(f'  啟動 {label}', flush=True)
        p = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
        procs.append((label, p))
        return p

    try:
        # world -> model root. Static ONLY when the base is held fixed; once
        # the base is part of the solution a static transform tells the safety
        # filter the robot never left the start, and every barrier row is then
        # built about the wrong place with a residual that still looks healthy.
        if a.fix_base:
            spawn('static_tf', [
                'ros2', 'run', 'tf2_ros', 'static_transform_publisher',
                '--x', str(a.base_x), '--y', str(a.base_y),
                '--z', str(a.spawn_z), '--yaw', str(a.base_yaw),
                '--frame-id', a.report_frame,
                '--child-frame-id', 'base_footprint'])
        else:
            spawn('odom_bridge', [
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '/odom_raw@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '--ros-args', '-r', '/odom_raw:=/odom',
                '-p', 'use_sim_time:=true'])
            spawn('cmd_vel_bridge', [
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '--ros-args', '-p', 'use_sim_time:=true'])
            spawn('base_tf', [
                sys.executable, os.path.join(HERE, 'base_tf_bridge.py'),
                '--frame', a.report_frame, '--z', str(a.spawn_z)])
        time.sleep(2.0)

        spawn('arm_link_distance', [
            'ros2', 'run', 'ammr_wholebody_mpc', 'arm_link_distance',
            '--ros-args',
            '-p', 'use_sim_time:=true',
            '-p', f'report_frame:={a.report_frame}',
            '-p', 'geometry:=links',
            '-p', f'wholebody_urdf:={urdf}',
            '-p', f'max_rows_per_link:={a.max_rows_per_link}',
            '-p', 'require_occlusion_feed:=false',
            # Quoted. The override is parsed as YAML, so an unquoted
            # obs_0::box:0.4,1,1.2:... inside a flow sequence is read as a
            # mixed float/integer list and rcl refuses the whole argument.
            '-p', 'obstacles:=[' + ','.join(f'"{x}"' for x in specs) + ']'])

        spawn('wholebody_safety', [
            'ros2', 'run', 'ammr_wholebody_mpc', 'wholebody_safety',
            '--ros-args',
            '-p', 'use_sim_time:=true',
            '-p', f'report_frame:={a.report_frame}',
            '-p', 'base_frame:=base_link',
            '-p', f'wholebody_urdf:={urdf}',
            '-p', f'fix_base:={"true" if a.fix_base else "false"}',
            '-p', f'control_rate:={a.rate}'])

        spawn('arm_vel_adapter',
              [sys.executable, os.path.join(HERE, 'arm_vel_adapter.py')])
        spawn('arm_vel_gate',
              [sys.executable, os.path.join(HERE, 'arm_vel_gate.py'),
               '--timeout', str(a.gate_timeout)])

        spawn('viz_barrier_live', [
            sys.executable, os.path.join(HERE, 'viz_barrier_live.py'),
            '--ros-args',
            '-p', 'use_sim_time:=true',
            '-p', f'report_frame:={a.report_frame}',
            '-p', f'wholebody_urdf:={urdf}',
            '-p', f'world_sdf:={a.world}',
            '-p', f'indep_n:={a.indep_n}'])

        if a.camera_topic:
            spawn('demo_cam_bridge', [
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                f'{a.camera_topic}@sensor_msgs/msg/Image[gz.msgs.Image',
                '--ros-args', '-p', 'use_sim_time:=true'])

        if not a.no_foxglove:
            spawn('foxglove_bridge', [
                'ros2', 'run', 'foxglove_bridge', 'foxglove_bridge',
                '--ros-args', '-p', 'port:=8765',
                '-p', 'use_sim_time:=true'])

        print('\n堆疊已啟動。Foxglove: ws://localhost:8765')
        print('Ctrl-C 結束全部。手臂命令唯一發布端是 arm_vel_gate。\n', flush=True)
        while True:
            time.sleep(1.0)
            for label, p in procs:
                if p.poll() is not None:
                    print(f'!! {label} 已結束，回傳碼 {p.returncode}',
                          file=sys.stderr, flush=True)
                    raise SystemExit(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print('\n收拾中…', flush=True)
        # Gate last: while anything upstream is still alive it must keep
        # publishing, and its own shutdown leaves a zero command behind.
        def sig(p, s):
            try:
                os.killpg(os.getpgid(p.pid), s)
            except (ProcessLookupError, PermissionError):
                pass

        for label, p in reversed(procs):
            if p.poll() is None:
                sig(p, signal.SIGINT)
        t0 = time.time()
        for label, p in reversed(procs):
            try:
                p.wait(timeout=max(0.5, 8.0 - (time.time() - t0)))
            except subprocess.TimeoutExpired:
                print(f'   {label} 未回應 SIGINT，改用 SIGKILL', file=sys.stderr)
                sig(p, signal.SIGKILL)
                try:
                    p.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass
        # Nothing from this stack may outlive it. A survivor keeps publishing on
        # the same topics as the next run and silently doubles every feed.
        left = []
        for label, p in procs:
            try:
                os.killpg(os.getpgid(p.pid), 0)
                left.append(label)
            except (ProcessLookupError, PermissionError):
                pass
        if left:
            print(f'   !! 這些程序群組仍在: {left}', file=sys.stderr)
        else:
            print('   所有程序群組已結束')
        try:
            os.unlink(urdf)
        except OSError:
            pass
        print('已停止。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
