"""Record a simulator camera topic to an mp4.

Why not screen capture
----------------------
x11grab records whatever is composited at a screen region, not a window's own
contents. On this machine the Gazebo window is routinely behind something else,
and two attempts at recording the demo captured the user's browser instead --
personal data written into the repository. A camera sensor placed in the world
can only ever see the simulated scene, so this is the recording path.

Frames are piped raw into ffmpeg rather than written as PNGs and assembled
afterwards: the intermediate files would be the same content on disk twice, and
the pipe keeps the recording bounded to this process's lifetime.

    python3 evaluation/record_gz_camera.py --topic /demo_cam --out out.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Recorder(Node):

    def __init__(self, topic, out, fps):
        super().__init__('gz_camera_recorder')
        self.out, self.fps = out, fps
        self.proc = None
        self.n = 0
        self.create_subscription(Image, topic, self._on_img,
                                 qos_profile_sensor_data)

    def _on_img(self, m: Image) -> None:
        if self.proc is None:
            if m.encoding not in ('rgb8', 'bgr8'):
                self.get_logger().error(f'不支援的編碼 {m.encoding}')
                return
            pix = 'rgb24' if m.encoding == 'rgb8' else 'bgr24'
            self.proc = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                 '-pix_fmt', pix, '-s', f'{m.width}x{m.height}',
                 '-framerate', str(self.fps), '-i', '-',
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                 '-pix_fmt', 'yuv420p', self.out],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            self.get_logger().info(
                f'錄影 {m.width}x{m.height} {m.encoding} → {self.out}')
        try:
            self.proc.stdin.write(bytes(m.data))
            self.n += 1
        except (BrokenPipeError, ValueError):
            pass

    def close(self):
        if self.proc is None:
            return 0
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=20)
        except Exception:                                       # noqa: BLE001
            self.proc.kill()
        return self.n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/demo_cam')
    ap.add_argument('--out', default='evaluation/results/pregrasp_demo.mp4')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--seconds', type=float, default=0.0,
                    help='stop after this long; 0 means run until SIGINT')
    a = ap.parse_args()
    rclpy.init()
    r = Recorder(a.topic, a.out, a.fps)
    t0 = time.monotonic()
    try:
        while a.seconds <= 0 or time.monotonic() - t0 < a.seconds:
            rclpy.spin_once(r, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        n = r.close()
        print(f'  {n} 幀寫入 {a.out}', flush=True)
        r.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
