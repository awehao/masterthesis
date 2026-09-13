"""在**事前宣告的模擬時刻**切斷本趟 adapter —— 接收端斷訊試驗。

安全規則（與本專案既有約束一致）：

* 只對 **runner 記錄的那一個 PID** 動作，PID 由參數傳入，**不做字串搜尋比對**
* 動作前先核對 `/proc/<pid>/cmdline` 確實是預期的 adapter（防 PID 重用），
  **不符就放棄、不殺**
* 每個子程序由 runner 以 `setsid` 起動，process group 專屬於它自己，
  因此送給該 group 不會波及其他工作階段

切斷時刻**事前固定**為「命令源第一則訊息的模擬時間 + --offset-s」，
與趟次結果無關，不是看到什麼再決定。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--pid', type=int, required=True, help='本趟 adapter 的 PID')
ap.add_argument('--expect-cmdline', default='arm_vel_adapter.py',
                help='核對用；不符即放棄，不殺')
ap.add_argument('--offset-s', type=float, required=True,
                help='自命令源第一則訊息起算的模擬時間偏移')
ap.add_argument('--out', required=True)
ap.add_argument('--timeout-s', type=float, default=180.0)
a = ap.parse_args()

import rclpy                                                   # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from std_msgs.msg import Float64MultiArray                     # noqa: E402
from rosgraph_msgs.msg import Clock                            # noqa: E402

CMD_IN = '/wholebody_safety/cmd_in'


def cmdline(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().decode('utf-8', 'replace').split('\x00')
    except OSError:
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Cut(Node):
    def __init__(self):
        super().__init__('wb_cut_adapter')
        self.sim_t = None
        self.t_first = None
        self.create_subscription(Clock, '/clock', self._clk, 10)
        self.create_subscription(Float64MultiArray, CMD_IN, self._cmd, 50)

    def _clk(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9

    def _cmd(self, _m):
        if self.t_first is None and self.sim_t is not None:
            self.t_first = self.sim_t
            print(f'[cut] 命令源第一則 sim {self.t_first:.3f}；'
                  f'預定切斷 sim {self.t_first + a.offset_s:.3f}', flush=True)


def main():
    os.makedirs(a.out, exist_ok=True)
    rec = {'schema': 'wb_cut_adapter/1', 'pid': a.pid,
           'offset_s': a.offset_s, 'aborted': None}

    cl = cmdline(a.pid)
    if cl is None:
        rec['aborted'] = f'PID {a.pid} 不存在'
    elif not any(a.expect_cmdline in x for x in cl):
        rec['aborted'] = (f'PID {a.pid} 的 cmdline 不含 {a.expect_cmdline!r}'
                          f'（實際 {cl[:3]}）—— **放棄，不殺**')
    if rec['aborted']:
        print(f"[cut] **{rec['aborted']}**", flush=True)
        json.dump(rec, open(os.path.join(a.out, 'cut_adapter.json'), 'w'),
                  ensure_ascii=False)
        return 7
    rec['cmdline_verified'] = ' '.join(x for x in cl if x)[:200]
    print(f'[cut] PID {a.pid} 身分核對通過', flush=True)

    rclpy.init()
    n = Cut()
    w0 = time.monotonic()
    done = False
    while rclpy.ok() and time.monotonic() - w0 < a.timeout_s and not done:
        rclpy.spin_once(n, timeout_sec=0.02)
        if n.t_first is not None and n.sim_t is not None \
                and n.sim_t >= n.t_first + a.offset_s:
            rec['cut_sim_t'] = round(n.sim_t, 4)
            rec['first_cmd_sim_t'] = round(n.t_first, 4)
            rec['planned_cut_sim_t'] = round(n.t_first + a.offset_s, 4)
            # 只送給這一個 PID 專屬的 process group（runner 以 setsid 起動）
            try:
                os.killpg(os.getpgid(a.pid), signal.SIGTERM)
                rec['signal'] = 'SIGTERM->pgid'
            except OSError as e:
                os.kill(a.pid, signal.SIGTERM)
                rec['signal'] = f'SIGTERM->pid（pgid 取得失敗：{e}）'
            print(f"[cut] **已切斷 adapter** @ sim {rec['cut_sim_t']:.3f}",
                  flush=True)
            done = True
    if not done:
        rec['aborted'] = '等到逾時仍未達到切斷時刻'
        print(f"[cut] **{rec['aborted']}**", flush=True)
        json.dump(rec, open(os.path.join(a.out, 'cut_adapter.json'), 'w'),
                  ensure_ascii=False)
        rclpy.try_shutdown()
        return 8

    # 確認真的結束（不以「送了訊號」代替「已停止」）
    t1 = time.monotonic()
    while time.monotonic() - t1 < 10.0 and alive(a.pid):
        rclpy.spin_once(n, timeout_sec=0.05)
    rec['exited'] = not alive(a.pid)
    rec['exit_wait_s'] = round(time.monotonic() - t1, 3)
    rec['sim_t_after_exit'] = round(n.sim_t, 4) if n.sim_t else None
    print(f"[cut] adapter 已結束={rec['exited']}（等待 {rec['exit_wait_s']}s）",
          flush=True)
    json.dump(rec, open(os.path.join(a.out, 'cut_adapter.json'), 'w'),
              ensure_ascii=False)
    rclpy.try_shutdown()
    return 0 if rec['exited'] else 9


sys.exit(main())
