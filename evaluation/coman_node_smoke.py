"""節點啟動冒煙測試：用 **runner 的實際參數** 起每個節點，確認它活得過初始化。

為什麼需要：離線套件全部通過、入口核對全部通過，仍然有兩類錯只在真正
`rclpy.init()` 與 `Node.__init__()` 時才會爆：

  * 參數字串格式錯（`obstacles` 每條規格含逗號，未逐項加引號 ⇒ RCLError）
  * 回呼方法不存在（`create_subscription(..., self._on_src_meta)` 而該方法
    因為替換字串沒對上而**從未被加進類別** ⇒ AttributeError）

兩者都讓節點在啟動瞬間死掉，而 runner 只會在 180 s 後以 /clock 逾時收場。

**不啟動 Isaac**：每個節點起 N 秒後送 SIGTERM，看是否在該期間內自行退出。
"""
from __future__ import annotations
import os, re, shlex, signal, subprocess, sys, textwrap, time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
RUNNER = os.path.join(HERE, 'run_coman_drawer20.sh')


def runner_vars():
    """取 runner 的預設參數值（**不執行它**）：只抽取變數指派那幾行後 eval。"""
    keys = ('PAIR_GAP', 'PAIR_D0', 'CONTACT_PAIRS', 'PAIR_ROWS',
            'PAIR_ROWS_EXEMPT', 'TIGHT_PAIRS', 'TIGHT_TOL', 'TIGHT_BUDGET',
            'URDF_TF', 'URDF_WB', 'STROKE')
    lines = [l.rstrip('\n') for l in open(RUNNER, encoding='utf-8')
             if re.match(r'^(' + '|'.join(keys) + r')=', l)]
    script = (f'WS={shlex.quote(WS)}\n' + '\n'.join(lines) + '\n'
              + '\n'.join(f'printf "%s\\t%s\\n" {k} "${k}"' for k in keys))
    out = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    return dict(l.split('\t', 1) for l in out.stdout.strip().split('\n')
                if '\t' in l)


def quoted(csv):
    return '[' + ','.join(f'"{x}"' for x in csv.split(',') if x.strip()) + ']'


class FakeClock:
    """冒煙測試期間發 /clock。

    **沒有時鐘，計時器就不會觸發** —— 節點過得了 `__init__` 卻從未跑過一個
    週期，`_tick` 裡的錯照不到（實測：`d.data += [...]` 的 TypeError 讓距離
    節點在第一個 tick 就死，冒煙測試卻回報「存活」）。
    """

    def __init__(self):
        self.p = subprocess.Popen(
            [sys.executable, '-c', textwrap.dedent("""
                import rclpy
                from rclpy.node import Node
                from rosgraph_msgs.msg import Clock
                rclpy.init()
                n = Node('smoke_clock')
                pub = n.create_publisher(Clock, '/clock', 10)
                t = [0.0]
                def tick():
                    t[0] += 0.01
                    m = Clock()
                    m.clock.sec = int(t[0])
                    m.clock.nanosec = int((t[0] - int(t[0])) * 1e9)
                    pub.publish(m)
                n.create_timer(0.01, tick)
                rclpy.spin(n)
            """)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        time.sleep(1.5)

    def stop(self):
        try:
            os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
            self.p.wait(timeout=5)
        except Exception:                                     # noqa: BLE001
            self.p.kill()


def smoke(name, argv, seconds=6.0):
    """起節點 seconds 秒。**提早退出＝失敗**。"""
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < seconds:
        if p.poll() is not None:
            out = p.stdout.read()
            return False, out[-1500:]
        time.sleep(0.2)
    # **殺整個行程群組**：`ros2 run` 只是外殼，真正的節點是它的子程序。
    # 先前只送 SIGTERM 給外殼 ⇒ 節點活下來，一次冒煙測試漏三個程序。
    # start_new_session=True 讓本程序自成一個 session，所以 killpg 只會影響
    # **本函式自己起的那一組**，不會碰到其他人的程序。
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        p.kill()
    return True, ''


def main() -> int:
    v = runner_vars()
    if not v.get('URDF_WB'):
        print('取不到 runner 變數', file=sys.stderr)
        return 2
    obs = subprocess.run([sys.executable,
                          os.path.join(HERE, 'coman_obstacle_specs.py')],
                         capture_output=True, text=True, check=True)
    obs_list = '[' + ','.join(f'"{x}"' for x in obs.stdout.split()
                              if x.strip()) + ']'
    jobs = [
        ('arm_link_distance',
         ['ros2', 'run', 'ammr_wholebody_mpc', 'arm_link_distance', '--ros-args',
          '-p', 'use_sim_time:=true', '-p', 'report_frame:=odom',
          '-p', 'geometry:=links', '-p', f'wholebody_urdf:={v["URDF_WB"]}',
          '-p', f'obstacles:={obs_list}',
          '-p', f'pair_rows:={quoted(v["PAIR_ROWS"])}',
          '-p', f'pair_rows_exempt:={quoted(v["PAIR_ROWS_EXEMPT"])}',
          '-p', f'tight_pairs:={quoted(v["TIGHT_PAIRS"])}',
          '-p', f'tight_tol:={v["TIGHT_TOL"]}',
          '-p', f'tight_budget_s:={v["TIGHT_BUDGET"]}'], 25.0),
        ('wholebody_safety',
         ['ros2', 'run', 'ammr_wholebody_mpc', 'wholebody_safety', '--ros-args',
          '-p', 'use_sim_time:=true', '-p', 'report_frame:=odom',
          '-p', 'base_frame:=base_link',
          '-p', f'wholebody_urdf:={v["URDF_WB"]}',
          '-p', 'freespace_confirmed:=false',
          '-p', f'pair_gap:={quoted(v["PAIR_GAP"])}',
          '-p', f'contact_pairs:={quoted(v["CONTACT_PAIRS"])}'], 8.0),
        ('robot_state_publisher',
         ['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher',
          v['URDF_TF'], '--ros-args', '-p', 'use_sim_time:=true'], 5.0),
        ('coman_diag_record',
         [sys.executable, '-u', os.path.join(HERE, 'coman_diag_record.py'),
          '--out', '/tmp/_smoke_diag.json'], 5.0),
    ]
    clk = FakeClock()          # 讓計時器真的觸發，才照得到 _tick 裡的錯
    bad = 0
    for name, argv, secs in jobs:
        ok, tail = smoke(name, argv, secs)
        print(f'  {name:26s}{"存活" if ok else "**啟動即死**"}')
        if not ok:
            bad += 1
            for ln in tail.strip().split('\n')[-8:]:
                print(f'      {ln}')
    clk.stop()
    print('節點冒煙測試：' + ('全部存活' if bad == 0 else f'**{bad} 個啟動即死**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
