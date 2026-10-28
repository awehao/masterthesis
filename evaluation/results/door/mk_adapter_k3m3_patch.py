"""ADAPTER k3m2（台架就緒）＋ w2（v3 轉送），祖先 k3m，三方合併 → k3m3（Codex r64）。重跑：python3 results/door/mk_adapter_k3m3_patch.py [--check]
衝突 3 處：檔頭（兩段並存）、Adapter 建構簽名（兩個新參數並存）、命令列與建構呼叫（兩旗標並存）。
語義修正 1 處（RG）：k3m2 的就緒回報以 v2 封裝入口／出口計數，v3 模式下 env_pub 為 None ⇒ 改為依實際啟用的路徑計數
  （v3：輸入＝求解端 v3 話題的發布者數、輸出＝env3_pub 的訂閱者數；判準與期限不變），計時器在 v2 或 v3 封裝任一啟用時建立。
"""
import subprocess
import sys

m = subprocess.run(['git', 'merge-file', '-p', 'arm_vel_adapter_k3m2.py', 'arm_vel_adapter_k3m.py', 'arm_vel_adapter_w2.py'], capture_output=True, text=True)
assert m.returncode == 3, m.returncode
s = m.stdout


def take(s):
    a = s.index('<<<<<<< ')
    b = s.index('\n=======\n', a)
    c = s.index('>>>>>>> ', b)
    return s[:a], s[s.index('\n', a) + 1:b + 1], s[b + 9:c], s[s.index('\n', c) + 1:]


pre, ours, theirs, rest = take(s)
assert theirs.endswith('"""Thin adapter: the safety filter\'s 9-vector to the arm\'s 6 joint velocities.\n')
hdr = ('# ADAPTER k3m3（Codex r64）：k3m2（台架就緒回報）與 w2（--cmd-env-v3）以 k3m 為祖先三方合併；就緒回報依實際啟用的封裝路徑計數（RG 標記）。\n'
       + theirs[:theirs.index('"""Thin adapter')])
s = pre + hdr + ours + rest
pre, ours, theirs, rest = take(s)
s = pre + "                 in_topic='/wholebody_safety/cmd_out', env_in_topic=None, bench_ready_run_id='', cmd_env_v3=False):\n" + rest
pre, ours, theirs, rest = take(s)
s = (pre + "    ap.add_argument('--bench-ready-run-id', default='', help='k3m2：台架就緒回報（本趟 run_id；預設關）')\n"
     + theirs.replace("cmd_env_v3=a.cmd_env_v3)", "bench_ready_run_id=a.bench_ready_run_id, cmd_env_v3=a.cmd_env_v3)") + rest)
assert '<<<<<<<' not in s and '>>>>>>>' not in s


def rep(o, n):
    global s
    assert s.count(o) == 1, o
    s = s.replace(o, n)


rep("""        self._env_in = (env_in_topic or ENV.TOPIC[ENV.ST_SAFETY])
        self._ready_seq = 0
        if self.bench_run and self.env_on:
""", """        self._env_in = (E3.TOPIC3[ENV.ST_SOLVER] if self.v3 else (env_in_topic or ENV.TOPIC[ENV.ST_SAFETY]))   # RG：依實際入口
        self._ready_seq = 0
        if self.bench_run and (self.env_on or self.v3):                                  # RG：v2 或 v3 封裝任一啟用
""")
rep("""        n_in, n_out = self.count_publishers(self._env_in), self.env_pub.get_subscription_count()
""", """        n_in = self.count_publishers(self._env_in)
        n_out = (self.env3_pub if self.v3 else self.env_pub).get_subscription_count()   # RG：依實際出口
""")
if '--check' in sys.argv:
    cur = open('arm_vel_adapter_k3m3.py').read()
    print('identical' if cur == s else 'DIFFERENT')
    sys.exit(0 if cur == s else 1)
open('arm_vel_adapter_k3m3.py', 'w').write(s)
print('written')
