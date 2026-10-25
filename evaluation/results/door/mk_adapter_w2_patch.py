"""arm_vel_adapter_k3m.py → arm_vel_adapter_w2.py（Codex 20261025_201543：mode_seq 與 source_seq 一起穿過 adapter）。
刪改一處：初始化的入口選擇——--cmd-env-v3 時**只**訂 v3 求解端封裝、只發 v3（不訂 v2、不訂舊九維，維持唯一控制入口）；其餘只插入。
核對：原檔除被替換的入口區塊外，其餘行依序全在新檔中。
"""
src = open('arm_vel_adapter_k3m.py').read()
out = src


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


OLD = """        self.env_on = bool(cmd_env)
        if self.env_on:
"""
NEW = """        self.env_on = bool(cmd_env)
        self.v3 = bool(cmd_env_v3)                                                     # W2：v3 時只走 v3（唯一控制入口）
        self.n_v3_out = self.n_v3_mode_mismatch = 0
        if self.v3:
            self.env_pub = None
            self.env3_pub = self.create_publisher(Float64MultiArray, E3.TOPIC3[ENV.ST_ADAPTER], 10)
            self.create_subscription(Float64MultiArray, E3.TOPIC3[ENV.ST_SOLVER], self.on_env3, 10)
            self.pub = None
        elif self.env_on:
"""
assert out.count(OLD) == 1
out = out.replace(OLD, NEW)
ib('"""Thin adapter: the safety filter',
   '# W2 版（arm_vel_adapter_k3m.py 的版本化複本；Codex 20261025_201543）：--cmd-env-v3 ⇒ 只訂 /wgmpc/cmd_env3、只發 /wb_vel_cmd_env3（cmd_envelope_v3，19 欄）；\n'
   '#   轉送時 **mode_seq 原樣保留**（與 source_seq 同封裝），轉換與檢查同 on_env；不訂 v2、不訂舊九維。預設關 ⇒ 與 k3m 版同行為。\n')
ia("import wgmpc_cmd_envelope as ENV                                  # noqa: E402\n",
   "import cmd_envelope_v3 as E3                                     # noqa: E402  W2：v3 封裝\n")
OLD2 = """    def __init__(self, ctrl=CTRL_DEFAULT, cmd_env=False,
                 in_topic='/wholebody_safety/cmd_out', env_in_topic=None):
"""
assert out.count(OLD2) == 1
out = out.replace(OLD2, """    def __init__(self, ctrl=CTRL_DEFAULT, cmd_env=False,
                 in_topic='/wholebody_safety/cmd_out', env_in_topic=None, cmd_env_v3=False):
""")
ib("    def on_odom(self, m):\n",
   """    def on_env3(self, m):
        \"\"\"W2：v3 封裝路徑。檢查與轉換同 on_env；**mode_seq 原樣轉送**（不改寫、不推測）。\"\"\"
        self.n_in += 1
        try:
            e = E3.decode3(m.data)
        except ValueError:
            self.n_bad += 1
            return
        d = np.asarray(e['u'], dtype=float)
        if len(d) < 9 or not np.all(np.isfinite(d[:9])) or not self.order_ok:
            self.n_bad += 1
            return
        if self.yaw is None or time.monotonic() - self.yaw_t > 0.3:
            self.n_bad += 1
            return
        v = world_to_body9(d, self.yaw)
        self._out_seq += 1
        om = Float64MultiArray()
        om.data = E3.encode3(e['run_id'], e['source_seq'], ENV.ST_ADAPTER, self._out_seq, e['output_seq'], e['derived'],
                             self.get_clock().now().nanoseconds * 1e-9, v, e['mode_seq'], kind=e['kind'])
        if E3.decode3(om.data)['mode_seq'] != e['mode_seq']:
            self.n_v3_mode_mismatch += 1
            return
        self.env3_pub.publish(om)
        self.n_out += 1
        self.n_v3_out += 1
        self.last_in = time.monotonic()

""")
ia("""    ap.add_argument('--env-in-topic', default=None,
""", "")
OLD3 = """    nd = Adapter(a.consumer_node, cmd_env=a.cmd_env,
                 in_topic=a.in_topic, env_in_topic=a.env_in_topic)
"""
assert out.count(OLD3) == 1
out = out.replace(OLD3, """    nd = Adapter(a.consumer_node, cmd_env=a.cmd_env,
                 in_topic=a.in_topic, env_in_topic=a.env_in_topic, cmd_env_v3=a.cmd_env_v3)
""")
ib("    a, _ = ap.parse_known_args()\n",
   "    ap.add_argument('--cmd-env-v3', action='store_true',\n"
   "                    help='（W2）只訂 /wgmpc/cmd_env3、只發 /wb_vel_cmd_env3；mode_seq 原樣轉送')\n")
open('arm_vel_adapter_w2.py', 'w').write(out)
removed = set(OLD.splitlines()) | set(OLD2.splitlines()) | set(OLD3.splitlines())
a_, b_ = src.splitlines(), out.splitlines()
j = 0
for ln in a_:
    if ln in removed and ln not in b_:
        continue
    while b_[j] != ln:
        j += 1
    j += 1
print('OK', len(a_), '→', len(b_))
