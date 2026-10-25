"""isaac_door_k2_sim_ev5.py → isaac_door_k2_sim_ev6.py（Codex 20261025_214144：ev6＋W2 模式權威接線）。**只插入**。
核對：原檔每一行依序全在新檔中（insert-only）。
"""
src = open('isaac_door_k2_sim_ev5.py').read()
out = src


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


ib('# EV5 版（W1e：暫停期提高手指驅動輸出上限',
   '# EV6 版（W2 保持／待交接的執行端模式權威；Codex 20261025_214144；規格 W2b_exec_wiring_spec.md）：isaac_door_k2_sim_ev5.py 的複本，**只做插入**：\n'
   '#   --w2-mode-run-id（預設空 ⇒ 與 EV5 同行為；需 --cmd-env、--run-id）：\n'
   '#   (1) 唯一入口改為 v3 封裝 /wb_vel_cmd_env3（cmd_envelope_v3，含 mode_seq）；v2 /wb_vel_cmd_env 收到只計數、不進鏈。\n'
   '#       回呼只解碼並排入有界佇列（另存接收牆鐘，不覆寫來源 stamp_sim_t）；每物理步在主迴圈依序交 door_w2_exec.W2Exec.on_env3。\n'
   '#   (2) 模式請求：任務端 /door/w2_mode_req_task → ModeAuthority3.on_request；登記為 pending 後才轉送求解端 /door/w2_mode_req（latched）。\n'
   '#   (3) 每物理步 select（先核請求期限、以原始 stamp 判新鮮）→ EnvIntake 進 E2 鏈（鏈接收時刻＝來源 stamp，K2 過濾照舊）→ ex.step\n'
   '#       → applied_source／限幅後九軸 → confirm_step；權威中止、入口違規 ⇒ 既有 K2 閂鎖中止（有界停止、不回切、不張爪）。\n'
   '#   (4) 每 2 物理步發布權威狀態 /door/w2_mode_state；room_run.json 另記 w2（權威統計、逐步列、權威確認列、接收紀錄）。\n'
   '#   測試模式 RELEASE_PREP_TEST 與 HELD 命令相同（不張爪、不退離、不增加角度）；速度、加速度、輪級、關節限制與 92 °C 熱中止不變。\n')
ia("import door_k3m_finger_effort as FEFF                            # noqa: E402  EV5：暫停期手指驅動輸出上限（純邏輯）\n",
   "import cmd_envelope_v3 as E3                                     # noqa: E402  EV6：v3 封裝（mode_seq）\n"
   "import door_w2_mode3 as W2M3                                     # noqa: E402  EV6：模式權威 r3\n"
   "import door_w2_exec as W2X                                       # noqa: E402  EV6：執行端接線（與假世界共用）\n"
   "import collections as _w2_col                                    # noqa: E402  EV6\n")
ib("ap.add_argument('--headless', default='true')\n",
   "ap.add_argument('--w2-mode-run-id', default='',                                    # EV6\n"
   "                help='（EV6）W2 模式權威 run_id；空＝關閉（EV5 行為）。需 --cmd-env、--run-id')\n"
   "ap.add_argument('--w2-cand-timeout-s', type=float, default=1.0)                    # EV6：候選期限（設計值，不改）\n"
   "ap.add_argument('--w2-post-commit-timeout-s', type=float, default=0.25)            # EV6\n")
ia("        if a.cmd_env:                                                              # EV4：求解端命令只經封裝入口\n"
   "            self.create_subscription(Float64MultiArray, ENV.TOPIC[ENV.ST_ADAPTER], self._counted(self._wb_env), 10)\n"
   "            self.applied_src_pub = self.create_publisher(Float64MultiArray, '/door/applied_src', 100)\n",
   "        if a.w2_mode_run_id:                                                       # EV6：v3 唯一入口、請求轉送、權威狀態\n"
   "            self.w2_inbox = _w2_col.deque()\n"
   "            self.w2_inbox_cap = 512\n"
   "            self.w2_n_overflow = self.w2_n_bad3 = self.w2_n_v2_ignored = 0\n"
   "            self.w2_req_q = []\n"
   "            self.create_subscription(Float64MultiArray, E3.TOPIC3[ENV.ST_ADAPTER], self._counted(self._wb_env3), 50)\n"
   "            _lat_w2 = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)\n"
   "            self.create_subscription(String, '/door/w2_mode_req_task', self._counted(self._w2_req), _lat_w2)\n"
   "            self.w2_req_pub = self.create_publisher(String, '/door/w2_mode_req', _lat_w2)\n"
   "            self.w2_state_pub = self.create_publisher(String, '/door/w2_mode_state', 50)\n")
ia("    def _wb_env(self, m: Float64MultiArray):                                        # EV4：封裝（身分＋九維值）\n",
   "        if a.w2_mode_run_id:                                                       # EV6：v2 不進鏈（唯一入口＝v3），只計數\n"
   "            self.w2_n_v2_ignored += 1\n"
   "            return\n")
ib("    def _ho(self, m: String):\n",
   "    def _wb_env3(self, m: Float64MultiArray):                                       # EV6：只解碼排隊；接收牆鐘另存\n"
   "        try:\n"
   "            e = E3.decode3(m.data)\n"
   "        except ValueError:\n"
   "            self.w2_n_bad3 += 1\n"
   "            return\n"
   "        if len(self.w2_inbox) >= self.w2_inbox_cap:\n"
   "            self.w2_n_overflow += 1\n"
   "            return\n"
   "        self.w2_inbox.append((e, _ev_time.monotonic()))\n"
   "\n"
   "    def _w2_req(self, m: String):                                                   # EV6\n"
   "        try:\n"
   "            self.w2_req_q.append(json.loads(m.data))\n"
   "        except Exception:\n"
   "            self.w2_req_q.append({'_unparsable': str(m.data)[:200]})\n"
   "\n")
ia("    env_stats = {'n_applied_src_reports': 0}\n",
   "    w2x = None                                                                      # EV6\n"
   "    if a.w2_mode_run_id:\n"
   "        if not a.cmd_env or not a.run_id:\n"
   "            print('[door] **--w2-mode-run-id 需要 --cmd-env 與 --run-id** ⇒ 中止', flush=True)\n"
   "            sys.exit(41)\n"
   "        w2_auth = W2M3.ModeAuthority3(a.w2_mode_run_id, run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER,\n"
   "                                      cand_timeout_s=a.w2_cand_timeout_s, cmd_max_age_s=a.max_cmd_age_s,\n"
   "                                      post_commit_timeout_s=a.w2_post_commit_timeout_s, dt=a.physics_dt,\n"
   "                                      a_lin=a.base_accel_lin, a_ang=a.base_accel_ang)\n"
   "        w2x = W2X.W2Exec(w2_auth, env_in, filt=lambda u: k2_latch.filter_wb(u))\n"
   "        w2_rt = {'n_viol_seen': 0, 'abort_handled': False, 'requests': [], 'relayed': []}\n")
ib("        # 交棒請求\n        if node.ho_req is not None:\n",
   "        _w2sel = None                                                               # EV6：v3 入口 → 權威 → 進鏈\n"
   "        if w2x is not None:\n"
   "            while node.w2_inbox:\n"
   "                _e6, _wall6 = node.w2_inbox.popleft()\n"
   "                w2x.on_env3(_e6, t, _wall6)\n"
   "            if len(w2_auth.violations) > w2_rt['n_viol_seen']:\n"
   "                _v6 = w2_auth.violations[w2_rt['n_viol_seen']]\n"
   "                w2_rt['n_viol_seen'] = len(w2_auth.violations)\n"
   "                _ev4_violation(f\"w2_entry:{_v6['why']}\")\n"
   "            while node.w2_req_q:\n"
   "                _r6 = node.w2_req_q.pop(0)\n"
   "                _k6, _i6 = w2_auth.on_request(_r6, step_id, t)\n"
   "                w2_rt['requests'].append({'step': step_id, 't': t, 'req': _r6, 'result': _k6})\n"
   "                if _k6 == 'pending':                                                # 權威登記後才轉送求解端\n"
   "                    node.w2_req_pub.publish(String(data=json.dumps(_r6, default=float)))\n"
   "                    w2_rt['relayed'].append({'step': step_id, 't': t, 'req_seq': _r6.get('req_seq')})\n"
   "            _w2sel = w2x.pre_step(step_id, t, chain)\n"
   "            if isinstance(_w2sel.get('fed'), str) and _w2sel['fed'].startswith('violation:'):\n"
   "                _ev4_violation(_w2sel['fed'].split(':', 1)[1])\n"
   "\n")
ib("        if fin_eff.record is not None:                                               # EV5：切換後每 1 s 讀回核對（不改值）\n",
   "        if w2x is not None:                                                         # EV6：鏈的實際來源與限幅後九軸 → 確認\n"
   "            w2x.post_step(step_id, t, chain, res.owner == AUTH_WHOLEBODY,\n"
   "                          [float(x) for x in res.base_cmd] + [float(x) for x in res.u_applied[3:]], _w2sel)\n"
   "            if w2_auth.aborted is not None and not w2_rt['abort_handled']:\n"
   "                w2_rt['abort_handled'] = True\n"
   "                _ev4_violation(f\"w2_mode:{w2_auth.aborted['why']}\")\n"
   "            if step_id % 2 == 0:\n"
   "                node.w2_state_pub.publish(String(data=json.dumps(w2_auth.state_msg(step_id, t), default=float)))\n")
ib("    out = os.path.join(a.out, 'room_run.json')\n    json.dump(rec, open(out, 'w'), ensure_ascii=False, indent=1, default=str)\n",
   "    if w2x is not None:                                                             # EV6\n"
   "        rec['w2'] = {'auth': w2_auth.stats(), 'rows': w2x.rows, 'auth_rows': w2_auth.applied_rows, 'recv_log': w2x.recv_log[:20000],\n"
   "                     'intake_acts': w2x.intake_acts, 'feed': w2x.feed, 'runtime': w2_rt,\n"
   "                     'n_v2_ignored': node.w2_n_v2_ignored, 'n_bad_v3': node.w2_n_bad3, 'n_inbox_overflow': node.w2_n_overflow,\n"
   "                     'cols': {'recv_log': ['source_seq', 'mode_seq', 'stamp_sim_t', 't_recv_sim', 't_recv_wall_mono', 'act']},\n"
   "                     'semantics': '鏈接收時刻＝來源 stamp_sim_t；src／mode_seq／chain_mode／u 由鏈讀出（applied_source），非指定'}\n")
open('isaac_door_k2_sim_ev6.py', 'w').write(out)

a_, b_ = src.splitlines(), out.splitlines()
j = 0
for line in a_:
    while j < len(b_) and b_[j] != line:
        j += 1
    assert j < len(b_), f'原行遺失：{line!r}'
    j += 1
print('insert-only ok：', len(a_), '→', len(b_))
