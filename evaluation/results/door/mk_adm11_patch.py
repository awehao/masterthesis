"""ADM9 → ADM11（W2 HELD／模式來源；Codex 20261025_201543）：只插入（每個錨點恰好一次；核對 ADM9 每行依序在新檔中）。"""
src = open('dl3_wgmpc_wg2_adm9_node.py').read()
out = src


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


ib('# ADM9（W1c：夾持期間偏移估計保持；',
   '# ADM11（W2 保持／待交接；Codex 20261025_201543，設計 W2_hold_handover_design_draft2.md，規格 W2a_held_mode_spec.md）：dl3_wgmpc_wg2_adm9_node.py 的版本化複本，**只做插入**：\n'
   '#   --w2-held（需 --pause-active-hold 與 --pause-arm-win-s > 0；預設關 ⇒ 與 ADM9 同行為）：監督器改 door_w2_mode.HeldSupervisorP4b——\n'
   '#   觀察通過記 OBS_PASSED 並進 HELD，持續發布保持命令與監看（不以觀察通過停發）；HELD 超過 --w2-held-max-s ⇒ HELD_TIMEOUT_END 後 DONE。\n'
   '#   --w2-mode-run-id（需 --w2-held 與 --cmd-env）：door_w2_mode.SolverModeSource；訂閱 /door/w2_mode_req（只在 HELD 接受、準備新模式首筆）與\n'
   '#   /door/w2_mode_state（執行端權威狀態，ack 遺失時亦以此為準）；每筆發布另以 v3 封裝（cmd_envelope_v3，同一 source_seq＋mode_seq）發到\n'
   '#   /wgmpc/cmd_env3。本包只有測試模式 RELEASE_PREP_TEST（命令與 HELD 相同，不張爪、不退離）。既有 v2 封裝與話題不變。\n')
ia('import door_k3m_pause as PZ                                      # noqa: E402  ADM4：decel_step（可行集合）\n',
   'import door_w2_mode as W2M                                       # noqa: E402  ADM11：HELD／模式來源\n'
   'import cmd_envelope_v3 as E3                                     # noqa: E402  ADM11：v3 封裝（mode_seq）\n')
ia("                           if self._pause_sw is not None else None)                        # ADM7\n",
   "        if getattr(a, 'w2_held', False) and self._pause_sw is not None:                # ADM11：HELD 監督器（門檻不變）\n"
   "            self._pause_sup = W2M.HeldSupervisorP4b(arm_win_s=a.pause_arm_win_s, held_max_s=a.w2_held_max_s)\n"
   "        self._w2src = (W2M.SolverModeSource(str(a.w2_mode_run_id)) if getattr(a, 'w2_mode_run_id', '') else None)   # ADM11\n"
   "        self._w2_pub = {'n_by_mode': {}, 'first_src_by_mode': {}, 'last': None}\n")
ia("            self.create_subscription(Float64MultiArray, '/door/applied_src', self._on_applied_src, 200)\n",
   "            if self._w2src is not None:                                                # ADM11：模式請求與執行端權威狀態\n"
   "                self.create_subscription(String, '/door/w2_mode_req', self._on_w2_req, _lat4)\n"
   "                self.create_subscription(String, '/door/w2_mode_state', self._on_w2_state, 50)\n"
   "                self._env3_pub = self.create_publisher(Float64MultiArray, E3.TOPIC3[ENV.ST_SOLVER], 10)\n")
ib("    def _pause_status(self, slot, note=None):                                         # ADM4\n",
   "    def _on_w2_req(self, m):                                                          # ADM11：只在 HELD 接受\n"
   "        try:\n"
   "            msg = json.loads(m.data)\n"
   "        except ValueError:\n"
   "            msg = None\n"
   "        self._w2src.on_request(msg, held=(self._pause_sup is not None and self._pause_sup.state == 'HELD'), t=self.sim_now())\n"
   "\n"
   "    def _on_w2_state(self, m):                                                        # ADM11：權威狀態為準\n"
   "        try:\n"
   "            self._w2src.on_state(json.loads(m.data), self.sim_now())\n"
   "        except ValueError:\n"
   "            pass\n"
   "\n")
ia("""            self.env_pub.publish(_em)
""", """            if self._w2src is not None:                                                # ADM11：同一 source_seq 另發 v3（mode_seq 綁定）
                _ms11 = int(self._w2src.tag())
                _e3 = Float64MultiArray()
                _e3.data = E3.encode3(self._run_id_n, self._source_seq, ENV.ST_SOLVER, self._source_seq,
                                      self._source_seq, True, _t, u_world, _ms11)
                self._env3_pub.publish(_e3)
                self._w2_pub['n_by_mode'][_ms11] = self._w2_pub['n_by_mode'].get(_ms11, 0) + 1
                self._w2_pub['first_src_by_mode'].setdefault(_ms11, int(self._source_seq))
                self._w2_pub['last'] = (int(self._source_seq), _ms11, float(_t))
""")
ia("                    offset_hold=self._of_hold,                                     # ADM9：凍結邊界紀錄\n",
   "                    w2=(None if self._w2src is None else                                # ADM11：模式來源紀錄\n"
   "                        {'events': self._w2src.events[:200], 'committed': self._w2src.committed,\n"
   "                         'n_by_mode': {str(k): v for k, v in self._w2_pub['n_by_mode'].items()},\n"
   "                         'first_src_by_mode': {str(k): v for k, v in self._w2_pub['first_src_by_mode'].items()},\n"
   "                         'last_pub': self._w2_pub['last']}),\n")
ib("    ap.add_argument('--park-uprev-tol', type=float, default=1e-6,\n",
   "    ap.add_argument('--w2-held', action='store_true',\n"
   "                    help='ADM11：觀察通過後進 HELD 持續保持發布與監看（需 --pause-active-hold 與 --pause-arm-win-s > 0）')\n"
   "    ap.add_argument('--w2-held-max-s', type=float, default=60.0, help='ADM11：HELD 有界超時（s）')\n"
   "    ap.add_argument('--w2-mode-run-id', default='', help='ADM11：模式來源 run_id（需 --w2-held 與 --cmd-env）')\n")
ib("    rclpy.init()\n    nd = WGMPCNode(a)\n",
   "    if a.w2_held and not (a.pause_active_hold and a.pause_arm_win_s > 0):\n"
   "        ap.error('--w2-held 需要 --pause-active-hold 與 --pause-arm-win-s > 0')\n"
   "    if a.w2_mode_run_id and not (a.w2_held and a.cmd_env):\n"
   "        ap.error('--w2-mode-run-id 需要 --w2-held 與 --cmd-env')\n")
open('dl3_wgmpc_wg2_adm11_node.py', 'w').write(out)
a_, b_ = src.splitlines(), out.splitlines()
j = 0
for ln in a_:
    while b_[j] != ln:
        j += 1
    j += 1
print('insert-only OK', len(a_), '→', len(b_))
