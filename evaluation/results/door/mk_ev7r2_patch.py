"""isaac_door_k2_sim_ev6r3.py → isaac_door_k2_sim_ev7r2.py（K3-B 受控釋放執行端 r2；Codex 20261026_010158；mk_ev7_patch.py〔ev7 r1，凍結 k3b r1〕不改。以下為 r1 說明：Codex 20261026_000706 D1／D3／D4；設計 K3B_impl_design_draft1.md）。
以 ev6r3（mk_ev6r3_patch.py 的輸出，凍結 w2c r2）為底，**只插入**；並核對 ev5 每行依序仍在 ev7 中（對 ev5 也是 insert-only）。
--w2-release 預設關 ⇒ 與 ev6r3 同行為。
"""
base = open('isaac_door_k2_sim_ev6r3.py').read()
ev5 = open('isaac_door_k2_sim_ev5.py').read()
out = base


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


ib('# EV6 r3 版（Codex 20261025_223632；',
   '# EV7 r2 版（Codex 20261026_010158）：手指命令閘門（中止不重新閉爪）、core r3（5 N tick 重核）。以下為 r1 說明：\n'
   '# EV7 版（K3-B 受控釋放執行端；Codex 20261026_000706；設計 K3B_impl_design_draft1.md，規格 K3B_release_clear_spec_draft2.md）：ev6r3 的複本，**只做插入**。\n'
   '#   --w2-release（需 --w2-mode-run-id 與 --pause-finger-max-effort > 0；預設關 ⇒ 與 ev6r3 同行為）：\n'
   '#   (1) 權威改 door_w2_mode5.ModeAuthority5：模式 0→1→2→3；mode 3 准入＝執行端 ExecReleaseMonitor.confirmed_fresh（請求時與提交當下）。\n'
   '#   (2) mode 2 提交步＝t_open：同一步依權威模式切換手指目標為張開（finger_target）；mode ≥ 1 時手指命令由權威模式決定，任務端夾爪命令只作參考；\n'
   '#       K2 閂鎖後照既有凍結（不張爪、不閉爪）。\n'
   '#   (3) 逐物理步 ExecReleaseMonitor（door_k3b_release2）：本步兩指開度、兩指—橫桿接觸（bar＋ambiguous）、接觸資料有效、門角；\n'
   '#       失敗 ⇒ 已知中止（release_monitor:<why>）⇒ 權威中止＋既有 K2 閂鎖路徑。\n'
   '#   (4) 5 N 恢復權威（door_k3b_release2.EffortRestoreAuthority）：/door/finger_effort_req（序號、mode 3、任務端 C_CLEAR 證據）→ 執行端以逐步資料\n'
   '#       再核（mode 3、確認後靜窗 ≥ 0.5 s、開度 ≥ 95 %、期限 0.5 s）→ 物理步 door_k3b_finger_restore.restore（兩指 20→5 N、讀回、非手指不變）→\n'
   '#       /door/finger_effort_ack；讀回失敗走既有中止；中止優先。\n'
   '#   (5) 權威狀態另帶 release（狀態、t_conf、步）與 effort（ack）；room_run.json w2 另記 release、effort。\n'
   '#   時序：解除監看用的兩指開度與接觸力都在本步 world.step **之前**讀取（即上一物理步結算後的狀態；與 room 紀錄列同一組值）。\n'
   '#   r2（Codex 20261026_010158）：手指命令改由 door_k3b_grip_gate.GripGate——閂鎖後凍結為最後實際下達值（不重新閉爪）；\n'
   '#       5 N 權威於 tick 重核證據年齡與解除資料新鮮度、拒收未來時間證據（door_k3b_release2 r2）。\n')
ia("import door_w2_exec2 as W2X2                                     # noqa: E402  EV6 r2：執行端接線 r2（與假世界共用）\n",
   "import door_w2_mode5 as W2M5                                     # noqa: E402  EV7：多段模式權威\n"
   "import door_k3b_release3 as K3BR                                 # noqa: E402  EV7 r2：解除監看與 5 N 恢復權威（core r3）\n"
   "import door_k3b_finger_restore as K3BF                           # noqa: E402  EV7：手指上限恢復\n"
   "import door_w2_exec3 as W2X3                                     # noqa: E402  EV7：逐步列記手指目標\n"
   "import door_k3b_grip_gate as K3BG                                # noqa: E402  EV7 r2：手指命令閘門（中止不重新閉爪）\n")
ia("ap.add_argument('--w2-post-commit-timeout-s', type=float, default=0.25)            # EV6\n",
   "ap.add_argument('--w2-release', action='store_true',                               # EV7\n"
   "                help='（EV7）K3-B 受控釋放：模式 2／3、逐步解除監看、5 N 恢復權威（需 --w2-mode-run-id 與 --pause-finger-max-effort）')\n")
ia("            self.w2_state_pub = self.create_publisher(String, '/door/w2_mode_state', 50)\n",
   "            self.w2_eff_q = []                                                         # EV7：5 N 恢復請求\n"
   "            self.create_subscription(String, '/door/finger_effort_req', self._counted(self._w2_eff_req), _lat_w2)\n"
   "            self.w2_eff_ack_pub = self.create_publisher(String, '/door/finger_effort_ack', _lat_w2)\n")
ib("    def _w2_req(self, m: String):                                                   # EV6\n",
   "    def _w2_eff_req(self, m: String):                                               # EV7\n"
   "        try:\n"
   "            self.w2_eff_q.append(json.loads(m.data))\n"
   "        except Exception:\n"
   "            self.w2_eff_q.append({'_unparsable': str(m.data)[:200]})\n"
   "\n")
ia("    w2x = None                                                                      # EV6\n",
   "    w2_xm = w2_effa = w2_grip = None                                                # EV7\n")
ia("        w2x = W2X2.W2Exec2(w2_auth, env_in, filt=lambda u: k2_latch.filter_wb(u))\n",
   "        if a.w2_release:                                                            # EV7：權威 r5＋逐步解除監看＋5 N 恢復權威\n"
   "            if not fin_eff.enabled:\n"
   "                print('[door] **--w2-release 需要 --pause-finger-max-effort > 0** ⇒ 中止', flush=True)\n"
   "                sys.exit(42)\n"
   "            w2_xm = K3BR.ExecReleaseMonitor(reg={'dt': a.physics_dt})\n"
   "            w2_auth = W2M5.ModeAuthority5(a.w2_mode_run_id, run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER,\n"
   "                                          cand_timeout_s=a.w2_cand_timeout_s, cmd_max_age_s=a.max_cmd_age_s,\n"
   "                                          post_commit_timeout_s=a.w2_post_commit_timeout_s, dt=a.physics_dt,\n"
   "                                          a_lin=a.base_accel_lin, a_ang=a.base_accel_ang, arm_rate_max=a.arm_rate_max,\n"
   "                                          gate3=lambda _s7: w2_xm.confirmed_fresh(_s7))\n"
   "            w2x = W2X3.W2Exec3(w2_auth, env_in, filt=lambda u: k2_latch.filter_wb(u))\n"
   "            w2_effa = K3BR.EffortRestoreAuthority(a.w2_mode_run_id)\n"
   "            w2_grip = K3BG.GripGate(a.finger_open, a.finger_closed)\n"
   "            w2_auth.extra_state = lambda: {'release': {'state': w2_xm.state, 'why': w2_xm.why, 't_open': w2_xm.t.get('open'),\n"
   "                                                       't_conf': w2_xm.t.get('B_CONFIRMED'), 'step_conf': w2_xm.step_conf,\n"
   "                                                       'last_valid_step': w2_xm.last_valid_step},\n"
   "                                           'effort': w2_effa.ack()}\n")
ia("""            return f"k3m_violation:{(k3m['violation'] or {}).get('why')}"\n""",
   """        if w2_xm is not None and w2_xm.state == 'FAILED':                               # EV7：解除監看失敗＝已知中止\n"""
   """            return f"release_monitor:{w2_xm.why}"\n""")
ia("            _w2sel, _relay6, _nv6 = w2x.step_begin(step_id, t, chain, known_abort=_w2_known_abort(), inbox=_ib6, requests=_rq6)\n",
   "            if w2_xm is not None and (_w2sel.get('event') or {}).get('kind') == 'commit' and w2_auth.mode_seq == 2:\n"
   "                w2_xm.open(step_id, t)                                                  # EV7：t_open＝mode 2 提交步\n")
ia("            w2x.sync_after(step_id, t, known_abort=_w2_known_abort())                # 本步 select 之後才發現的既有中止\n",
   "            if w2_effa is not None:                                                     # EV7：5 N 恢復權威（中止優先）\n"
   "                if w2_auth.aborted is not None or k2_latch.latched is not None:\n"
   "                    w2_effa.abort('w2_or_k2_abort', step_id, t)\n"
   "                while node.w2_eff_q:\n"
   "                    _r7 = node.w2_eff_q.pop(0)\n"
   "                    _k7 = w2_effa.request(_r7, step_id, t, w2_auth.mode_seq)\n"
   "                    node.w2_eff_ack_pub.publish(String(data=json.dumps(dict(w2_effa.ack(), result=_k7, step=step_id, t=t), default=float)))\n"
   "                _fq7b = [float(q_meas[fidx[j]]) for j in FJ] if FJ else [0.0, 0.0]\n"
   "                _o7 = w2_effa.tick(step_id, t, w2_xm, w2_auth.mode_seq, min(_fq7b) >= 0.95 * a.finger_open)\n"
   "                if _o7 == 'apply':\n"
   "                    _rr7 = K3BF.restore(fin_eff, robot.get_articulation_controller(), a.pause_finger_original_effort, step_id, t)\n"
   "                    w2_effa.confirm(_rr7['ok'], step_id, t)\n"
   "                    if not _rr7['ok']:\n"
   "                        _ev4_violation(f\"w2_effort_restore:{_rr7['why']}\")\n"
   "                if _o7 is not None:\n"
   "                    node.w2_eff_ack_pub.publish(String(data=json.dumps(dict(w2_effa.ack(), result=_o7, step=step_id, t=t), default=float)))\n")
ia("        _fcmd = k2_latch.grip(a.finger_open if node.grip_cmd is None\n"
   "                              else min(max(float(node.grip_cmd), a.finger_closed),\n"
   "                                       a.finger_open))\n",
   "        if w2_grip is not None:                                                     # EV7 r2：權威模式決定張閉；閂鎖後凍結最後實際下達值\n"
   "            _fcmd = w2_grip.command(step_id, t, k2_latch, w2_auth.mode_seq, w2_auth.finger_target(), _fcmd)\n")
ib("        # 發布已完成逐步驗收的同時刻資料；不在 world.step 後重讀並混用時間。\n",
   "        if w2_xm is not None and w2_xm.state not in ('IDLE', 'FAILED'):                 # EV7：逐物理步解除監看（本步接觸）\n"
   "            _fq7 = [float(q_meas[fidx[j]]) for j in FJ] if FJ else None\n"
   "            _pf7 = (None if _lc is None else [float(_lc.get(_n7, {}).get('bar', 0.0)) + float(_lc.get(_n7, {}).get('ambiguous', 0.0))\n"
   "                                              for _n7 in ('uflite_finger1', 'uflite_finger2')])\n"
   "            w2_xm.step(step_id, t, _fq7, _pf7, _lc is not None, door_q=float(_qd_rec))\n")
ia("                     'semantics': '鏈接收時刻＝來源 stamp_sim_t；src／mode_seq／chain_mode／u 由鏈讀出（applied_source），非指定'}\n",
   "    if w2x is not None and w2_xm is not None:                                       # EV7\n"
   "        rec['w2']['release'] = w2_xm.report()\n"
   "        rec['w2']['effort'] = {'ack': w2_effa.ack(), 'events': w2_effa.events[:100], 'restore': (fin_eff.record or {}).get('restore')}\n"
   "        rec['w2']['grip_gate'] = w2_grip.report()\n")
open('isaac_door_k2_sim_ev7r2.py', 'w').write(out)

for name, src in (('ev6r3', base), ('ev5', ev5)):
    a_, b_ = src.splitlines(), out.splitlines()
    j = 0
    for line in a_:
        while j < len(b_) and b_[j] != line:
            j += 1
        assert j < len(b_), f'{name} 原行遺失：{line!r}'
        j += 1
    print(f'insert-only ok（對 {name}）：', len(a_), '→', len(b_))
