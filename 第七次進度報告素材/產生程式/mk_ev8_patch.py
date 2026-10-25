"""EV7r2 → EV8（K3-B 閉爪卸載候選整合；Codex 20261026_025647）：只插入（每個錨點恰好一次；核對 ev7r2 每行依序在新檔中）。
--w2-unload（需 --w2-release）預設關 ⇒ 與 ev7r2 同行為。
"""
src = open('isaac_door_k2_sim_ev7r2.py').read()
out = src


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


ib('# EV7 r2 版（Codex 20261026_010158）：',
   '# EV8 版（K3-B 閉爪卸載候選整合；Codex 20261026_025647；規格 K3B_unload_spec_draft1.md、K3B_unload_integ_r1.md）：ev7r2 的版本化複本，**只做插入**。\n'
   '#   --w2-unload（需 --w2-release；預設關 ⇒ 與 ev7r2 同行為）：\n'
   '#   (1) 權威改 door_w2_mode8.ModeAuthority8（unload＝True、gate2＝ExecUnloadMonitor.confirmed_fresh；缺 gate2 建構即拒）。\n'
   '#   (2) mode 1 提交步 start 卸載監看；每物理步 world.step 之前：既有中止（W2 權威或 K2 閂鎖）優先 ⇒ abort、不設定；\n'
   '#       否則依登錄斜坡 set_max_efforts 兩指並當步讀回（door_k3b_unload_ctrl.set_ramp；非手指上限不變；定期讀回參考值隨之更新），失敗走既有中止。\n'
   '#   (3) 本步接觸、門角、底盤、手臂與讀回 → ExecUnloadMonitor.step（同一物理步資料）；mode 2 提交步 handoff。\n'
   '#   (4) 權威狀態附 unload 摘要與最近一次定期讀回（finger_effort）；卸載監看 FAILED 為已知中止。\n'
   '#   (5) 卸載模式不受理 5 N 恢復請求（不偽造恢復動作）；room_run w2.unload／w2.unload_ctrl 紀錄。\n')
ia('import door_k3b_grip_gate as K3BG                                # noqa: E402  EV7 r2：手指命令閘門（中止不重新閉爪）\n',
   'import door_k3b_unload2 as K3BU                                  # noqa: E402  EV8：閉爪卸載監看 r2\n'
   'import door_k3b_unload_ctrl as K3BUC                             # noqa: E402  EV8：卸載斜坡設定與讀回\n'
   'import door_w2_mode8 as W2M8                                     # noqa: E402  EV8：權威 r8（gate2）\n')
ib("ap.add_argument('--headless', default='true')\n",
   "ap.add_argument('--w2-unload', action='store_true',                                # EV8\n"
   "                help='（EV8）K3-B 閉爪卸載候選：mode 1 提交起 20→5 N 斜坡、低速確認後才准 mode 2（需 --w2-release）')\n")
ia("    w2_xm = w2_effa = w2_grip = None                                                # EV7\n",
   "    w2_xu, w2_xu_ctrl, w2_xu_step = None, [], {'cur': None}                          # EV8\n")
ib("        w2_rt = {'abort_handled': False}\n",
   "        if getattr(a, 'w2_unload', False):                                          # EV8：權威 r8＋卸載監看\n"
   "            if not a.w2_release:\n"
   "                print('[door] **--w2-unload 需要 --w2-release** ⇒ 中止', flush=True)\n"
   "                sys.exit(43)\n"
   "            w2_xu = K3BU.ExecUnloadMonitor(reg={'dt': a.physics_dt})\n"
   "            w2_auth = W2M8.ModeAuthority8(a.w2_mode_run_id, run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER,\n"
   "                                          cand_timeout_s=a.w2_cand_timeout_s, cmd_max_age_s=a.max_cmd_age_s,\n"
   "                                          post_commit_timeout_s=a.w2_post_commit_timeout_s, dt=a.physics_dt,\n"
   "                                          a_lin=a.base_accel_lin, a_ang=a.base_accel_ang, arm_rate_max=a.arm_rate_max,\n"
   "                                          gate3=lambda _s7: w2_xm.confirmed_fresh(_s7), gate2=lambda _s8: w2_xu.confirmed_fresh(_s8), unload=True)\n"
   "            w2x = W2X3.W2Exec3(w2_auth, env_in, filt=lambda u: k2_latch.filter_wb(u))\n"
   "\n"
   "            def _fe_last8():\n"
   "                _p8 = ((fin_eff.record or {}).get('periodic') or [])\n"
   "                return None if not _p8 else {'readback': _p8[-1].get('readback'), 't': _p8[-1].get('t'), 'ok': _p8[-1].get('ok')}\n"
   "            w2_auth.extra_state = lambda: {'release': {'state': w2_xm.state, 'why': w2_xm.why, 't_open': w2_xm.t.get('open'),\n"
   "                                                       't_conf': w2_xm.t.get('B_CONFIRMED'), 'step_conf': w2_xm.step_conf,\n"
   "                                                       'last_valid_step': w2_xm.last_valid_step},\n"
   "                                           'effort': w2_effa.ack(), 'unload': w2_xu.state_brief(), 'finger_effort': _fe_last8()}\n")
ib("        if w2_xm is not None and w2_xm.state == 'FAILED':                               # EV7：解除監看失敗＝已知中止\n",
   "        if w2_xu is not None and w2_xu.state == 'FAILED':                               # EV8：卸載監看失敗＝已知中止\n"
   "            return f\"unload_monitor:{w2_xu.why}\"\n")
ia("                w2_xm.open(step_id, t)                                                  # EV7：t_open＝mode 2 提交步\n",
   "            if w2_xu is not None and (_w2sel.get('event') or {}).get('kind') == 'commit':    # EV8：mode 1 起卸載；mode 2 交給釋放監看\n"
   "                if w2_auth.mode_seq == 1:\n"
   "                    w2_xu.start(step_id, t)\n"
   "                elif w2_auth.mode_seq == 2:\n"
   "                    w2_xu.handoff(step_id, t)\n")
ia("                    _r7 = node.w2_eff_q.pop(0)\n",
   "                    if w2_xu is not None:                                                    # EV8：卸載模式不受理 5 N 恢復（不偽造恢復動作）\n"
   "                        _r7 = {'run_id': None, 'unload_mode_rejected': _r7}\n")
ib("        if fin_eff.record is not None:                                               # EV5：切換後每 1 s 讀回核對（不改值）\n",
   "        w2_xu_step['cur'] = None                                                     # EV8：卸載斜坡（world.step 之前；中止優先）\n"
   "        if w2_xu is not None and w2_xu.state in ('RAMPING', 'SETTLING', 'UNLOADED'):\n"
   "            _ka8 = _w2_known_abort()\n"
   "            if w2_auth.aborted is not None or k2_latch.latched is not None or _ka8 is not None:\n"
   "                w2_xu.abort(step_id, t, f\"w2_or_k2_abort:{_ka8 or (w2_auth.aborted or {}).get('why')}\")\n"
   "            else:\n"
   "                _fu8 = w2_xu.command(t)\n"
   "                if _fu8 is not None:\n"
   "                    _ru8 = K3BUC.set_ramp(fin_eff, robot.get_articulation_controller(), _fu8, step_id, t)\n"
   "                    if len(w2_xu_ctrl) < 2000 and (not _ru8['ok'] or step_id % 10 == 0 or abs(_fu8 - 5.0) < 1e-9):\n"
   "                        w2_xu_ctrl.append(_ru8)\n"
   "                    w2_xu_step['cur'] = (_fu8, _ru8['readback'])\n"
   "                    if not _ru8['ok']:\n"
   "                        _ev4_violation(f\"w2_unload_set:{_ru8['why']}\")\n")
ia("            w2_xm.step(step_id, t, _fq7, _pf7, _lc is not None, door_q=float(_qd_rec))\n",
   "        if w2_xu is not None and w2_xu.state in ('RAMPING', 'SETTLING', 'UNLOADED') and w2_xu_step['cur'] is not None:   # EV8：同一物理步資料\n"
   "            _pfu8 = (None if _lc is None else [float(_lc.get(_n8, {}).get('bar', 0.0)) + float(_lc.get(_n8, {}).get('ambiguous', 0.0))\n"
   "                                               for _n8 in ('uflite_finger1', 'uflite_finger2')])\n"
   "            w2_xu.step(step_id, t, w2_xu_step['cur'][1], w2_xu_step['cur'][0], _pfu8, _lc is not None, float(_qd_rec),\n"
   "                       [float(p_now[0]), float(p_now[1]), float(yw)], [float(x) for x in q_arm])\n")
ia("        rec['w2']['grip_gate'] = w2_grip.report()\n",
   "        if w2_xu is not None:                                                           # EV8\n"
   "            rec['w2']['unload'] = w2_xu.report()\n"
   "            rec['w2']['unload_ctrl'] = w2_xu_ctrl\n")
open('isaac_door_k2_sim_ev8.py', 'w').write(out)

a_, b_ = src.splitlines(), out.splitlines()
j = 0
for line in a_:
    while j < len(b_) and b_[j] != line:
        j += 1
    assert j < len(b_), f'原行遺失：{line!r}'
    j += 1
print('insert-only ok：', len(a_), '→', len(b_))
