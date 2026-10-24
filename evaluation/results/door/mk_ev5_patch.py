src = open('isaac_door_k2_sim_ev4.py').read()
out = src
def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)
def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)
ib('# EV4 版（K3-M 受控暫停接入；',
   '# EV5 版（W1e：暫停期提高手指驅動輸出上限；Codex 20261025_023608，規格 W1e_pause_finger_effort_spec.md）：isaac_door_k2_sim_ev4.py 的複本，**只做插入**：\n'
   '#   --pause-finger-max-effort N（預設 0 ⇒ 與 EV4 同行為，只多記一筆啟動讀回）：執行端暫停切換那一步（world.step 之前）把兩指 finger_joint1／2 的\n'
   '#   有效驅動輸出上限（ArticulationController.set_max_efforts）設為 N 並讀回，之後每 1 s 讀回核對；閉合目標、kp／kd、摩擦、手臂／底盤控制與判準不變。\n'
   '#   紀錄 room_run finger_effort：啟動讀回、旗標值、暫停切換時的 old／set／readback／ok 與定期讀回（door_k3m_finger_effort）。需 --park-gate-mobile。\n'
   '#   r2（Codex 20261025_024339 必修①③）：啟動讀回缺失／非法／不是兩指各 --pause-finger-original-effort（5）⇒ 啟用時拒跑（38）；\n'
   '#   切換設定／讀回失敗、非有限、不符或非手指上限被改 ⇒ 保存原因、走既有 K2 中止閂鎖（/door/k3m_abort），暫停 ack 改回覆拒絕；\n'
   '#   定期讀回失敗或被改回 ⇒ 同樣中止。不自動重試、不加力、不放爪；原停止與保護門檻不變。切換前後記完整 DOF 上限向量與名稱。\n')
ia("import door_k3m_corridor as K3MCOR                               # noqa: E402  EV3：走廊（純函式）\n",
   "import door_k3m_finger_effort as FEFF                            # noqa: E402  EV5：暫停期手指驅動輸出上限（純邏輯）\n")
ib("ap.add_argument('--headless', default='true')\n",
   "ap.add_argument('--pause-finger-max-effort', type=float, default=0.0,             # EV5\n"
   "                help='（EV5）暫停切換那一步起兩指有效驅動輸出上限（N）；0＝不改（URDF effort 5）')\n"
   "ap.add_argument('--pause-finger-original-effort', type=float, default=5.0,        # EV5 r2\n"
   "                help='（EV5）啟動讀回應有的兩指原上限（URDF effort）；啟用時不符即拒跑')\n")
ia("    print(f'[room] 手臂收攏於 {[round(v, 6) for v in stow]}', flush=True)\n",
   "    _fe_idx = [fidx[j] for j in FJ]                                                   # EV5：啟動讀回（不改值）\n"
   "    fin_eff_init = FEFF.read_init(robot.get_articulation_controller(), _fe_idx, FJ, names, a.pause_finger_original_effort)\n"
   "    fin_eff = FEFF.PauseFingerEffort(a.pause_finger_max_effort, _fe_idx, FJ)\n"
   "    print(f'[room] 手指驅動輸出上限讀回 {fin_eff_init[\"readback\"]}（{FJ}）；暫停期設定 {a.pause_finger_max_effort or \"不改\"}', flush=True)\n"
   "    if fin_eff.enabled and (not a.park_gate_mobile or len(_fe_idx) != 2):\n"
   "        print('[room] **--pause-finger-max-effort 需要 --park-gate-mobile 與兩隻手指關節** ⇒ 中止', flush=True)\n"
   "        return 37\n"
   "    if fin_eff.enabled and not fin_eff_init['ok']:\n"
   "        print(f'[room] **手指驅動上限啟動讀回不合格**（{fin_eff_init[\"why\"]}）⇒ 拒跑', flush=True)\n"
   "        return 38\n")
ib("    while rclpy.ok():\n",
   "    def _w1e_abort(_why5, _step5, _t5):                                             # EV5 r2：W1e 讀回異常 ⇒ 既有 K2 中止閂鎖\n"
   "        rec['events'].append({'sim_t': _t5, 'step': _step5, 'w1e_finger_effort_abort': _why5})\n"
   "        if k2_latch.latched is None:\n"
   "            _gnow5 = (a.finger_open if node.grip_cmd is None else min(max(float(node.grip_cmd), a.finger_closed), a.finger_open))\n"
   "            k2_latch.on_msg(f'w1e_finger_effort：{_why5}', _step5, _t5, _gnow5)\n"
   "            k2_latch.expire_chain(chain)\n"
   "            ex.cancel_handover(f'w1e_finger_effort：{_why5}')\n"
   "            node.k3m_abort_pub.publish(String(data=json.dumps({'why': f'w1e_finger_effort：{_why5}', 't': _t5,\n"
   "                                                               'source': 'executor_w1e'}, default=float)))\n"
   "        print(f'[door] **W1e 手指驅動上限異常 ⇒ 中止閂鎖**（{_why5}）@ step {_step5}', flush=True)\n"
   "\n")
ib("        # ---- 真正寫進 API ----\n",
   "        if fin_eff.record is not None:                                               # EV5：切換後每 1 s 讀回核對（不改值）\n"
   "            _pe5 = fin_eff.periodic(robot.get_articulation_controller(), step_id, t)\n"
   "            if _pe5 is not None and _pe5['ok'] is not True:\n"
   "                _w1e_abort(f'periodic：{_pe5[\"why\"]}', step_id, t)\n")
ia("                        print(f'[door] **K3-M 暫停切換**（權威）@ step {step_id}', flush=True)\n",
   "                        if fin_eff.enabled:                                          # EV5：同一步、world.step 之前\n"
   "                            _fer = fin_eff.on_pause_switch(robot.get_articulation_controller(), step_id, t)\n"
   "                            print(f'[door] **W1e 手指驅動輸出上限** {_fer[\"old\"]} → {_fer[\"set\"]}，讀回 {_fer[\"readback\"]}'\n"
   "                                  f'（{\"生效\" if _fer[\"ok\"] else \"**讀回不符**\"}）@ step {step_id}', flush=True)\n"
   "                            if _fer['ok'] is not True:                               # EV5 r2：不回覆正常暫停 ack\n"
   "                                _w1e_abort(f'switch：{_fer[\"why\"]}', step_id, t)\n"
   "                                k3m['pause_switch_w1e_failed'] = dict(_ackp, why=_fer['why'])\n"
   "                                _kp, _ackp = 'reject', {'run_id': _ackp.get('run_id'), 'seq': _ackp.get('seq'),\n"
   "                                                        'why': f'w1e_finger_effort：{_fer[\"why\"]}', 'step': step_id}\n")
ib("    if a.cmd_env:                                                                   # EV4：封裝與套用來源統計（含 n_plain_after_env）\n",
   "    rec['finger_effort'] = {'init': fin_eff_init, 'arg_pause_max_effort': a.pause_finger_max_effort,   # EV5\n"
   "                            'arg_original_effort': a.pause_finger_original_effort,\n"
   "                            'pause': fin_eff.record, 'pause_switch_failed': k3m.get('pause_switch_w1e_failed'), 'tol': FEFF.TOL}\n")
open('isaac_door_k2_sim_ev5.py', 'w').write(out)
a_, b_ = src.splitlines(), out.splitlines()
j = 0
for ln in a_:
    while b_[j] != ln:
        j += 1
    j += 1
print('insert-only OK', len(a_), '→', len(b_))
