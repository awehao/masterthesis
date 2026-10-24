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
   '#   紀錄 room_run finger_effort：啟動讀回、旗標值、暫停切換時的 old／set／readback／ok 與定期讀回（door_k3m_finger_effort）。需 --park-gate-mobile。\n')
ia("import door_k3m_corridor as K3MCOR                               # noqa: E402  EV3：走廊（純函式）\n",
   "import door_k3m_finger_effort as FEFF                            # noqa: E402  EV5：暫停期手指驅動輸出上限（純邏輯）\n")
ib("ap.add_argument('--headless', default='true')\n",
   "ap.add_argument('--pause-finger-max-effort', type=float, default=0.0,             # EV5\n"
   "                help='（EV5）暫停切換那一步起兩指有效驅動輸出上限（N）；0＝不改（URDF effort 5）')\n")
ia("    print(f'[room] 手臂收攏於 {[round(v, 6) for v in stow]}', flush=True)\n",
   "    _fe_idx = [fidx[j] for j in FJ]                                                   # EV5：啟動讀回（不改值）\n"
   "    fin_eff_init = FEFF.read_init(robot.get_articulation_controller(), _fe_idx, FJ)\n"
   "    fin_eff = FEFF.PauseFingerEffort(a.pause_finger_max_effort, _fe_idx, FJ)\n"
   "    print(f'[room] 手指驅動輸出上限讀回 {fin_eff_init[\"readback\"]}（{FJ}）；暫停期設定 {a.pause_finger_max_effort or \"不改\"}', flush=True)\n"
   "    if fin_eff.enabled and (not a.park_gate_mobile or len(_fe_idx) != 2):\n"
   "        print('[room] **--pause-finger-max-effort 需要 --park-gate-mobile 與兩隻手指關節** ⇒ 中止', flush=True)\n"
   "        return 37\n")
ib("        # ---- 真正寫進 API ----\n",
   "        if fin_eff.record is not None:                                               # EV5：切換後每 1 s 讀回核對（不改值）\n"
   "            fin_eff.periodic(robot.get_articulation_controller(), step_id, t)\n")
ia("                        print(f'[door] **K3-M 暫停切換**（權威）@ step {step_id}', flush=True)\n",
   "                        if fin_eff.enabled:                                          # EV5：同一步、world.step 之前\n"
   "                            _fer = fin_eff.on_pause_switch(robot.get_articulation_controller(), step_id, t)\n"
   "                            print(f'[door] **W1e 手指驅動輸出上限** {_fer[\"old\"]} → {_fer[\"set\"]}，讀回 {_fer[\"readback\"]}'\n"
   "                                  f'（{\"生效\" if _fer[\"ok\"] else \"**讀回不符**\"}）@ step {step_id}', flush=True)\n")
ib("    if a.cmd_env:                                                                   # EV4：封裝與套用來源統計（含 n_plain_after_env）\n",
   "    rec['finger_effort'] = {'init': fin_eff_init, 'arg_pause_max_effort': a.pause_finger_max_effort,   # EV5\n"
   "                            'pause': fin_eff.record, 'tol': FEFF.TOL}\n")
open('isaac_door_k2_sim_ev5.py', 'w').write(out)
a_, b_ = src.splitlines(), out.splitlines()
j = 0
for ln in a_:
    while b_[j] != ln:
        j += 1
    j += 1
print('insert-only OK', len(a_), '→', len(b_))
