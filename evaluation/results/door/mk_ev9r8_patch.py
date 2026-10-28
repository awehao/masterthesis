"""EV9r7 ＋ EV8（祖先 EV5）三方合併 → EV9r8 起點，再插入 RG8（Codex r63／r64）。重跑：python3 results/door/mk_ev9r8_patch.py [--check]
合併：git merge-file -p ev9r7 ev5 ev8；唯一衝突＝檔頭說明，兩段並存。RG8 變更以「前文＋原文 → 前文＋新文」逐一替換，每個錨點恰好一次。
"""
import json
import subprocess
import sys

m = subprocess.run(['git', 'merge-file', '-p', 'isaac_door_k2_sim_ev9r7.py', 'isaac_door_k2_sim_ev5.py', 'isaac_door_k2_sim_ev8.py'],
                   capture_output=True, text=True)
assert m.returncode == 1, m.returncode                               # 恰 1 處衝突
s = m.stdout
a, b, c = s.index('<<<<<<< '), s.index('\n=======\n'), s.index('>>>>>>> ')
ours, theirs, tail = s[s.index('\n', a) + 1:b + 1], s[b + 9:c], s[s.index('\n', c) + 1:]
HDR = "# EV9 r8（Codex r63／r64）：ev9r7（台架初態線，祖先 ev5）與 ev8（W2／K3-B 線，祖先 ev5）以 git merge-file 三方合併（兩線皆自 ev5「只插入」；\n#   唯一文字衝突＝檔頭說明，兩段並存如下）；其上只插入 --regrasp（見 RG8 標記）。--regrasp 未給 ⇒ 回歸基準＝ev9r7 接觸資料流＋原控制行為。\n#   RG8（--regrasp，需 --w2-mode-run-id、--bench-start、--contact-v3-out、--regrasp-ref；與 --w2-release／--w2-unload 互斥，不符 ⇒ 拒跑 66）：\n#   (1) 啟動核：參考包 door_regrasp_ref_registry.validate；門規格＝登錄核驗所用 door_local_v3.yaml（sha256）；門根變換＝單位（參考包座標＝門局部座標）。\n#   (2) 權威改 door_regrasp_mode3.RegraspAuthority（mode 1–8：換抓＋通過中拉關＋放手退出；gate 2–8；W2Exec3、GripGate 同 K3-B）；\n#       gate 2–6 與 HoldConfirm 由 door_regrasp_exec_core.RegraspExecCore（gate 2＝Gate2StopObs：訂閱 /wgmpc/pause_status 取 ZERO_PUB 為窗起點，\n#       逐物理步以判定器 P2／P3 同一套程式核停止與證據覆蓋；不合格只使 mode 2 不准入）\n#       每物理步以**同一份 v3 快照分類紀錄**驅動（與假執行端共用類別）；提交當步（world.step 之前）on_commit 啟動／結束 gate；\n#       gate 狀態經權威 extra_state 'gates' 發給任務端；v3 接觸窗口摘要每 2 步發 /door/contact_v3_summary。\n#   (3) 換抓中止原因（接觸違規、資料失效 > 0.2 s、gate／短保持 FAILED）⇒ 既有 K2 閂鎖（鏈內命令過期、取消交棒、/door/k3m_abort）。\n#   (4) 台架首筆：W2 唯一入口＝v3，凍結期以收件匣 mode_seq 0 的序號判斷；釋放時只把宣告那一筆經 W2Exec.on_env3→pre_step 進鏈（bench_release_w2），\n#       較新者留在收件匣；第 1 物理步不消化收件匣（首筆固定，同 v2 路徑）。\n#   (5) mode 3 提交步 k 起 K3-M 走廊／包絡監看不再判（最後判定於 k−1；k3m.regrasp_end＝core.handover 的 last_old_step／first_new_step），\n#       換抓保護自 k 起接手（提交當下須有有效 v3 證據，否則中止）；不關閉其他保護。\n#   **修改既有敘述 4 處**（其餘只插入）：收件匣 while 條件（首筆固定）、走廊監看 if 條件（交接後不再判）、台架 release_intake 呼叫改為 RG8／原路徑二選一、\n#       首步 step_begin 改為 RG8 時沿用釋放當下的 select 結果（聯測 iii 發現：步 0 二次 select 未經 confirm ⇒ 權威 confirm_missing 中止）。\n# ---- 以下為 ev9r7 線說明 ----\n"
out = s[:a] + HDR + ours + '# ---- 以下為 ev8 線說明 ----\n' + theirs + tail
assert '<<<<<<<' not in out and '>>>>>>>' not in out
OPS = [
[
"#   (2) 權威改 door_regrasp_mode3.RegraspAuthority（mode 1–8：換抓＋通過中拉關＋放手退出；gate 2–8；W2Exec3、GripGate 同 K3-B）；\n#       gate 2–6 與 HoldConfirm 由 door_regrasp_exec_core.RegraspExecCore（gate 2＝Gate2StopObs：訂閱 /wgmpc/pause_status 取 ZERO_PUB 為窗起點，\n#       逐物理步以判定器 P2／P3 同一套程式核停止與證據覆蓋；不合格只使 mode 2 不准入）\n",
"#       每物理步以**同一份 v3 快照分類紀錄**驅動（與假執行端共用類別）；提交當步（world.step 之前）on_commit 啟動／結束 gate；\n#       gate 狀態經權威 extra_state 'gates' 發給任務端；v3 接觸窗口摘要每 2 步發 /door/contact_v3_summary。\n",
""
],
[
"                help='（EV7）K3-B 受控釋放：模式 2／3、逐步解除監看、5 N 恢復權威（需 --w2-mode-run-id 與 --pause-finger-max-effort）')\nap.add_argument('--w2-unload', action='store_true',                                # EV8\n                help='（EV8）K3-B 閉爪卸載候選：mode 1 提交起 20→5 N 斜坡、低速確認後才准 mode 2（需 --w2-release）')\n",
"",
"ap.add_argument('--regrasp', action='store_true',                                  # RG8\n                help='（EV9r8）換抓 mode 0–6：權威 door_regrasp_mode1.RegraspAuthority＋執行端核心 door_regrasp_exec_core（需 --w2-mode-run-id、'\n                     '--bench-start、--contact-v3-out、--regrasp-ref；與 --w2-release／--w2-unload 互斥）')\nap.add_argument('--regrasp-ref', default='', help='（EV9r8）已登錄換抓參考包（door_regrasp_ref_registry.validate 須通過）')\n"
],
[
"import door_k3b_unload_ctrl as K3BUC                             # noqa: E402  EV8：卸載斜坡設定與讀回\nimport door_w2_mode8 as W2M8                                     # noqa: E402  EV8：權威 r8（gate2）\nimport collections as _w2_col                                    # noqa: E402  EV6\n",
"",
"import door_regrasp_mode1 as RGM                                 # noqa: E402  RG8：換抓權威（版本名）\nimport door_regrasp_mode3 as RGM3                                # noqa: E402  RG8：換抓＋關門權威 r4（mode 1–8、gate 2–8）\nimport door_regrasp_exec_core as RGX                             # noqa: E402  RG8：換抓執行端核心（與假執行端共用）\nimport door_regrasp_ref_registry as RGREG                        # noqa: E402  RG8：參考包登錄核驗\nimport door_geom_v3 as G3R                                       # noqa: E402  RG8：內側把手位姿\nimport yaml                                                      # noqa: E402  RG8：門規格（與登錄核驗同一份 yaml）\n"
],
[
"            self.w2_req_pub = self.create_publisher(String, '/door/w2_mode_req', _lat_w2)\n            self.w2_state_pub = self.create_publisher(String, '/door/w2_mode_state', 50)\n            self.w2_eff_q = []                                                         # EV7：5 N 恢復請求\n",
"",
"            if getattr(a, 'regrasp', False):                                           # RG8：v3 接觸窗口摘要給任務端\n                self.rg8_c3_pub = self.create_publisher(String, '/door/contact_v3_summary', 50)\n                self.rg8_pause_status = None                                           # RG8：求解端暫停監督（ZERO_PUB 宣告 ⇒ gate 2 窗起點）\n                self.create_subscription(String, '/wgmpc/pause_status', self._counted(self._rg8_ps), _lat_w2)\n"
],
[
"            self.w2_n_overflow += 1\n            return\n        self.w2_inbox.append((e, _ev_time.monotonic()))\n",
"",
"\n    def _rg8_ps(self, m: String):                                                   # RG8\n        try:\n            self.rg8_pause_status = json.loads(m.data)\n        except Exception:\n            self.rg8_pause_status = {'_unparsable': str(m.data)[:200]}\n"
],
[
"    w2x = None                                                                      # EV6\n    w2_xm = w2_effa = w2_grip = None                                                # EV7\n    w2_xu, w2_xu_ctrl, w2_xu_step = None, [], {'cur': None}                          # EV8\n",
"",
"    rg8_core, rg8_rt, rg8_cfg = None, {'step': 0, 'abort_handled': False, 'n_c3_pub': 0}, None   # RG8\n\n    def _rg8_tcp():                                                                 # RG8：同一份運動學模型的 TCP 世界位姿（本步實測狀態）\n        _qm8 = robot.get_joint_positions()\n        _p8, _q8 = robot.get_world_pose()\n        return np.asarray(K.fk([float(_p8[0]), float(_p8[1]), float(yaw_of(_q8))] + [float(_qm8[idx[j]]) for j in ARM], 'link_tcp'), float)\n"
],
[
"                                                       't_conf': w2_xm.t.get('B_CONFIRMED'), 'step_conf': w2_xm.step_conf,\n                                                       'last_valid_step': w2_xm.last_valid_step},\n                                           'effort': w2_effa.ack(), 'unload': w2_xu.state_brief(), 'finger_effort': _fe_last8()}\n",
"",
"        if getattr(a, 'regrasp', False):                                            # RG8：換抓權威＋執行端核心（啟動即核互斥與登錄）\n            _why8 = None\n            if not (a.bench_start and a.contact_v3_out and a.regrasp_ref):\n                _why8 = '--regrasp 需 --bench-start、--contact-v3-out、--regrasp-ref'\n            elif a.w2_release or getattr(a, 'w2_unload', False):\n                _why8 = '--regrasp 與 --w2-release／--w2-unload 互斥'\n            if _why8 is None:\n                try:\n                    rg8_pkg = json.load(open(a.regrasp_ref))\n                except (OSError, ValueError) as _e8:\n                    rg8_pkg, _why8 = None, f'參考包讀取失敗：{_e8}'\n            if _why8 is None:\n                _ok8, _w8 = RGREG.validate(rg8_pkg)\n                if not _ok8:\n                    _why8 = f'參考包登錄核驗不過：{_w8}'\n            _spec_reg8 = os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')\n            if _why8 is None and hashlib.sha256(open(a.door_asset, 'rb').read()).hexdigest() != hashlib.sha256(open(_spec_reg8, 'rb').read()).hexdigest():\n                _why8 = '門規格與登錄核驗所用 door_local_v3.yaml 不同'\n            if _why8 is None and float(np.max(np.abs(_door_T() - np.eye(4)))) > 1e-6:\n                _why8 = f'門根變換非單位（參考包座標＝門局部座標）：{_door_T().tolist()}'\n            if _why8 is not None:\n                print(f'[door] **RG8 拒跑**：{_why8}', flush=True)\n                sys.exit(66)\n            rg8_spec = yaml.safe_load(open(a.door_asset))\n            w2_auth = RGM3.RegraspAuthority(a.w2_mode_run_id, run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER,\n                                           cand_timeout_s=a.w2_cand_timeout_s, cmd_max_age_s=a.max_cmd_age_s,\n                                           post_commit_timeout_s=a.w2_post_commit_timeout_s, dt=a.physics_dt,\n                                           a_lin=a.base_accel_lin, a_ang=a.base_accel_ang, arm_rate_max=a.arm_rate_max)\n            w2x = W2X3.W2Exec3(w2_auth, env_in, filt=lambda u: k2_latch.filter_wb(u))\n            w2_grip = K3BG.GripGate(a.finger_open, a.finger_closed)\n            rg8_core = RGX.RegraspExecCore(w2_auth, rg8_pkg, lambda _q8: _door_T() @ G3R.handle_pose_inner(rg8_spec, _q8), dt=a.physics_dt)\n            w2_auth.extra_state = lambda: {'gates': rg8_core.gates_state(rg8_rt['step']), 'regrasp_abort': rg8_core.abort}\n            rg8_cfg = {'ref': os.path.relpath(a.regrasp_ref, WS), 'ref_sha256': hashlib.sha256(open(a.regrasp_ref, 'rb').read()).hexdigest(),\n                       'core_version': RGX.VERSION, 'authority_version': RGM3.VERSION, 'door_spec_sha256': hashlib.sha256(open(a.door_asset, 'rb').read()).hexdigest()}\n            print(f'[door] **RG8 換抓**：權威 {RGM3.VERSION}、核心 {RGX.VERSION}、參考包 {rg8_cfg[\"ref\"]}', flush=True)\n"
],
[
"                bench_rel.state, bench_rel.released = 'aborted', {'why': f'abort_or_stop：{node.k2_abort_msg or node.stop_req}'}\n                break\n            _es = None if node.wb_env_latest is None else node.wb_env_latest.get('source_seq')\n",
"",
"            if rg8_core is not None:                                                # RG8：W2 唯一入口＝v3；以收件匣中 mode_seq 0 的最新序號判斷\n                _es = max((_e8.get('source_seq') for _e8, _ in node.w2_inbox if _e8.get('mode_seq') == 0), default=None)\n"
],
[
"            if bench_rel.try_release(_es, time.monotonic()) != 'frozen':\n                if bench_rel.state == 'released':                                   # (R2)：真正入口與命令鏈\n                    _ab = node.k2_abort_msg is not None or node.stop_req is not None\n",
"                    _ok, _why, _irec = BENCH.release_intake(env_in, chain, node.wb_env_latest, (bench_rel.announced or {}).get('source_seq'),\n                                                            t_b0, k2_latch.filter_wb, _ab)\n",
"                    if rg8_core is not None:                                        # RG8：只把宣告那一筆經 W2Exec 進鏈；較新者留在收件匣\n                        _as8 = (bench_rel.announced or {}).get('source_seq')\n                        _e8 = next((_x8 for _x8, _ in node.w2_inbox if _x8.get('source_seq') == _as8), None)\n                        _keep8 = [(_x8, _w8) for _x8, _w8 in node.w2_inbox if not (_x8.get('source_seq') is not None and _as8 is not None\n                                                                                 and _x8.get('source_seq') <= _as8)]\n                        node.w2_inbox.clear()\n                        node.w2_inbox.extend(_keep8)\n                        _ok, _why, _irec, rg8_rt['rel_sel'] = RGX.bench_release_w2(w2x, chain, _e8, _as8, 1, t_b0, _ab)   # 步號＝主迴圈台架首步（下方 step_id＝1）\n                    else:\n                        _ok, _why, _irec = BENCH.release_intake(env_in, chain, node.wb_env_latest, (bench_rel.announced or {}).get('source_seq'),\n                                                                t_b0, k2_latch.filter_wb, _ab)\n"
],
[
"        _w2sel = None                                                               # EV6 r2：已知中止 → v3 入口 → 請求 → 進鏈\n        if w2x is not None:\n            _ib6 = []\n",
"            while node.w2_inbox:\n",
"            while node.w2_inbox and not (rg8_core is not None and bench_first_pending):   # RG8：第 1 步首筆固定（同 v2 路徑）\n"
],
[
"            while node.w2_inbox:\n                _ib6.append(node.w2_inbox.popleft())\n            _rq6, node.w2_req_q = list(node.w2_req_q), []\n",
"            _w2sel, _relay6, _nv6 = w2x.step_begin(step_id, t, chain, known_abort=_w2_known_abort(), inbox=_ib6, requests=_rq6)\n",
"            if rg8_core is not None and bench_first_pending and rg8_rt.get('rel_sel') is not None:   # RG8：首步已於釋放時 select（步 0），不再 select\n                _w2sel, _relay6, _nv6 = rg8_rt['rel_sel'], [], []\n                node.w2_req_q = _rq6 + node.w2_req_q\n            else:\n                _w2sel, _relay6, _nv6 = w2x.step_begin(step_id, t, chain, known_abort=_w2_known_abort(), inbox=_ib6, requests=_rq6)\n            if rg8_core is not None and (_w2sel.get('event') or {}).get('kind') == 'commit':   # RG8：提交當步（world.step 之前）啟動／結束 gate\n                rg8_core.on_commit(step_id, t, w2_auth.mode_seq, _rg8_tcp())\n"
],
[
"                    _ev4_violation(f\"w2_mode:{w2_auth.aborted['why']}\")\n            if step_id % 2 == 0:\n                node.w2_state_pub.publish(String(data=json.dumps(w2_auth.state_msg(step_id, t), default=float)))\n",
"",
"                if rg8_core is not None:                                                # RG8：v3 接觸窗口摘要（空窗口不發）\n                    _s8 = rg8_core.summary()\n                    if _s8 is not None:\n                        node.rg8_c3_pub.publish(String(data=json.dumps(_s8, default=float)))\n                        rg8_rt['n_c3_pub'] += 1\n"
],
[
"                                                        'why': f'w1e_finger_effort：{_fer[\"why\"]}', 'step': step_id}\n                    k3m['events'].append({'step': step_id, 't': t, 'kind': 'pause_' + _kp, 'msg': _ackp})\n                    node.k3m_pause_ack_pub.publish(String(data=json.dumps(_ackp, default=float)))\n",
"                if k3m['active'] and k3m['violation'] is None:\n",
"                if rg8_core is not None and rg8_core.handover is not None and 'regrasp_end' not in k3m:   # RG8：走廊監看判定保存至交接步\n                    k3m['regrasp_end'] = dict(rg8_core.handover)\n                if k3m['active'] and k3m['violation'] is None and 'regrasp_end' not in k3m:\n"
],
[
"        _hm_s = xfc.GetLocalToWorldTransform(hprim)\n        _HM_s = np.array([[float(_hm_s[r][c]) for r in range(4)] for c in range(4)], float)\n        _snap.take(step_id, lambda: leaf_pair_v.get_contact_force_data(dt=a.physics_dt))   # EV9r7：本步唯一一次讀取\n",
"",
"        _c3_r = None                                                                # RG8：未寫出即視為本步無紀錄（不當作無接觸）\n"
],
[
"                                               for _n8 in ('uflite_finger1', 'uflite_finger2')])\n            w2_xu.step(step_id, t, w2_xu_step['cur'][1], w2_xu_step['cur'][0], _pfu8, _lc is not None, float(_qd_rec),\n                       [float(p_now[0]), float(p_now[1]), float(yw)], [float(x) for x in q_arm])\n",
"",
"        if rg8_core is not None and _w2sel is not None:                                # RG8：同一物理步資料（同一份 v3 快照分類紀錄）\n            rg8_rt['step'] = step_id\n            _z8 = (((node.rg8_pause_status or {}).get('outcome') or {}).get('transitions') or {}).get('ZERO_PUB')\n            if _z8 is not None:                                                     # gate 2 窗起點＝求解端宣告的 ZERO_PUB\n                rg8_core.g[2].set_zero(_z8)\n            _T8 = np.asarray(K.fk([float(p_now[0]), float(p_now[1]), float(yw)] + [float(x) for x in q_arm], 'link_tcp'), float)\n            _why8s = rg8_core.step(step_id, t, _w2sel.get('mode_seq'), _c3_r, _T8, [float(p_now[0]), float(p_now[1]), float(yw)],\n                                   [float(x) for x in q_arm], [float(q_meas[fidx[j]]) for j in FJ] if FJ else None, float(_qd_rec),\n                                   base_cmd=[float(x) for x in res.base_cmd], arm_cmd=[float(x) for x in res.u_applied[3:]])\n            if _why8s is not None and not rg8_rt['abort_handled']:\n                rg8_rt['abort_handled'] = True\n                if k2_latch.latched is None:\n                    _g8 = (a.finger_open if node.grip_cmd is None else min(max(float(node.grip_cmd), a.finger_closed), a.finger_open))\n                    k2_latch.on_msg(f'regrasp：{_why8s}', step_id, t, _g8)\n                    k2_latch.expire_chain(chain)\n                    ex.cancel_handover(f'regrasp：{_why8s}')\n                    node.k3m_abort_pub.publish(String(data=json.dumps({'why': f'regrasp：{_why8s}', 't': t, 'source': 'executor_rg8'}, default=float)))\n                rec['events'].append({'sim_t': t, 'step': step_id, 'regrasp_abort': _why8s})\n                print(f'[door] **RG8 換抓中止**（K2 閂鎖）@ step {step_id}：{_why8s}', flush=True)\n"
],
[
"                              chain_mode={'0': 'normal', '1': 'timeout', '2': 'stop_unverified', '3': 'fail'},\n                              note='source_seq＝−1：本步不是正常套用的全身命令（逾時減速、停止、非全身控制或對照表外）')\n    if w2x is not None:                                                             # EV6\n",
"",
"        if rg8_core is not None:                                                     # RG8\n            rec['regrasp'] = {'config': rg8_cfg, 'core': rg8_core.report(), 'runtime': dict(rg8_rt), 'grip_gate': w2_grip.report(),\n                              'k3m_regrasp_end': k3m.get('regrasp_end')}\n"
]
]
for ctx, old, new in reversed(OPS):                                  # 逆序：後段替換不影響前段的前文
    assert out.count(ctx + old) == 1, (ctx[-80:], old[:80])
    out = out.replace(ctx + old, ctx + new)
if '--check' in sys.argv:
    cur = open('isaac_door_k2_sim_ev9r8.py').read()
    print('identical' if cur == out else 'DIFFERENT')
    sys.exit(0 if cur == out else 1)
open('isaac_door_k2_sim_ev9r8.py', 'w').write(out)
print('written')
