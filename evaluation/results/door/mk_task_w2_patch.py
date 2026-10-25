"""door_k3m_task_node_w1b2.py → door_k3m_task_node_w2.py（Codex 20261025_214144：任務端 HELD_WAIT／請求）。**只插入**。
核對：原檔每一行依序全在新檔中（insert-only）。
"""
src = open('door_k3m_task_node_w1b2.py').read()
out = src


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


ib('# W1b2 任務節點（door_k3m_task_node_w1b.py 的版本化複本',
   '# W2 任務節點（door_k3m_task_node_w1b2.py 的版本化複本，只插入；Codex 20261025_214144；設計 W2_hold_handover_design_draft2）：\n'
   '#   --w2-mode-run-id（預設空 ⇒ 與 w1b2 同行為）：策略換為 door_k3m_policy_w2.K3MW2Policy——求解端 HELD（觀察通過）後不結束、\n'
   '#   held_wait 滿 --w2-req-delay-s 送一次模式請求到 /door/w2_mode_req_task（latched；執行端權威登記後才轉送求解端）；\n'
   '#   只以執行端權威狀態 /door/w2_mode_state 判提交（applied_confirmed）；測試模式保持 --w2-test-hold-s 後 DONE；\n'
   '#   請求被拒、確認不符、提交逾時、權威中止、狀態停更、HELD 有界超時 ⇒ 中止（有界失敗、不回切）。task.json 另記 w2；trace 每列另記 w2。\n'
   '#   測試模式不張爪、不退離、不增加角度；監看、門檻、pause_total_s 不變。\n')
ia("import door_k3m_policy_p2 as PP2                                          # noqa: E402  p2：SETTLE 完成訊號\n",
   "import door_k3m_policy_w2 as PW2                                          # noqa: E402  W2：HELD_WAIT／模式請求\n")
ia("    ap.add_argument('--lw-heartbeat-max-age-s', type=float, default=1.0)      # 觀察器心跳最大年齡（牆鐘）\n",
   "    ap.add_argument('--w2-mode-run-id', default='')                            # W2：空 ⇒ 關閉（w1b2 行為）\n"
   "    ap.add_argument('--w2-req-delay-s', type=float, default=1.0)\n"
   "    ap.add_argument('--w2-commit-timeout-s', type=float, default=2.0)\n"
   "    ap.add_argument('--w2-test-hold-s', type=float, default=3.0)\n"
   "    ap.add_argument('--w2-state-stale-s', type=float, default=1.5)\n")
ia("    pol = PP2.K3MP2Policy(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()))\n",
   "    if a.w2_mode_run_id:                                                   # W2：同參數另建 W2 策略（取代上一行的物件）\n"
   "        pol = PW2.K3MW2Policy(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()),\n"
   "                              w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
   "                              test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s)\n")
ia("    rep['policy_p2_sha256'] = hashlib.sha256(open(PP2.__file__, 'rb').read()).hexdigest()\n",
   "    if a.w2_mode_run_id:                                                   # W2\n"
   "        rep['policy_w2_sha256'] = hashlib.sha256(open(PW2.__file__, 'rb').read()).hexdigest()\n")
ia("    nd = Node2()\n    nd.pol = pol\n",
   "    if a.w2_mode_run_id:                                                   # W2：請求（給執行端權威）與權威狀態\n"
   "        _lat_w2 = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)\n"
   "        nd.w2_req_pub = nd.create_publisher(String, '/door/w2_mode_req_task', _lat_w2)\n"
   "        nd.w2_bad_state = 0\n"
   "\n"
   "        def _w2_state_cb(m_):\n"
   "            try:\n"
   "                pol.on_w2_state(json.loads(m_.data), nd._t_now())\n"
   "            except Exception:                                              # noqa: BLE001\n"
   "                nd.w2_bad_state += 1\n"
   "        nd.create_subscription(String, '/door/w2_mode_state', _w2_state_cb, 50)\n")
ib("            _g = pol.geom_last\n",
   "            if a.w2_mode_run_id:                                               # W2：模式請求（已中止 ⇒ 清除不發）\n"
   "                for _rq in pol.take_w2_outbox():\n"
   "                    nd.w2_req_pub.publish(String(data=json.dumps(_rq, default=float)))\n"
   "                    rep['events'].append({'t': t, 'w2_mode_request_published': _rq})\n")
ia("            rep['trace'][-1].update(wall_mono=_w2_loop0, loop_seq=_loop_seq2, trace_seq=len(rep['trace']))   # W1b2\n",
   "            if a.w2_mode_run_id:                                               # W2\n"
   "                rep['trace'][-1]['w2'] = pol.w2_state\n")
ib("        rep['shutdown'] = _w2_sd\n        save()\n",
   "        if a.w2_mode_run_id:                                                   # W2\n"
   "            rep['w2'] = dict(pol.w2_report(), n_bad_state=getattr(nd, 'w2_bad_state', None))\n")
open('door_k3m_task_node_w2.py', 'w').write(out)

# insert-only 核對
a_, b_ = src.splitlines(), out.splitlines()
j = 0
for line in a_:
    while j < len(b_) and b_[j] != line:
        j += 1
    assert j < len(b_), f'原行遺失：{line!r}'
    j += 1
print('insert-only ok：', len(a_), '→', len(b_))
