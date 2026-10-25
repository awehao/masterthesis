"""door_k3m_task_node_w2.py → door_k3m_task_node_k3b.py（K3-B 受控釋放任務端；Codex 20261026_000706）。**只插入**；
並核對 w1b2 每行依序仍在新檔中（task_node_w2 本身即 w1b2 的 insert-only）。--w2-release 預設關 ⇒ 與 task_node_w2 同行為。
"""
base = open('door_k3m_task_node_w2.py').read()
w1b2 = open('door_k3m_task_node_w1b2.py').read()
out = base


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


ib('# W2 任務節點（door_k3m_task_node_w1b2.py 的版本化複本',
   '# K3-B 任務節點（door_k3m_task_node_w2.py 的版本化複本，只插入；Codex 20261026_000706；規格 K3B_release_clear_spec_draft2.md）：\n'
   '#   --w2-release（需 --w2-mode-run-id；預設關 ⇒ 與 w2 同行為）：策略換為 door_k3m_policy_k3b.K3MReleasePolicy——測試模式保持後請求 mode 2、\n'
   '#   RELEASE 相位（解除監看、mode 3 請求、退離目標、5 N 恢復請求、退離終點保持後 DONE；不收臂）。\n'
   '#   另發 /door/finger_effort_req（latched）、訂 /door/finger_effort_ack；trace 每列另記 k3b；task.json 另記 k3b。\n')
ia("import door_k3m_policy_w2 as PW2                                          # noqa: E402  W2：HELD_WAIT／模式請求\n",
   "import door_k3m_policy_k3b as PK3B                                        # noqa: E402  K3-B：受控釋放策略\n")
ia("    ap.add_argument('--w2-state-stale-s', type=float, default=1.5)\n",
   "    ap.add_argument('--w2-release', action='store_true')                         # K3-B：需 --w2-mode-run-id\n")
ia("                              test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s)\n",
   "    if a.w2_release:                                                       # K3-B：同參數另建受控釋放策略（取代上一行的物件）\n"
   "        if not a.w2_mode_run_id:\n"
   "            raise SystemExit('--w2-release 需要 --w2-mode-run-id')\n"
   "        pol = PK3B.K3MReleasePolicy(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()),\n"
   "                                    w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
   "                                    test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s)\n")
ia("        nd.create_subscription(String, '/door/w2_mode_state', _w2_state_cb, 50)\n",
   "        if a.w2_release:                                                   # K3-B：5 N 恢復請求與 ack\n"
   "            nd.eff_req_pub = nd.create_publisher(String, '/door/finger_effort_req', _lat_w2)\n"
   "            nd.eff_bad_ack = 0\n"
   "\n"
   "            def _eff_ack_cb(m_):\n"
   "                try:\n"
   "                    pol.on_effort_ack(json.loads(m_.data), nd._t_now())\n"
   "                except Exception:                                          # noqa: BLE001\n"
   "                    nd.eff_bad_ack += 1\n"
   "            nd.create_subscription(String, '/door/finger_effort_ack', _eff_ack_cb, _lat_w2)\n")
ia("                for _rq in pol.take_w2_outbox():\n"
   "                    nd.w2_req_pub.publish(String(data=json.dumps(_rq, default=float)))\n"
   "                    rep['events'].append({'t': t, 'w2_mode_request_published': _rq})\n",
   "            if a.w2_release:                                                   # K3-B：5 N 恢復請求（已中止 ⇒ 清除不發）\n"
   "                for _rq in pol.take_eff_outbox():\n"
   "                    nd.eff_req_pub.publish(String(data=json.dumps(_rq, default=float)))\n"
   "                    rep['events'].append({'t': t, 'k3b_effort_request_published': _rq})\n")
ia("                rep['trace'][-1]['w2'] = pol.w2_state\n",
   "            if a.w2_release:                                                   # K3-B\n"
   "                rep['trace'][-1]['k3b'] = {'state': pol.k3b.state, 's_cmd': pol.rel['s_cmd'], 'rel': pol.rel['state']}\n")
ia("            rep['w2'] = dict(pol.w2_report(), n_bad_state=getattr(nd, 'w2_bad_state', None))\n",
   "        if a.w2_release:                                                       # K3-B\n"
   "            rep['k3b'] = dict(pol.k3b_report(), n_bad_ack=getattr(nd, 'eff_bad_ack', None))\n")
open('door_k3m_task_node_k3b.py', 'w').write(out)

for name, src in (('task_node_w2', base), ('w1b2', w1b2)):
    a_, b_ = src.splitlines(), out.splitlines()
    j = 0
    for line in a_:
        while j < len(b_) and b_[j] != line:
            j += 1
        assert j < len(b_), f'{name} 原行遺失：{line!r}'
        j += 1
    print(f'insert-only ok（對 {name}）：', len(a_), '→', len(b_))
