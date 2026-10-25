"""door_k3m_task_node_k3b.py → door_k3m_task_node_k3b2.py（Codex 20261026_010158：策略 r2）。**只插入**；並核對 w1b2 每行依序仍在新檔中。
--w2-release 時以 door_k3m_policy_k3b2.K3MReleasePolicy2 取代 r1 策略物件（同參數）。
"""
base = open('door_k3m_task_node_k3b.py').read()
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


ib('# K3-B 任務節點（door_k3m_task_node_w2.py 的版本化複本',
   '# K3-B 任務節點 r2（door_k3m_task_node_k3b.py 的版本化複本，只插入；Codex 20261026_010158）：--w2-release 時策略改 door_k3m_policy_k3b2.K3MReleasePolicy2\n'
   '#   （core r3：新帶、終點保持逐筆核對；樣本 s_cmd＝上一輪已發布目標；錨點證據）。其餘同 k3b。\n')
ia("import door_k3m_policy_k3b as PK3B                                        # noqa: E402  K3-B：受控釋放策略\n",
   "import door_k3m_policy_k3b2 as PK3B2                                      # noqa: E402  K3-B r2：受控釋放策略 r2\n")
ia("    if a.w2_release:                                                       # K3-B：同參數另建受控釋放策略（取代上一行的物件）\n"
   "        if not a.w2_mode_run_id:\n"
   "            raise SystemExit('--w2-release 需要 --w2-mode-run-id')\n"
   "        pol = PK3B.K3MReleasePolicy(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()),\n"
   "                                    w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
   "                                    test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s)\n",
   "        pol = PK3B2.K3MReleasePolicy2(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()),   # K3-B r2\n"
   "                                      w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
   "                                      test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s)\n")
open('door_k3m_task_node_k3b2.py', 'w').write(out)

for name, src in (('task_node_k3b', base), ('w1b2', w1b2)):
    a_, b_ = src.splitlines(), out.splitlines()
    j = 0
    for line in a_:
        while j < len(b_) and b_[j] != line:
            j += 1
        assert j < len(b_), f'{name} 原行遺失：{line!r}'
        j += 1
    print(f'insert-only ok（對 {name}）：', len(a_), '→', len(b_))
