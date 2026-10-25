"""door_k3m_task_node_k3b2.py → door_k3m_task_node_k3b3.py（Codex 20261026_012939：策略 r3，三維收斂）。**只插入**；並核對 w1b2 每行依序仍在新檔中。
--w2-release 時以 door_k3m_policy_k3b2.K3MReleasePolicy2 取代 r1 策略物件（同參數）。
"""
base = open('door_k3m_task_node_k3b2.py').read()
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


ib('# K3-B 任務節點 r2（door_k3m_task_node_k3b.py 的版本化複本',
   '# K3-B 任務節點 r3（door_k3m_task_node_k3b2.py 的版本化複本，只插入；Codex 20261026_012939）：--w2-release 時策略改 door_k3m_policy_k3b3.K3MReleasePolicy3\n'
   '#   （core r4：終點收斂以實測 TCP 三維位置核對）。其餘同 k3b2。\n')
ia("import door_k3m_policy_k3b2 as PK3B2                                      # noqa: E402  K3-B r2：受控釋放策略 r2\n",
   "import door_k3m_policy_k3b3 as PK3B3                                      # noqa: E402  K3-B r3：受控釋放策略 r3\n")
ia("                                      test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s)\n",
   "        pol = PK3B3.K3MReleasePolicy3(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()),   # K3-B r3\n"
   "                                      w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
   "                                      test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s)\n")
open('door_k3m_task_node_k3b3.py', 'w').write(out)

for name, src in (('task_node_k3b2', base), ('w1b2', w1b2)):
    a_, b_ = src.splitlines(), out.splitlines()
    j = 0
    for line in a_:
        while j < len(b_) and b_[j] != line:
            j += 1
        assert j < len(b_), f'{name} 原行遺失：{line!r}'
        j += 1
    print(f'insert-only ok（對 {name}）：', len(a_), '→', len(b_))
