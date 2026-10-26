"""door_k3m_task_node_k3b4.py → door_k3m_task_node_k3b5.py（策略 r5：核心 release5 終點收斂窗修正）。**只插入**；並核對 k3b4 與 w1b2 每行依序仍在新檔中。
--w2-unload 時以 door_k3m_policy_k3b5.K3MReleasePolicy5(unload=True) 取代 r4 策略物件（同參數）。
"""
base = open('door_k3m_task_node_k3b4.py').read()
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


ib('# K3-B 任務節點 r4（door_k3m_task_node_k3b3.py 的版本化複本',
   '# K3-B 任務節點 r5（door_k3m_task_node_k3b4.py 的版本化複本，只插入）：--w2-unload 時策略改 door_k3m_policy_k3b5.K3MReleasePolicy5\n'
   '#   （核心 door_k3b_release5：終點收斂窗保留窗外最新一筆）。其餘同 k3b4。\n')
ia("import door_k3m_policy_k3b4 as PK3B4                                      # noqa: E402  K3-B r4：閉爪卸載候選\n",
   "import door_k3m_policy_k3b5 as PK3B5                                      # noqa: E402  K3-B r5：核心 release5\n")
ia("                                          test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s, unload=True)\n",
   "            pol = PK3B5.K3MReleasePolicy5(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()),\n"
   "                                          w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
   "                                          test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s, unload=True)   # K3-B r5\n")
open('door_k3m_task_node_k3b5.py', 'w').write(out)

for name, src in (('task_node_k3b4', base), ('w1b2', w1b2)):
    a_, b_ = src.splitlines(), out.splitlines()
    j = 0
    for line in a_:
        while j < len(b_) and b_[j] != line:
            j += 1
        assert j < len(b_), f'{name} 原行遺失：{line!r}'
        j += 1
    print(f'insert-only ok（對 {name}）：', len(a_), '→', len(b_))
