"""door_k3m_task_node_k3b3.py → door_k3m_task_node_k3b4.py（Codex 20261026_025647：閉爪卸載候選整合，策略 r4）。**只插入**；並核對 k3b3 與 w1b2 每行依序仍在新檔中。
--w2-unload（需 --w2-release）時以 door_k3m_policy_k3b4.K3MReleasePolicy4(unload=True) 取代 r3 策略物件（同參數）。
"""
base = open('door_k3m_task_node_k3b3.py').read()
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


ib('# K3-B 任務節點 r3（door_k3m_task_node_k3b2.py 的版本化複本',
   '# K3-B 任務節點 r4（door_k3m_task_node_k3b3.py 的版本化複本，只插入；Codex 20261026_025647）：--w2-unload（需 --w2-release）時\n'
   '#   策略改 door_k3m_policy_k3b4.K3MReleasePolicy4(unload=True)：mode 1 後依卸載准入請求 mode 2、mode 3 另等求解端 release_settled、\n'
   '#   不送 5 N 請求改讀回核對。未給 --w2-unload ⇒ 與 k3b3 同行為。\n')
ia("import door_k3m_policy_k3b3 as PK3B3                                      # noqa: E402  K3-B r3：受控釋放策略 r3\n",
   "import door_k3m_policy_k3b4 as PK3B4                                      # noqa: E402  K3-B r4：閉爪卸載候選\n")
ia("        pol = PK3B3.K3MReleasePolicy3(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()),   # K3-B r3\n"
   "                                      w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
   "                                      test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s)\n",
   "        if a.w2_unload:                                                    # K3-B r4\n"
   "            pol = PK3B4.K3MReleasePolicy4(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()),\n"
   "                                          w2_run_id=a.w2_mode_run_id, req_delay_s=a.w2_req_delay_s, commit_timeout_s=a.w2_commit_timeout_s,\n"
   "                                          test_hold_s=a.w2_test_hold_s, state_stale_s=a.w2_state_stale_s, unload=True)\n"
   "    if getattr(a, 'w2_unload', False) and not a.w2_release:\n"
   "        raise SystemExit('--w2-unload 需要 --w2-release')\n")
ia("    ap.add_argument('--w2-release', action='store_true')                         # K3-B：需 --w2-mode-run-id\n",
   "    ap.add_argument('--w2-unload', action='store_true')                          # K3-B r4：需 --w2-release\n")
open('door_k3m_task_node_k3b4.py', 'w').write(out)

for name, src in (('task_node_k3b3', base), ('w1b2', w1b2)):
    a_, b_ = src.splitlines(), out.splitlines()
    j = 0
    for line in a_:
        while j < len(b_) and b_[j] != line:
            j += 1
        assert j < len(b_), f'{name} 原行遺失：{line!r}'
        j += 1
    print(f'insert-only ok（對 {name}）：', len(a_), '→', len(b_))
