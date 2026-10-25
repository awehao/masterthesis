"""door_k3m_task_node_w1b.py → door_k3m_task_node_w1b2.py：唯一刪改＝移除迴圈內「每 100 筆同步 save()」兩行；其餘只插入。"""
src = open('door_k3m_task_node_w1b.py').read()
out = src
def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)
def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)
DEL = "            if len(rep['trace']) % 100 == 0:\n                save()\n"
assert out.count(DEL) == 1
out = out.replace(DEL, "            _tw2.put(rep['trace'][-1])                                          # W1b2：有界非阻塞逐列紀錄（取代每 100 筆同步 save）\n")
ib("# W1b 任務節點（door_k3m_task_node_p2c.py 的版本化複本；",
   "# W1b2 任務節點（door_k3m_task_node_w1b.py 的版本化複本；Codex 20261025_034022：任務端最小紀錄修正）：\n"
   "#   唯一刪改＝移除迴圈內「每 100 筆在同一迴圈同步重寫整份 task.json」（w1f_60_1 第 1000 筆後 0.30 s 更新間隙的候選原因，未確認根因）；\n"
   "#   改為 bounded_trace_writer.BoundedTraceWriter：每列深拷貝後 put_nowait 進有界佇列（容量 4096），背景執行緒序列化、append 到\n"
   "#   <out 去 .json>_trace.jsonl、定期 flush、覆寫心跳檔 <…>_trace_hb.json；佇列滿 ⇒ 丟棄並計數。trace 每列另記 wall_mono（本輪迴圈起點\n"
   "#   單調時間）、loop_seq（迴圈輪序）、trace_seq；LoopTimer 記每輪迴圈耗時（最大、p99、> 0.1 s 次數）。收尾：先有期限（5 s）關閉寫入端，\n"
   "#   再在迴圈外寫一次完整 task.json；task.json 另記 trace_log（入列／寫入／丟棄／寫入失敗／未寫完／逾期、complete）與 loop_timing。\n"
   "#   任務邏輯、策略、話題、門檻全部不變。\n")
ia("import door_k3m_geom_monitor as K3MGEO                                    # noqa: E402\n",
   "import bounded_trace_writer as BTW                                        # noqa: E402  W1b2：有界非阻塞紀錄\n")
ib("    rclpy.init()\n    nd = Node2()\n",
   "    _tw_base = a.out[:-5] if a.out.endswith('.json') else a.out                 # W1b2\n"
   "    _tw2 = BTW.BoundedTraceWriter(_tw_base + '_trace.jsonl', capacity=4096, heartbeat_path=_tw_base + '_trace_hb.json').start()\n"
   "    _lt2 = BTW.LoopTimer(over_s=0.1)\n"
   "    _loop_seq2 = 0\n")
ia("        while rclpy.ok():\n            for _ in range(64):\n                rclpy.spin_once(nd, timeout_sec=0.0)\n",
   "            _w2_loop0 = time.monotonic()                                       # W1b2：本輪迴圈起點\n"
   "            _loop_seq2 += 1\n")
# 在 trace append 之後補三欄（append 敘述結束於 '}})\n' 且其後緊接 put）
anchor = "            _tw2.put(rep['trace'][-1])"
ib(anchor, "            rep['trace'][-1].update(wall_mono=_w2_loop0, loop_seq=_loop_seq2, trace_seq=len(rep['trace']))   # W1b2\n")
ia("            if act['finish'] or (w_abort is not None and wnow - w_abort > a.wall_post_abort_s):\n                if w_abort is not None and not act['finish']:\n                    rep['post_abort_wall_cap_hit'] = True\n                break\n",
   "            _lt2.record(_w2_loop0, time.monotonic(), _loop_seq2)                # W1b2：完整一輪的耗時（不含 sleep）\n")
ib("        rep['received_any_sync'] = sync_once\n        save()\n",
   "        rep['trace_log'] = _tw2.close(deadline_s=5.0)                           # W1b2：有期限收尾，再寫一次完整 task.json\n"
   "        rep['trace_log']['jsonl_semantics'] = '每行 {enq_seq, enq_wall_mono, row}；row＝該筆 trace（含 wall_mono／loop_seq／trace_seq）'\n"
   "        rep['loop_timing'] = _lt2.stats()\n")
open('door_k3m_task_node_w1b2.py', 'w').write(out)
a_, b_ = src.splitlines(), out.splitlines()
dl = [l for l in DEL.splitlines()]
j = 0
for ln in a_:
    if ln in dl:
        continue
    while b_[j] != ln:
        j += 1
    j += 1
print('only-delete-2 + insert OK', len(a_), '→', len(b_))
