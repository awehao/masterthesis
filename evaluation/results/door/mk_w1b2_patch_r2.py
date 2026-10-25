"""door_k3m_task_node_w1b.py → door_k3m_task_node_w1b2.py（r2；Codex 20261025_144446）。
刪改只有兩處（其餘只插入）：
  (D1) 移除迴圈內「每 100 筆同步 save()」兩行；
  (D2) 收尾順序：原「save() → 送 /wgmpc/stop 與 /room/stop → spin」改為「送 /wgmpc/stop 與 /room/stop → 有期限關閉紀錄器（2 s）→ save() → spin」，
       控制停止不再等待任何紀錄寫檔（原檔的 save() 本身也在停止之前）。
核對：原檔除 D1 兩行與 D2 被搬移的區塊外，其餘行依序全在新檔中；被搬移的行原樣出現在新檔。
"""
import sys
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


D1 = "            if len(rep['trace']) % 100 == 0:\n                save()\n"
assert out.count(D1) == 1
out = out.replace(D1, "            rep['trace'][-1].update(wall_mono=_w2_loop0, loop_seq=_loop_seq2, trace_seq=len(rep['trace']))   # W1b2\n"
                      "            _tw2.put(rep['trace'][-1])                                          # W1b2：有界非阻塞逐列紀錄（取代每 100 筆同步 save）\n"
                      "            _lt2.record(_w2_loop0, time.monotonic(), _loop_seq2)                # W1b2 r2：本輪起點（回呼處理前）至入列；結束輪也計入\n")
D2_OLD = ("        rep['received_any_sync'] = sync_once\n"
          "        save()\n"
          "        try:\n"
          "            nd.stop_pub.publish(String(data='door_k3a 結束'))\n"
          "            nd.room_stop_pub.publish(String(data=f'door_k3a 結束：{pol.phase} {pol.abort or \"\"}'))\n"
          "            for _ in range(10):\n"
          "                rclpy.spin_once(nd, timeout_sec=0.05)\n"
          "        except Exception:\n"
          "            pass\n")
assert out.count(D2_OLD) == 1, 'D2 anchor'
D2_NEW = ("        rep['received_any_sync'] = sync_once\n"
          "        _w2_sd = {'order': ['stop_and_room_stop_published', 'trace_writer_closed', 'task_json_saved', 'spin_flush']}   # W1b2 r2：收尾順序\n"
          "        try:\n"
          "            nd.stop_pub.publish(String(data='door_k3a 結束'))\n"
          "            nd.room_stop_pub.publish(String(data=f'door_k3a 結束：{pol.phase} {pol.abort or \"\"}'))\n"
          "            _w2_sd['stop_published_wall'] = time.monotonic()\n"
          "        except Exception as _e2:                                                # noqa: BLE001\n"
          "            _w2_sd['stop_publish_error'] = f'{type(_e2).__name__}: {_e2}'\n"
          "        rep['trace_log'] = _tw2.close(deadline_s=2.0)                           # W1b2 r2：呼叫端只做有期限等待（不做 I/O）\n"
          "        rep['trace_log']['jsonl_semantics'] = '每行 {enq_seq, enq_wall_mono, row}；row＝該筆 trace（含 wall_mono／loop_seq／trace_seq）'\n"
          "        rep['loop_timing'] = dict(_lt2.stats(), semantics='每筆 trace 一輪：本輪起點（回呼處理前）至入列完成', n_loops_total=_loop_seq2)\n"
          "        _w2_sd['writer_closed_wall'] = time.monotonic()\n"
          "        rep['shutdown'] = _w2_sd\n"
          "        save()\n"
          "        try:\n"
          "            for _ in range(10):\n"
          "                rclpy.spin_once(nd, timeout_sec=0.05)\n"
          "        except Exception:\n"
          "            pass\n")
out = out.replace(D2_OLD, D2_NEW)
ib("# W1b 任務節點（door_k3m_task_node_p2c.py 的版本化複本；",
   "# W1b2 任務節點（door_k3m_task_node_w1b.py 的版本化複本；Codex 20261025_034022／20261025_144446：任務端最小紀錄修正）：\n"
   "#   刪改兩處：(D1) 移除迴圈內「每 100 筆在同一迴圈同步重寫整份 task.json」（w1f_60_1 第 1000 筆後 0.30 s 更新間隙的候選原因，未確認根因）；\n"
   "#   (D2) 收尾順序改為先送 /wgmpc/stop 與 /room/stop，再有期限（2 s）關閉紀錄器、寫一次完整 task.json（控制停止不等紀錄）。\n"
   "#   紀錄改用 bounded_trace_writer.BoundedTraceWriter（r2）：每列深拷貝後 put_nowait 進有界佇列（容量 4096），背景執行緒序列化、append 到\n"
   "#   <out 去 .json>_trace.jsonl、定期 flush、覆寫心跳檔、收尾 flush／close；佇列滿 ⇒ 丟棄計數；寫入／flush／close 失敗 ⇒ incomplete。\n"
   "#   trace 每列另記 wall_mono（本輪迴圈起點，回呼處理之前的單調時間）、loop_seq、trace_seq；LoopTimer 記每輪（起點→入列）耗時。\n"
   "#   task.json 另記 trace_log、loop_timing、shutdown（收尾順序與時間）。任務邏輯、策略、話題、門檻全部不變。\n")
ia("import door_k3m_geom_monitor as K3MGEO                                    # noqa: E402\n",
   "import bounded_trace_writer as BTW                                        # noqa: E402  W1b2：有界非阻塞紀錄\n")
ib("    rclpy.init()\n    nd = Node2()\n",
   "    _tw_base = a.out[:-5] if a.out.endswith('.json') else a.out                 # W1b2\n"
   "    _tw2 = BTW.BoundedTraceWriter(_tw_base + '_trace.jsonl', capacity=4096, heartbeat_path=_tw_base + '_trace_hb.json').start()\n"
   "    _lt2 = BTW.LoopTimer(over_s=0.1)\n"
   "    _loop_seq2 = 0\n")
ia("        while rclpy.ok():\n",
   "            _w2_loop0 = time.monotonic()                                       # W1b2 r2：本輪迴圈起點（回呼處理之前）\n"
   "            _loop_seq2 += 1\n")
open('door_k3m_task_node_w1b2.py', 'w').write(out)
moved = {l for l in D2_OLD.splitlines()}
removed = set(D1.splitlines())
a_, b_ = src.splitlines(), out.splitlines()
j = 0
for ln in a_:
    if ln in removed or (ln in moved and ln.strip() not in ("rep['received_any_sync'] = sync_once",)):
        continue
    while b_[j] != ln:
        j += 1
    j += 1
assert all(l in b_ for l in moved if l.strip() != 'save()') and b_.count('        save()') >= 1
print('D1/D2 only; rest insert-only OK', len(a_), '→', len(b_))
