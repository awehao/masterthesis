"""ADM12 → ADM13（K3-B：mode 2 實際提交點切換為釋放保持監看；Codex 20261026_021025 §2）：
只插入（每個錨點恰好一次；核對 ADM12 每行依序在新檔中）。--w2-release-hold 預設關 ⇒ 與 ADM12 同行為。
"""
src = open('dl3_wgmpc_wg2_adm12_node.py').read()
out = src


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


ib('# ADM12（K3-B 受控釋放：退離模式；',
   '# ADM13（K3-B：mode 2 釋放保持監看交接；Codex 20261026_021025 §2，door_w2_mode6）：\n'
   '#   dl3_wgmpc_wg2_adm12_node.py 的版本化複本，**只做插入**。--w2-release-hold（需 --w2-retreat；預設關 ⇒ 與 ADM12 同行為）：\n'
   '#   (1) 監督器改 HeldSupervisorP4b6：外層仍為 HELD（請求受理與 mode 3 交接條件不變），內部於 mode 2 實際提交點切換為 release_open：\n'
   '#       底盤、手臂仍用舊 HELD 門檻且持續發布保持命令；門速改為保護上限 --w2-release-door-rate（預設 0.05 rad/s＝任務端 door_abort_rate）。\n'
   '#   (2) 得知提交：權威狀態 mode_seq ≥ 2 時，以狀態訊息 commits 列的 mode 2 提交時刻呼叫 release_open（提交前窗口門速須低）。\n'
   '#   (3) 狀態延遲／ack 遺失：已受理 mode 2 請求（prep 或已見提交）時設 pending2；只有門超舊門檻時暫緩判定 ≤ 0.2 s，逾時照舊 FAIL。\n'
   '#   門角帶、幾何、資料新鮮度仍由任務端／執行端負責，本節點不取代。\n')
ia('import door_w2_mode5 as W2M5                                     # noqa: E402  ADM12：多段模式、HELD 交接、模式來源 r5\n',
   'import door_w2_mode6 as W2M6                                     # noqa: E402  ADM13：mode 2 釋放保持監看\n')
ia("                         'n_retreat_cycles': 0, 'target_src': {}, 'first_cand_seq': None, 'cand_fail_why': []}\n",
   "        self._w2rh = None                                                              # ADM13：mode 2 釋放保持（預設關）\n"
   "        if getattr(a, 'w2_release_hold', False) and self._w2r is not None:\n"
   "            self._pause_sup = W2M6.HeldSupervisorP4b6(arm_win_s=a.pause_arm_win_s, held_max_s=a.w2_held_max_s,\n"
   "                                                      door_rate_release=a.w2_release_door_rate)\n"
   "            self._w2rh = {'release_open': None, 'pending2_events': [], 'n_commit2_no_t': 0}\n")
ib("    ap.add_argument('--w2-retreat', action='store_true', help='ADM12：模式延伸到 RELEASE_OPEN／RETREAT（需 --w2-mode-run-id）')\n",
   "    ap.add_argument('--w2-release-hold', action='store_true', help='ADM13：mode 2 提交點切換釋放保持監看（需 --w2-retreat）')\n"
   "    ap.add_argument('--w2-release-door-rate', type=float, default=W2M6.DOOR_RATE_RELEASE, help='ADM13：釋放保持門速保護上限 rad/s')\n")
ia("    if getattr(a, 'w2_retreat', False) and not a.w2_mode_run_id:\n"
   "        ap.error('--w2-retreat 需要 --w2-mode-run-id')\n",
   "    if getattr(a, 'w2_release_hold', False) and not getattr(a, 'w2_retreat', False):\n"
   "        ap.error('--w2-release-hold 需要 --w2-retreat')\n")
ia("        self._w2src.on_request(msg, held=(self._pause_sup is not None and self._pause_sup.state == 'HELD'), t=self.sim_now())\n",
   "        if self._w2rh is not None:                                                      # ADM13：受理 mode 2 即設 pending2\n"
   "            self._w2rh_pending()\n")
ia("            self._w2src.on_state(json.loads(m.data), self.sim_now())\n",
   "            if self._w2rh is not None:                                                  # ADM13：得知 mode 2 提交 ⇒ 切換釋放保持\n"
   "                self._w2rh_state(json.loads(m.data))\n")
ib("    def _on_w2_state(self, m):                                                        # ADM11：權威狀態為準\n",
   "    def _w2rh_pending(self):                                                          # ADM13\n"
   "        _p13 = self._w2src.committed >= 2 or (self._w2src.prep is not None and self._w2src.prep.get('mode_seq') == 2)\n"
   "        if _p13 != self._pause_sup.pending2 and len(self._w2rh['pending2_events']) < 20:\n"
   "            self._w2rh['pending2_events'].append({'sim_t': self.sim_now(), 'pending2': bool(_p13)})\n"
   "        self._pause_sup.set_pending2(_p13, self.sim_now())\n"
   "\n"
   "    def _w2rh_state(self, st):                                                        # ADM13\n"
   "        self._w2rh_pending()\n"
   "        if self._w2src.committed < 2 or self._pause_sup.state != 'HELD' or self._pause_sup.sub != 'held':\n"
   "            return\n"
   "        _tc13 = W2M6.commit_t_of(st, 2)\n"
   "        if _tc13 is None:                                                              # 無提交時刻 ⇒ 不切換（暫緩逾時照舊 FAIL）\n"
   "            self._w2rh['n_commit2_no_t'] += 1\n"
   "            return\n"
   "        _ok13 = self._pause_sup.release_open(_tc13, self.sim_now())\n"
   "        self._w2rh['release_open'] = {'sim_t': self.sim_now(), 't_commit': _tc13, 'ok': bool(_ok13),\n"
   "                                      'checks': self._pause_sup.release_checks}\n"
   "        print(f'[wg2] **ADM13 mode 2 提交 {_tc13:.3f} → 釋放保持** ok={_ok13}', flush=True)\n"
   "\n")
ia("                                dict({k: v for k, v in self._w2r.items() if k != 'cand_hist'}, n_cand_hist_left=len(self._w2r['cand_hist']))),\n",
   "                    w2_release_hold=(None if self._w2rh is None else dict(self._w2rh)),     # ADM13\n")
open('dl3_wgmpc_wg2_adm13_node.py', 'w').write(out)

a_, b_ = src.splitlines(), out.splitlines()
j = 0
for line in a_:
    while j < len(b_) and b_[j] != line:
        j += 1
    assert j < len(b_), f'原行遺失：{line!r}'
    j += 1
print('insert-only ok：', len(a_), '→', len(b_))
