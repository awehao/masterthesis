"""ADM12 → ADM16（K3-B：監督器 r9；Codex 20261026_034842）：
只插入（每個錨點恰好一次；核對 ADM12 每行依序在新檔中）。--w2-hold-r9 預設關 ⇒ 與 ADM12 同行為。不以 ADM13／ADM14 為基底（ADM14 的掛鉤邏輯沿用並擴充）。
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
   '# ADM16（K3-B：監督器 r9；Codex 20261026_034842，door_w2_mode9）：與 ADM15 同一組插入，唯監督器改 HeldSupervisorP4b9\n'
   '#   （瞬態 r2：無效樣本中斷收斂證據；延遲得知提交時重放緩衝內提交後樣本）。以下各項同 ADM15：\n'
   '#   dl3_wgmpc_wg2_adm12_node.py 的版本化複本，**只做插入**（不以 r6／r7 掛鉤的節點版本為基底）。--w2-hold-r9（需 --w2-retreat；預設關 ⇒ 與 ADM12 同行為）：\n'
   '#   (1) 監督器改 HeldSupervisorP4b8（外層 HELD；held →〔--w2-unload：unload → unloaded〕→ release_open → release_settled；\n'
   '#       瞬態監看 TCP 以本節點 WholeBodyKinematics 計算；mode 3 交接只准自 release_settled）。\n'
   '#   (1b) 已驗證的本趟權威狀態 unload.t_settling（斜坡完成、讀回 5 N）⇒ set_exec_settling，重新收斂窗自此起算。\n'
   '#   (2) 提交通知只取**已驗證**的本趟權威狀態：run_id 相符、未中止、commits 列 mode_seq＝k 且 applied_confirmed True、\n'
   '#       對應 commit_req_seqs＝求解端受理的 req_seq、受理時刻 ≤ 提交時刻 ≤ 現在（door_w2_mode7.validated_commit）；他趟訊息一律不進入。\n'
   '#   (3) 受理請求即設 pending[k]；監督器於通知當下核暫緩期限、提交前證據，超限記錄不抹除。\n')
ia('import door_w2_mode5 as W2M5                                     # noqa: E402  ADM12：多段模式、HELD 交接、模式來源 r5\n',
   'import door_w2_mode7 as W2M7                                     # noqa: E402  ADM16：validated_commit\n'
   'import door_w2_mode9 as W2M9                                     # noqa: E402  ADM16：監督器 r9（瞬態 r2、延遲得知重放）\n')
ia("                         'n_retreat_cycles': 0, 'target_src': {}, 'first_cand_seq': None, 'cand_fail_why': []}\n",
   "        self._w2r7 = None                                                              # ADM16（預設關）\n"
   "        if getattr(a, 'w2_hold_r9', False) and self._w2r is not None:\n"
   "            _K15, _tcp15 = self.K, a.tcp\n"
   "            self._pause_sup = W2M9.HeldSupervisorP4b9(arm_win_s=a.pause_arm_win_s, held_max_s=a.w2_held_max_s, unload=a.w2_unload,\n"
   "                                                      door_rate_protect=a.w2_release_door_rate,\n"
   "                                                      tcp_fn=lambda _b, _q: _K15.fk(np.asarray(list(_b) + list(_q), float), _tcp15))\n"
   "            self._w2r7 = {'req_seq': {}, 't_prep': {}, 'commits': {}, 'rejected': {}, 'n_other_run': 0}\n")
ib("    ap.add_argument('--w2-retreat', action='store_true', help='ADM12：模式延伸到 RELEASE_OPEN／RETREAT（需 --w2-mode-run-id）')\n",
   "    ap.add_argument('--w2-hold-r9', action='store_true', help='ADM16：HELD 監督 r9（需 --w2-retreat）')\n"
   "    ap.add_argument('--w2-unload', action='store_true', help='ADM16：mode 1 為閉爪卸載候選（需 --w2-hold-r9）')\n"
   "    ap.add_argument('--w2-release-door-rate', type=float, default=W2M7.DOOR_RATE_PROTECT, help='ADM16：門角差分速度保護上限 rad/s')\n")
ia("    if getattr(a, 'w2_retreat', False) and not a.w2_mode_run_id:\n"
   "        ap.error('--w2-retreat 需要 --w2-mode-run-id')\n",
   "    if getattr(a, 'w2_hold_r9', False) and not getattr(a, 'w2_retreat', False):\n"
   "        ap.error('--w2-hold-r9 需要 --w2-retreat')\n"
   "    if getattr(a, 'w2_unload', False) and not getattr(a, 'w2_hold_r9', False):\n"
   "        ap.error('--w2-unload 需要 --w2-hold-r9')\n")
ia("        self._w2src.on_request(msg, held=(self._pause_sup is not None and self._pause_sup.state == 'HELD'), t=self.sim_now())\n",
   "        if self._w2r7 is not None:                                                      # ADM16：受理即設 pending\n"
   "            self._w2r7_pending()\n")
ia("            self._w2src.on_state(json.loads(m.data), self.sim_now())\n",
   "            if self._w2r7 is not None:                                                  # ADM16：只用已驗證的本趟權威狀態\n"
   "                self._w2r7_state(json.loads(m.data))\n")
ib("    def _on_w2_state(self, m):                                                        # ADM11：權威狀態為準\n",
   "    def _w2r7_pending(self):                                                          # ADM16\n"
   "        _p = self._w2src.prep\n"
   "        for _k in (1, 2):\n"
   "            if _p is not None and _p.get('mode_seq') == _k and self._w2r7['req_seq'].get(_k) is None:\n"
   "                self._w2r7['req_seq'][_k], self._w2r7['t_prep'][_k] = _p.get('req_seq'), self.sim_now()\n"
   "            self._pause_sup.set_pending(_k, (_p is not None and _p.get('mode_seq') == _k) or self._w2src.committed >= _k, self.sim_now())\n"
   "\n"
   "    def _w2r7_state(self, st):                                                        # ADM16\n"
   "        if not isinstance(st, dict) or st.get('run_id') != self._w2src.run_id:\n"
   "            self._w2r7['n_other_run'] += 1\n"
   "            return\n"
   "        self._w2r7_pending()\n"
   "        _u15 = W2M9.exec_unload_of(st)\n"
   "        if _u15 is not None and self._pause_sup.sub == 'unload' and self._pause_sup.t_settling is None and isinstance(_u15.get('t_settling'), (int, float)) and math.isfinite(_u15['t_settling']):\n"
   "            self._w2r7['settling'] = {'sim_t': self.sim_now(), 't_settling': _u15['t_settling'],\n"
   "                                      'ok': bool(self._pause_sup.set_exec_settling(float(_u15['t_settling']), self.sim_now()))}\n"
   "        _k = self._pause_sup.next_mode()\n"
   "        if _k is None or self._pause_sup.state != 'HELD':\n"
   "            return\n"
   "        _tc, _why = W2M7.validated_commit(st, self._w2src.run_id, _k, self._w2r7['req_seq'].get(_k), self._w2r7['t_prep'].get(_k), self.sim_now())\n"
   "        if _tc is None:\n"
   "            if _why != 'not_committed':\n"
   "                _r = self._w2r7['rejected'].setdefault(str(_k), {})\n"
   "                _r[_why] = _r.get(_why, 0) + 1\n"
   "            return\n"
   "        _ok = self._pause_sup.on_commit(_k, _tc, self.sim_now())\n"
   "        self._w2r7['commits'][str(_k)] = {'sim_t': self.sim_now(), 't_commit': _tc, 'ok': bool(_ok)}\n"
   "        print(f'[wg2] **ADM16 mode {_k} 提交 {_tc:.3f} → {self._pause_sup.sub}** ok={_ok}', flush=True)\n"
   "\n")
ia("                                dict({k: v for k, v in self._w2r.items() if k != 'cand_hist'}, n_cand_hist_left=len(self._w2r['cand_hist']))),\n",
   "                    w2_r7=(None if self._w2r7 is None else dict(self._w2r7)),                 # ADM16\n")
open('dl3_wgmpc_wg2_adm16_node.py', 'w').write(out)

a_, b_ = src.splitlines(), out.splitlines()
j = 0
for line in a_:
    while j < len(b_) and b_[j] != line:
        j += 1
    assert j < len(b_), f'原行遺失：{line!r}'
    j += 1
print('insert-only ok：', len(a_), '→', len(b_))
