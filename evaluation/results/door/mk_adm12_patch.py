"""ADM11 → ADM12（K3-B 受控釋放：mode 3 退離候選與 HELD 交接；Codex 20261026_000706 D1；設計 K3B_impl_design_draft1.md §2）：
只插入（每個錨點恰好一次；核對 ADM11 每行依序在新檔中）。--w2-retreat 預設關 ⇒ 與 ADM11 同行為。
"""
src = open('dl3_wgmpc_wg2_adm11_node.py').read()
out = src


def ib(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, text + anchor)


def ia(anchor, text):
    global out
    assert out.count(anchor) == 1, anchor
    out = out.replace(anchor, anchor + text)


ib('# ADM11（W2 保持／待交接；',
   '# ADM12（K3-B 受控釋放：退離模式；Codex 20261026_000706，設計 K3B_impl_design_draft1.md，規格 K3B_release_clear_spec_draft2.md）：\n'
   '#   dl3_wgmpc_wg2_adm11_node.py 的版本化複本，**只做插入**。--w2-retreat（需 --w2-mode-run-id；預設關 ⇒ 與 ADM11 同行為）：\n'
   '#   (1) 模式序列延伸到 2 RELEASE_OPEN（保持形，旁路候選同 ADM11）與 3 RETREAT（door_w2_mode5）；監督器改 HeldSupervisorP4b5。\n'
   '#   (2) mode 3 候選：準備期間於一般發布之後，以**話題退離目標**對同一快照、同一 u_prev、同一 base_hold 另解一次（冷啟動，不改暖啟動、\n'
   '#       偏移估計、u_prev、一般流目標），通過既有 publish_feasible 才以獨立 source_seq 發 v3 mode 3；**不寫入延遲預測歷史**，\n'
   '#       只有執行端回報該 source_seq 已實際套用（/door/applied_src）時才插入歷史（不同內容的候選不當作已套用命令）。\n'
   '#   (3) 權威狀態顯示已提交 mode 3，或執行端回報 mode 3 候選已實際套用（先到者；後者在監督器核對該筆之前）⇒ 監督器 handover\n'
   '#       （HELD 結束、held_end handover）；之後目標＝話題退離目標，\n'
   '#       底盤仍為原暫停錨點的保持伺服（不重設錨點、不回 PARK_HOLD 路徑），手臂不再歸零。\n')
ia('import cmd_envelope_v3 as E3                                     # noqa: E402  ADM11：v3 封裝（mode_seq）\n',
   'import door_w2_mode5 as W2M5                                     # noqa: E402  ADM12：多段模式、HELD 交接、模式來源 r5\n')
ia("        self._w2_pub = {'n_by_mode': {}, 'first_src_by_mode': {}, 'last': None}\n",
   "        self._w2r = None                                                               # ADM12：退離模式（預設關）\n"
   "        if getattr(a, 'w2_retreat', False) and self._pause_sw is not None:\n"
   "            self._pause_sup = W2M5.HeldSupervisorP4b5(arm_win_s=a.pause_arm_win_s, held_max_s=a.w2_held_max_s)\n"
   "            self._w2src = W2M5.SolverModeSource5(str(a.w2_mode_run_id))\n"
   "            self._w2r = {'handover': None, 'cand_hist': {}, 'n_cand_solve': 0, 'n_cand_fail': 0, 'n_cand_pub': 0, 'n_hist_insert': 0,\n"
   "                         'n_retreat_cycles': 0, 'target_src': {}, 'first_cand_seq': None, 'cand_fail_why': []}\n")
ib("    ap.add_argument('--w2-mode-run-id', default='', help='ADM11：模式來源 run_id（需 --w2-held 與 --cmd-env）')\n",
   "    ap.add_argument('--w2-retreat', action='store_true', help='ADM12：模式延伸到 RELEASE_OPEN／RETREAT（需 --w2-mode-run-id）')\n")
ia("    if a.w2_mode_run_id and not (a.w2_held and a.cmd_env):\n        ap.error('--w2-mode-run-id 需要 --w2-held 與 --cmd-env')\n",
   "    if getattr(a, 'w2_retreat', False) and not a.w2_mode_run_id:\n"
   "        ap.error('--w2-retreat 需要 --w2-mode-run-id')\n")
ib("            if self._pause_active():                                                  # ADM4：凍結目標（不追隨新目標）\n",
   "            _T12, _src12 = T_cyc, _tgt_src                                              # ADM12：話題目標（退離用，暫停凍結前）\n"
   "            if self._w2r is not None and self._w2r['handover'] is None and self._w2src.committed >= 3 \\\n"
   "                    and self._pause_sup.state == 'HELD' and self._pause_sup.handover(self.sim_now()):\n"
   "                self._w2r['handover'] = {'sim_t': self.sim_now(), 'slot': slot,\n"
   "                                         'anchor': (None if self._pause.get('anchor') is None else [float(x) for x in self._pause['anchor']])}\n"
   "                print(f'[wg2] **ADM12 HELD 交接 → 退離**（錨點不變）@ slot {slot}', flush=True)\n")
ia("                T_cyc, _tgt_src = np.array(self._pause['T'], float), 'pause_frozen'\n",
   "            if self._w2r is not None and self._w2r['handover'] is not None:              # ADM12：交接後追退離目標（錨點不變）\n"
   "                T_cyc, _tgt_src = np.array(_T12, float), f'w2_retreat:{_src12}'\n"
   "                self._w2r['n_retreat_cycles'] += 1\n"
   "                self._w2r['target_src'][_src12] = self._w2r['target_src'].get(_src12, 0) + 1\n")
ib("            _zero4 = False\n",
   "            _u_out12 = None if _u_out is None else np.asarray(_u_out, float).copy()     # ADM12：暫停處理前的求解輸出\n")
ib("                rec['pause_u_solve'] = [float(x) for x in np.asarray(r.u0, float)]       # r2：QP 求解值\n",
   "                if self._w2r is not None and self._w2r['handover'] is not None and _u_out12 is not None:   # ADM12：交接後手臂不歸零\n"
   "                    _u_out = _u_out12\n"
   "                    rec['w2_retreat_out'] = True\n")
ia("            rec['publish_sim_t_exact'] = self._last_pub_t\n",
   "            if self._w2r is not None and self._w2src.tag_candidate3() == 3:              # ADM12：mode 3 候選另解並另發（不改一般流）\n"
   "                self._w2r['n_cand_solve'] += 1\n"
   "                _why12 = None\n"
   "                try:\n"
   "                    if self._gate is not None:\n"
   "                        _rc12 = solve_sp(self.K, make_z(_q_sol, _s_sol), u_prev, np.array(_T12, float), self.cfg, U_warm=None)\n"
   "                    else:\n"
   "                        _rc12 = solve(self.K, q0, u_prev, np.array(_T12, float), self.cfg, U_warm=None)\n"
   "                except Exception as _e12:                                             # noqa: BLE001\n"
   "                    _rc12, _why12 = None, f'solve_error:{_e12!r}'[:120]\n"
   "                if _rc12 is not None and not _rc12.ok:\n"
   "                    _why12 = f'solve_not_ok:{_rc12.reason}'\n"
   "                if _why12 is None:\n"
   "                    _uc12 = np.asarray(_rc12.u0, float)\n"
   "                    _fk12, _fw12 = PSUP.publish_feasible(_uc12, u_prev, self.cfg)\n"
   "                    if not _fk12:\n"
   "                        _why12 = 'infeasible:' + '+'.join(_fw12)\n"
   "                if _why12 is None:\n"
   "                    _uw12 = body_to_world(float(q0[2])) @ _uc12\n"
   "                    self._source_seq += 1\n"
   "                    _tc12 = self.sim_now()\n"
   "                    _e3c12 = Float64MultiArray()\n"
   "                    _e3c12.data = E3.encode3(self._run_id_n, self._source_seq, ENV.ST_SOLVER, self._source_seq, self._source_seq, True,\n"
   "                                             _tc12, _uw12, 3)\n"
   "                    self._env3_pub.publish(_e3c12)\n"
   "                    self._w2r['cand_hist'][int(self._source_seq)] = (float(_tc12), _uc12.copy())\n"
   "                    while len(self._w2r['cand_hist']) > 64:\n"
   "                        self._w2r['cand_hist'].pop(next(iter(self._w2r['cand_hist'])))\n"
   "                    self._w2r['n_cand_pub'] += 1\n"
   "                    if self._w2r['first_cand_seq'] is None:\n"
   "                        self._w2r['first_cand_seq'] = int(self._source_seq)\n"
   "                    self._w2_pub['n_by_mode'][3] = self._w2_pub['n_by_mode'].get(3, 0) + 1\n"
   "                    self._w2_pub['first_src_by_mode'].setdefault(3, int(self._source_seq))\n"
   "                    self._w2_pub['n_candidate'] = self._w2_pub.get('n_candidate', 0) + 1\n"
   "                    rec['w2_cand3'] = {'source_seq': int(self._source_seq), 'u': [float(x) for x in _uc12]}\n"
   "                else:\n"
   "                    self._w2r['n_cand_fail'] += 1\n"
   "                    if len(self._w2r['cand_fail_why']) < 20:\n"
   "                        self._w2r['cand_fail_why'].append({'slot': slot, 'why': _why12})\n")
ia("    def _on_applied_src(self, m):                                                     # ADM4：逐物理步套用來源\n",
   "        if self._w2r is not None:                                                       # ADM12：候選經回報實際套用後才進延遲預測歷史\n"
   "            try:\n"
   "                _sq12 = int(float(list(m.data)[2]))\n"
   "            except (TypeError, ValueError, IndexError, OverflowError):\n"
   "                _sq12 = -1\n"
   "            _c12 = self._w2r['cand_hist'].pop(_sq12, None) if _sq12 >= 0 else None\n"
   "            if _c12 is not None:\n"
   "                _i12 = len(self._cmd_hist)\n"
   "                while _i12 > 0 and self._cmd_hist[_i12 - 1][0] > _c12[0]:\n"
   "                    _i12 -= 1\n"
   "                self._cmd_hist.insert(_i12, (_c12[0], np.asarray(_c12[1], float).copy(), _sq12))\n"
   "                self._w2r['n_hist_insert'] += 1\n"
   "                if self._w2r['handover'] is None and self._pause_sup is not None and self._pause_sup.state == 'HELD' \\\n"
   "                        and self._pause_sup.handover(self.sim_now()):                    # 候選被實際套用＝執行端已提交 ⇒ 先交接再核\n"
   "                    self._w2r['handover'] = {'sim_t': self.sim_now(), 'slot': None, 'by': 'applied_candidate', 'source_seq': _sq12,\n"
   "                                             'anchor': (None if self._pause.get('anchor') is None else [float(x) for x in self._pause['anchor']])}\n")
ia("                         'last_pub': self._w2_pub['last'], 'n_candidate': self._w2_pub.get('n_candidate', 0)}),\n",
   "                    w2_retreat=(None if self._w2r is None else                          # ADM12\n"
   "                                dict({k: v for k, v in self._w2r.items() if k != 'cand_hist'}, n_cand_hist_left=len(self._w2r['cand_hist']))),\n")
open('dl3_wgmpc_wg2_adm12_node.py', 'w').write(out)

a_, b_ = src.splitlines(), out.splitlines()
j = 0
for line in a_:
    while j < len(b_) and b_[j] != line:
        j += 1
    assert j < len(b_), f'原行遺失：{line!r}'
    j += 1
print('insert-only ok：', len(a_), '→', len(b_))
