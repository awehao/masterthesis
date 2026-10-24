# W1c：夾持期間偏移估計保持（單一變因候選）——差異規格＋實作＋測試＋凍結 w1c r1（**未開跑**）

依 Codex 20261025_002129：
- 可進入版本化規格＋實作＋測試；
- 沿用 30° 與 W1b r2 其餘全部設定；
- **單一變因候選，不是已確認的解法**；本輪尚不核可實跑。

## 0. 依據（W1b_30_1_load_analysis.md r2，只描述）
- door_wide_w1b_30_1：OPEN 中 F2−F1 由約 −2.7 增到 +7.3 N，自由模型等效位姿差指標的垂直分量由 +8.7 增到 +14.9 mm，實測 TCP 相對把手幾乎不動。
- 偏移觀測器自 ENGAGE 起更新 361 次，已發布 d̂ j2 16.84 → 26.2 mrad。
- 相關接近同義反覆，不是獨立成因證據。
- K3-A 三種偏移估計變體在夾持期間 d̂ 都仍更新，「夾持期間保持」未實跑過。

## 1. 唯一差異
| 項目 | W1b r2 | W1c |
|---|---|---|
| 求解節點 | ADM8 | **ADM9**（ADM8 只插入，原行改動 0）＋`--offset-hold-on-free` |
| 偏移估計更新 | OPEN 後持續更新 | 底盤模式首次切入 free 的那一輪（本輪觀測器更新之前）凍結，之後保持到 PAUSE／DONE 或中止 |

其餘完全相同：θ 30°、OPEN 逾時 52 s、延伸走廊表與內帶、包絡 r2、觀察器 34°、引導 w_vref 1、p4b 暫停、權重、夾持力、摩擦、全部保護。

## 2. ADM9 保持語意
- **凍結值**：當時已可取得的 d̂（上一輪更新後）；不事後回選。
- **真正停止更新**：之後每輪以阻擋理由 `offset_hold` 呼叫既有 IntervalFeed，保持 d̂ 並清錨點。
  - 觀測器內部 d̂、延遲預測與 QP 用的偏差 b＝α⊙d̂ 同步自同一保持值，不是只凍結輸出。
  - 停止、失效、閘門非 ARMED 仍優先。
- **不恢復**：進 PAUSE 不重新學習夾持負載。
- **紀錄** `stats.offset_hold`：凍結 sim_t、快照步號、凍結值、凍結前更新數、首筆受影響的已發布解（slot、publish_sim_t_exact、送入求解的 d̂）；已發布列標 `offset_hold`。

## 3. 判定器 w1c（w1b r2 全部照用＋保持證據；缺證據不通過）
- **H1**：stats.offset_hold 欄位完整且有限。
- **H2**：凍結時刻在首次 applied_free 起 0.06 s 內。
- **H3**：凍結起每列 blocked:*、無 updated、d_after＝凍結值；凍結值＝凍結前最後一筆。
- **H4**：首筆受影響已發布解在凍結後且 d̂＝凍結值；凍結後已發布列都標 offset_hold。
- **H5**：報告更新數＝凍結前更新數。

## 4. 測試（系統 Python，全過）
| 測試 | 項目 |
|---|---|
| `test_dl3_wgmpc_wg2_adm9_hold_ros.py` | 假世界（free 3.0 s、pause 5.0 s）。**H1** 啟用：凍結前有更新；首次 free 那一輪凍結（sim_t 3.04、步號 304）；之後 273 列皆 blocked:offset_hold、d_after＝凍結值、無更新；首筆受影響已發布解在凍結後；報告更新數＝凍結前 40；暫停照常 DONE。**H0** 未啟用：free 後仍更新（原行為），無凍結紀錄。**H2** 保持中停止：停止原因 stop_topic、停止後無任何紀錄列。**H3** 缺 interval ⇒ 參數錯誤。替身無偏移，凍結值為 0，只核語意與紀錄 |
| `test_door_k3m_check_w1c.py` 8 | 夾具＝ADM9 假世界 H1／H0 實際求解紀錄：一致 ⇒ 通過；缺紀錄或 H0 ⇒ missing；凍結後插入 updated、d_after 改變 ⇒ not_held；凍結時刻偏離 ⇒ boundary；first_published 缺或早於凍結 ⇒ invalid；更新數不符；凍結值非最後一筆；登錄未含 solver_change ⇒ 不通過 |
| 回歸 | w1b 判定器、任務節點 w1b（含 main 啟動）、ADM8 協同與暫停假世界 |

## 5. 申請單趟 `door_wide_w1c_30_1`（凍結 w1c r1）
- 乾淨環境、92 °C；失敗保留不補跑。
- **分報**：開門（θ）、承載（F1／F2、F2−F1、承載點）、追蹤（抓取漂移、指標）、引導與走廊、SETTLE、主動保持命令確認、實際停止、保護、偏移估計保持證據。
- 附 side／close 影片。
- 舊幾何餘裕只作參考，不當成新軌跡的保證。

## 6. r2（Codex 20261025_003515 條件式核可：只補判定器與測試，不改控制律）
- 判定器 w1c r2 補三個誤收缺口：
  - **H3'**：凍結起每個實際求解列（帶 solve_in）都必須有 offset_interval，不得篩掉；不帶 solve_in 的診斷列（暫停觀察）另列計數；已發布列必須是求解列。
  - **H6**：實際使用值 solve_in.offset_d_hat＝d_frozen、arm_bias＝α⊙d_frozen；α 取求解端 args.arm_ident，讀不到 ⇒ 不通過。
  - **H4'**：first_published 須精確對上凍結後第一筆真實已發布列的 slot 與 publish_sim_t_exact，且 d_hat_in＝該列 solve_in.offset_d_hat＝d_frozen。
- `test_door_k3m_check_w1c.py` 12 項：原 8 項，加 K10 刪已發布列 offset_interval、K11 改 offset_d_hat／arm_bias、K12 first_published slot／時間不存在、K13 診斷列另計。
- 新增 `test_dl3_adm9_offset_hold_logic.py` 3 項：**六軸非零 d̂**，直接取用 ADM9 節點的 `_offset_update_interval`，配真 IntervalOffsetObserver／IntervalFeed。
  - 凍結值＝當時非零 d̂；之後 20 輪 blocked:offset_hold；觀測器內部 d、_of_d、bias＝α⊙d 都不變；更新計數不變；錨點清除。
  - 保持中停止 ⇒ stop_rx 優先、值仍保持；未啟用不凍結。
- ADM9 控制邏輯、跑法、登錄與其餘設定**與 r1 相同**（r1 凍結清單中除判定器與測試外逐檔雜湊不變）。
- 凍結 `freeze_door_wide_w1c_dev_r2.sha256`。
