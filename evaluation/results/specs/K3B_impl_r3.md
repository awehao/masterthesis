# K3-B 受控釋放實作 r3（Codex 20261026_012939 最後必修：終點收斂改三維工具位移）

r1、r2（`K3B_impl_r1.md`、`K3B_impl_r2.md`；凍結 k3b r1、r2）不改，本文件為 r3 增補。
- Codex 012939 裁定：
  - r2 大致補足；5 mm 追蹤落後界限與帶 r3 可用於本次單趟隔離實跑。
  - 最後一項必修：終點收斂須檢查三維工具位移，而非只看 s_meas 投影。
  - 補完、反例與回歸通過、建立新版本與新凍結並核對後，條件式核可單趟 `door_wide_k3b_30_1`。
- 帶 r3 的定位（依 Codex）：
  - v4e 是 0.5 mm 細格**取樣**驗證，不是連續證明；
  - 最薄餘量（退出後配對高於 5 mm 約 0.007 mm，s 0.06、Δ＋3.0°）**不作安全宣稱**。

## 1. 版本化（r2 檔案一律不改）
| r3 檔案 | 取代 | 差異 |
|---|---|---|
| `door_k3b_release4.py` | door_k3b_release3 | 終點收斂三維化（§2） |
| `door_k3m_policy_k3b3.py`（`K3MReleasePolicy3`） | policy_k3b2 | 改用 release4，其餘相同 |
| `door_k3m_task_node_k3b3.py`（`mk_task_k3b3_patch.py`；對 k3b2 與 w1b2 皆只插入，648 行） | task_node_k3b2 | 改建策略 r3 |
| `door_k3m_check_k3b3.py` | check_k3b2 | K13 改三維重算（§3），其餘沿用 check_k3b2 全部項目 |
| `run_door_wide_k3b_r3.sh` | run_door_wide_k3b_r2 | 只換任務節點 k3b3、判定器 k3b3、凍結 k3b r3；執行端仍 ev7r2、求解端仍 ADM12 |

執行端 ev7r2、手指閘門、求解端 ADM12 都沒有變動，沿用凍結 k3b r2 的內容。

## 2. 終點收斂改三維（release4）
- r3 的問題：TaskReleaseMonitor 的收斂與保持只核 0.5 s 窗內 s_meas 投影的位移。工具若沿把手 x 或 z 振盪，投影不變，就會漏檢。
- r4 的做法：
  - 監看軌跡記錄 `(t, s_meas, p3)`，p3 是固定 H(q_rel) 座標中實測 TCP 的三維位置。
  - 「位移」改為 0.5 s 窗內三維**最大點對距離**，門檻仍為 ≤ 1 mm。
- 適用範圍：
  - 進入 RETREAT_DONE 前的收斂判斷；
  - RETREAT_DONE 期間的保持（`end_hold_moved`）。
- 不變的部分：
  - 姿態偏移界與其他門檻；
  - 不新增門檻；
  - 離區（s_meas < 0.095 ⇒ `end_hold_left_zone`）、資料中斷計時重起、連續新鮮 ≥ 3 s 才 COMPLETE。

## 3. 判定器 K13 三維重算（check_k3b3）
- 由 room 逐物理步 FK 算出實測 TCP 在 H(q_rel) 座標的三維位置。
- 重算範圍：從 RETREAT_DONE 前 0.5 s 起，到 COMPLETE 為止的每個 0.5 s 窗，最大點對距離 ≤ 1 mm。
- 拒收理由：
  - `K13_end_hold_moved_3d`（窗內再動）；
  - `K13_end_hold_left_zone`（s_meas < 0.095）；
  - `K13_end_hold_coverage`（逐步紀錄未覆蓋保持期）；
  - `K13_end_hold_missing`（缺 RETREAT_DONE 或 COMPLETE 時刻）。
- 輸出：`analysis/door_k3m_check_k3b3.json`。

## 4. 反例與回歸（均不需 Isaac）
**r3 新增**
- `test_door_k3b_release4.py`：17 項全過（r3 的 16 項加 T7）。
  - T7：重現 Codex 指出的情形，終點沿 x ±2.5 mm（或 z ±0.8 mm）振盪，s_meas 投影不變。r3 仍會 COMPLETE。
  - r4 的結果：進入前就振盪 ⇒ 不進 RETREAT_DONE；保持中才開始振盪 ⇒ `end_hold_moved`；無振盪的正例照常 COMPLETE。
- `test_door_k3m_check_k3b3.py`：9 組全過。
  - Q0–Q10 沿用 r2；Q8 改為預期 `K13_end_hold_moved_3d`。
  - Q11：終點 x／z 振盪 ⇒ r3 判定器拒收；r2 判定器誤收（對照）。
  - **測試用合成資料，不是實跑結果。**
- `test_door_k3m_policy_k3b3.py`：5 組全過。
- `test_k3b_wiring_static_r3.py`：7 項全過。其中重新產生 mk_task_k3b3 與現檔逐位元相同、只插入；W7 核對 k3b3 匯入 r4。

**真任務節點 r3 路由**（`test_task_node_k3b3_routing.py`）：N1–N3 全過。
- N1：DONE、exit 0；請求序 1、2、3。
  - 5 N 證據 s 20.8 mm、配對 13.2 mm、band_ok；
  - monitor COMPLETE、讀回 True；s_cmd 單調到 0.10、離線 IK 169 次無失敗；
  - 近區配對最小 5.69 mm、非手指 7.96 mm、帶餘量 +0.1°。
- N2：5 N 被拒 ⇒ 中止 `k3b_effort_reject:evidence_not_clear`、發 /door/k2_abort。
- N3：確認逾時 ⇒ 中止 `k3b_exec_release:confirm_timeout`、未請求 mode 3。
- 本測試預設 domain 原為 240，超出 ROS 2 上限（232），起始失敗；已改為 211 後執行。

**r2 回歸**：release3 16、policy_k3b2 5、check_k3b2 8、靜態 r2 6、grip_gate 4，全過。

## 5. 單趟 `door_wide_k3b_30_1` 執行條件（Codex 012939）
- 任務：30° 開門 → 張爪 → 解除確認 → 退離 → 5 N 恢復 → 終點保持；**不收臂**。
- 環境：乾淨環境（去掉 .venv-dl0、unset VIRTUAL_ENV）、ROS_DOMAIN_ID 97、92 °C 熱守護。
- 凍結：起跑前與結束後都核對凍結 k3b r3。
- 紀錄：失敗也保留，不補跑、不覆寫；不新增影片。
- 分項回報：
  - 釋放、近區退出、5 N、退離；
  - 三維收斂與保持；
  - 保護與紀錄完整性。
