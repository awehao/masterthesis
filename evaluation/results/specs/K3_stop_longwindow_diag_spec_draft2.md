# K3 停止後長窗隔離診斷 draft-2（已實作＋凍結 r1；**未核可實跑**）

依 Codex 20261023_230237 修正 draft-1（`K3_stop_longwindow_diag_spec.md`，不改）。單趟；實作與凍結完成後再送審，通過才另核可。

## 1. 差異表（相對 K3-A-B EV：run_door_k3ab_ev.sh／door_k3ab_ev_1）
| 項目 | K3-A-B EV | LW（本診斷） | 性質 |
|---|---|---|---|
| 執行端、展開、轉接、求解器與全部參數 | — | 同 | 不改 |
| 任務節點 | door_k3ab_task_node.py | `door_k3ab_lw_task_node.py`（只插入，見下兩列；docstring 第一行改寫） | 版本化複本 |
| 中止後觀察 | post_abort_observe_s 3.0（K2Config，從策略中止起算） | `--post-abort-observe-s 10`，同樣從策略中止起算 | 觀察參數 |
| 任務節點收 /door/lw_guard | 無 | 訂閱（latched）；收到 ⇒ 立即 finish，task.json 記 `lw_guard`（訊息＋接收牆鐘） | 新增結束路徑 |
| 獨立觀察器 | 無 | `door_k3ab_lw_observer.py`＋純函式 `door_k3ab_lw_guard.py`（§3） | **新增實驗保護** |
| 跑法 | run_door_k3ab_ev.sh | `run_door_k3ab_lw.sh`：換任務節點、加參數、起觀察器（觀察器 5 s 後未存活 ⇒ exit 86）、收尾 TERM 觀察器並列出 armed／trip／closed、換凍結 | 版本化複本 |
| 判定器 | door_k3ab_check.py | 同（照常判 K3-A-B，預期未完成） | 不改 |
| 只讀分析 | — | `door_k3ab_lw_analysis.py`（§2） | 新增 |
| 凍結 | freeze_door_k3ab_ev_dev_r3（145 檔） | `freeze_door_k3ab_lw_dev_r1`＝r3 清單＋新檔 | — |

## 2. 只讀分析（`door_k3ab_lw_analysis.py`）
**結束分支**（互斥）：
- `observation_terminated_early`：task 有 lw_guard ⇒ 停止判定一律記為 `unverified_observation_terminated_early`；
- `latched`：恰一筆 k2_abort_latched，時間零點取這一步；
- `done_branch_observation_insufficient`：無閂鎖且到 DONE ⇒ 記觀察不足，不補跑；
- `no_latch_other`：其餘。

**停住判定**：
- **速度估計**：取 10 個連續步號的窗口，以實際 Δt 估速度。缺步、重複步、非有限值 ⇒ 該點無效，不補零。
- **門檻**（四項同時）：底盤 ≤ 1 mm/s 且 ≤ 0.005 rad/s、門 ≤ 0.002 rad/s、實測手臂 ≤ 1e-3 rad/s。
- **時刻**：分列首次低速 `first_low_t`、確認那段的起點 `confirm.run_start_t`、滿 1 s 的確認時刻 `confirm.confirm_t`。
- **確認後再動**：記 `re_motion`，結果為 `stopped_confirmed_then_re_motion`；另報窗末狀態。
- **未確認**：記 `not_stopped_within_covered_X_s`，X＝閂鎖到最後一步的實際時間；`window_short` 標出不足 10 s。
- 另報各分量（base／door／arm）的偵測結果，以及逐步速度峰值（相鄰步）。0.1 s 窗口速度只代表窗口平均。

**其他輸出**：
- 幾何沿實測軌跡逐步重算，報取樣最小值；殼—橫桿 < 5 mm 的區段逐段列出（起訖、最小值、當時的非設計接觸、牆接觸與腕力）。只讀，不稱監看觸發。
- 段 A（策略→閂鎖）沿用 door_k3m_stop_diag_r2.seg_a。

## 3. 實驗保護（獨立觀察器；收到 /door/k2_abort 才生效）
| 條件 | 界 | 來源 |
|---|---|---|
| 門角 | ≥ 5° | 本次隔離診斷的範圍上限（不是安全依據） |
| 門速 | \|q̇\| > 0.05 rad/s | 原 K2Config.door_abort_rate |
| 非設計接觸 | 窗口最大值 > 0.1 N，不要求持續 | 原 contact_abort_n；執行端窗口為逐物理步最大值，每個窗口一收到就判，短脈衝不漏 |
| 牆接觸 | 窗口最大值 > 0.1 N | 同上 |
| 腕力 | 窗口最大值 ≥ 30 N | 原 wrist_abort_n |
| 接觸窗口無效 | n_invalid > 0、欄位缺失、非有限 | — |
| 狀態無效 | 長度不是 29、非有限、步號或時間倒退（重複只略過） | — |
| 停更 | 狀態或接觸最後有效訊息的牆鐘年齡 > 0.2 s（arm 後從未收到就從 arm 起算） | 原 stale_s |

- **觸發**：發 /door/lw_guard（原因、來源步、來源模擬時間、接收牆鐘、觸發牆鐘）與 /room/stop，並寫 `lw_observer.json`。只有首次觸發算數，之後鎖定。
  - 記為觀察提前終止，**不是**控制器成功停住。
- **正常結束**：他方的 /room/stop（任務節點到期 finish）⇒ 觀察器 close()，之後停更不算觸發。
- 訂閱深度 100（不以 depth 1 丟窗口）；每 20 ms 檢查新鮮度。

**既有紀錄對照**（供判斷保護是否會在正常停止中觸發）：
- 3 趟 EV 的任務節點狀態年齡最大 0.023–0.088 s，接觸年齡最大 0.023–0.088 s，都 < 0.2 s。
- 7 趟中止後 |q̇_door| 最大 0.046 rad/s，接近 0.05 界。若長窗中門速超過，觀察器會觸發提前終止。照實報告，不調界。

## 4. 測試
`test_door_k3ab_lw.py` 13 項，涵蓋：
- 短脈衝接觸觸發與首次鎖定；牆、腕、門角、門速、窗口無效；
- 狀態停更（含重複步不更新年齡）、接觸停更、arm 後從未收到；
- 非有限、倒退、欄位缺失；未 arm 不觸發；正常結束後不觸發；
- 缺步／重複步／非有限值都使停住無法確認；低速後再動被標出；
- 觀察窗不足的措辭；DONE 與其他無閂鎖分支；保護提前終止記為未驗證；
- 首次低速與確認分列；殼—橫桿 < 5 mm 區段。

煙霧核對：以既有 door_k3ab_ev_1 跑分析器，結果為 `not_stopped_within_covered_3.06_s`、`window_short`；殼—橫桿 < 5 mm 區段自閂鎖後 0.99 s 起，持續到實錄末（取樣最小 4.36 mm），期間非設計與牆接觸皆 0。

## 5. 不宣稱
- 單趟介入觀察；不做根因主張。
- 不是 K3-M 停止的證據，也不解除 K3-M 功能趟的進入條件。
- 5 mm 門檻保留：閂鎖後低於它的區段只是失敗狀態的只讀觀察，不恢復控制、不改判通過。
