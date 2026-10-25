# door_wide_k3b_30_1：30° 開門＋受控暫停＋K3-B 受控釋放（首趟；**中止，不通過**）

- 核可：Codex 20261026_012939 條件式核可單趟。
- 凍結：`freeze_door_wide_k3b_dev_r3`（376 檔）。
  - 起跑前：跑法自動核對與手動核對，均一致。
  - 跑後：手動核對一致，r1、r2 仍一致。
- 跑法：`run_door_wide_k3b_r3.sh`。
  - 跑法開頭的說明句「不張爪、不退離」是沿用基底腳本的舊文字。
  - 本趟任務節點 k3b3 帶 `--w2-release`，實際有張爪。
- 環境：
  - 乾淨環境（/usr/bin/python3、無 VIRTUAL_ENV）、ROS_DOMAIN_ID 97；
  - 92 °C 守護：CPU 峰值 83 °C，熱中止 0 次。
- 機器：hostname msi、AMD Ryzen AI 9 465、RTX 5060 Laptop（驅動 595.91.07）、kernel 7.0.0-31；溫度來源 hwmon:k10temp。
- 單趟，失敗保留，不補跑、不新增影片。

## 結論
- 開門 30°、SETTLE、PAUSE、HELD、mode 1、mode 2 提交都正常。
- 張爪後門扇**自行繼續轉動**：
  - 張爪首步起約 1.05°/s，0.16 s 內 +0.17°；
  - 3 s 後仍約 0.2°/s，紀錄結束時門角 30.95°，仍在轉。
- 任務端一般幾何監看的非手指淨空**預測值**跌破 5 mm：
  - 實測 5.61 mm，預測 3.55 mm（T_pred 0.2 s、含門速）；
  - 在 RELEASE 0.08 s 後以 `k3b_geometry:geometry_clearance_nonfinger` 中止。
- 未請求 mode 3、未退離、未請求 5 N，三維收斂與保持未進入。
- 判定器 `door_k3m_check_k3b3` 不通過，結果在 `analysis/door_k3m_check_k3b3.json`。

## 時序（同一模擬時鐘，s）
| 事件 | 時刻 |
|---|---|
| OPEN | 20.82 |
| SETTLE | 52.66（門角 29.89°） |
| PAUSE | 53.98（門角 29.99°）；執行端／求解端切換 54.03（步 5401） |
| 求解端 ZERO_PUB／STOPPED | 56.31／56.41 |
| 求解端 OBS_PASSED＝HELD | 66.47 |
| mode 1（RELEASE_PREP_TEST） | 請求 67.54；執行端登記 67.59；提交 67.67 |
| mode 2（RELEASE_OPEN） | 請求 70.80；登記 70.83；提交 70.91（步 7089）＝張爪命令起點 t_open |
| 求解端暫停監督器 FAIL `moved_in_held` | 70.96 |
| task 進 RELEASE（基準步 7084） | 70.94 |
| task 中止 `k3b_geometry:geometry_clearance_nonfinger` | 71.02 |
| 執行端看到閂鎖（步 7105）：W2 中止、手指閘門凍結、5 N 權威中止 | 71.07 |
| 執行端解除監看 B_CONFIRMED | 71.71（門角 30.66°） |
| 執行端解除監看 FAILED `pair_force_after_confirm` | 72.70（步 7268） |
| 執行端 K3-M 監看違規 `corridor_yaw` | 73.40（步 7338） |
| room 紀錄結束 | 74.12（步 7410） |

## 分項
**張爪與釋放（物理上達成）**
- 手指開度：
  - mode 2 提交步起由 0.0016 m 張開，約 0.6 s 達 0.0089 m；
  - 中止後維持張開到紀錄結束。
- 兩指—橫桿接觸力：
  - 提交步 12.98／10.07 N，下一步（7090）即為 0；
  - 之後只在步 7268 出現一筆 finger1 0.43 N，見下。
- 執行端解除確認在 71.71 達成（開度 ≥ 95%、力 < 0.1 N 持續 0.5 s）。
  - 這發生在任務端中止之後；mode 3 准入早已因中止而關閉。

**中止原因：門扇在張爪後持續轉動**
- 門角變化：
  - 暫停期約 15 s 穩定在 30.232–30.234°；
  - 張爪後逐步增加：7090 為 30.243°、7105 為 30.402°、7268 約 30.79°、7410 為 30.955°；
  - 最後門速 0.0036 rad/s。
- 門模型：
  - 鉸鏈驅動 stiffness、damping、maxForce 都是 0；
  - 只有關節摩擦與剛體角阻尼（`door_asset_v2`）。
  - 所以門轉動不是驅動造成的。
- 成因推測（**未驗證**）：
  - 張爪瞬間釋放了夾持預載的接觸衝量，門因阻尼低而緩慢減速；
  - 無接觸時 K2 0° 曾見的漂移只有 0.047°/s，本趟初速約大 20 倍。
- 一般幾何淨空（任務節點同一 GeometryMonitor）：
  - 非手指最小淨空，以 [實測, 預測] 列出：
    - 暫停期 [7.85, 7.44] mm；
    - 70.94 為 [7.16, 6.11] mm；
    - 71.02 為 [5.61, 3.55] mm ⇒ stop；
  - 手指淨空 33.9 mm。
- 中止後門繼續轉，手臂不動：
  - 步 7268 橫桿碰到已張開的 finger1（0.43 N，單步）；
  - 執行端解除監看因此判 `pair_force_after_confirm`。
- 非手指部位（含 uflite_gripper_link）全程未出現接觸力。
- r2 規格 §2(a) 已列此風險（「張爪釋放 12 N 預載時門若被推動，更易判失敗」）。本趟觸發的是一般淨空預測，比門角帶更早。

**中止路徑（K3-B 必修 1 在實跑中運作）**
- 手指閘門：
  - 閂鎖後第一次看到（步 7105）即把凍結值由 0.0（閉合）改為最後實際下達值 0.0089（張開）；
  - room `w2.grip_gate` 有記錄；
  - 手指沒有重新閉合。
- W2 權威：中止 `external:k2_latch:…`（步 7105）。
- 5 N 權威：中止 `w2_or_k2_abort`，未曾請求 5 N，`restore` 為 null。
  - 手指驅動上限停在 20 N：啟動讀回 5、切換讀回 20，定期讀回 19 次皆正常。
- 求解端：
  - 暫停監督器在 70.96 判 `moved_in_held`（門檻：底盤 1 mm／0.005 rad、門 0.002 rad、手臂 0.001）；
  - 原因是張爪後門與手臂負載改變。
  - 這一項不是本趟中止的觸發來源，但表示 K3-B 釋放階段仍沿用 HELD 監督，**張爪本身就會使它 FAIL**，需處置。

**未進入的項目**
- 近區退出（C_CLEAR）、mode 3、退離：未進入。
- 5 N 恢復：未請求。
- 三維收斂與保持：未進入。
- TaskReleaseMonitor 停在 ARMED，samples 0。中止發生在第一筆釋放樣本之前。

**保護與量測**
| 項目 | 數值 |
|---|---|
| 腕力峰 | 10.97 N（門檻 30） |
| 非預期接觸、牆接觸 | 0 N、0 N |
| 門速峰（開門段判準） | 0.0388 rad/s |
| 夾持漂移峰 | 1.02 mm／0.0034 rad |
| 中止後底盤 | 步 7080→7410：平移 7.2 mm、yaw +0.64°，執行端 K3-M 監看於 73.40 判 `corridor_yaw`（yaw 偏差 0.0140 rad） |
| CPU 峰值 | 83 °C |

**紀錄完整性**
- trace：927 筆，state_step 無重複、無倒退。
  - 最大牆鐘間隔 0.092 s、最大模擬間隔 0.12 s；
  - 迴圈 p99 0.040 s、最大 0.042 s，> 0.1 s 0 次。
- 一般幾何紀錄 599 列，無無效列、重算不一致 0。
- room：逐物理步紀錄到步 7410。
- 各節點正常結束（task rc 1＝中止、其餘 0）。

## 判定器理由（`door_k3m_check_k3b3`，pass False）
| 理由 | 來源 |
|---|---|
| `aborted:k3b_geometry:geometry_clearance_nonfinger`、`termination:ABORTED`、相序止於 RELEASE | 本趟中止，屬實 |
| `M1_commits_not_1_2_3_confirmed` | 只提交 mode 1、2，屬實 |
| `k3m_executor_violation` | 中止後底盤 yaw 漂移的 `corridor_yaw`（73.40），屬實 |
| `finger_bar_outside_allowed_phase` | 中止後門轉碰 finger1 0.43 N（門檻 0.1），屬實 |
| `grasp_finger_contact_below_min` | 舊鏈把 RELEASE／ABORTED 列計入夾持段；張爪後接觸 0。舊判準不認得 RELEASE，屬連帶 |
| `door_angle_pre_open_over_1deg` | 舊鏈統計值 30.95°，是中止後的列被計入開門前窗所致，屬連帶 |
| `record_does_not_cover_task_window`、`trace_unverifiable` | 舊鏈要求 DONE 時刻；本趟無 DONE，屬連帶 |

「屬連帶」是依舊判準程式邏輯（door_k3a_check 的 GRASP／DONE 窗定義）所作的歸因，判定器本身沒有放寬。

## 待處置（送 Codex）
1. 張爪後門扇不靜止：約 1°/s 初速、緩慢減速。現行設計假設張爪時門近乎靜止，與實測不符。
2. 一般淨空預測與門角帶都會被門的自轉觸發；中止後手臂不動，門會轉進張開的手指。
3. 求解端 HELD 監督在 mode 2 張爪時仍以門 0.002 rad 判 `moved_in_held`。
4. 中止後底盤 yaw 漂移 0.64° 觸發 K3-M 走廊。
