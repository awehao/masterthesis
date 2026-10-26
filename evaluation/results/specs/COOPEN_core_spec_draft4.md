# 連續協同通行：核心段接觸驗證 — 規格 draft-4（接近進度閘控，g 版）

draft-1～3 保留。本版依 Codex 20261027_003024：door_coopen_core_1 封存為「接近段中止，核心段未進入」；下一包只改接近參考的推進管理（控制策略變更，不是已證明安全的接近能力）。範圍、量測、夾持證據、安全層現況同 draft-3；本文件只列差異。

## 1. door_coopen_core_1 封存與診斷（只讀）
- **結果**：APPROACH 14.98 s finger_bar_outside_allowed_phase（finger2 0.94 N）；chain_pass False、core_evidence insufficient、登錄核對 OK。
- **觀察（Codex 接受）**：
  - 接近目標沿登錄軸推進（垂直分量 0）；
  - 實測 TCP 側向偏離隨接近增大到 8.8 mm，正面夾法兩趟終點 1.4 mm；
  - finger2 提前碰桿，既有保護中止；
  - 未取得開門與出口進展證據。
- **配對模型核對**（`door_coopen_core_1_model_check.py`，同一物理區間，取證 arm_sp 與 room q_arm_meas 逐步配對，一步預測 x⁺＝x＋α(s−x)＋b）：

| 趟次 | j2 等效穩態偏移（只模型） | 加求解器 d̂ 後 | 一步殘差標準差 |
|---|---|---|---|
| core_1 | −0.027 rad | −0.0014 rad | 1e-4 rad |
| w1f_60_3 | +0.015 rad | −0.0019 rad | 2.4e-4 rad |
| w2c_30_1 | +0.015 rad | −0.0019 rad | 2.4e-4 rad |

  - 關節層「模型＋d̂」的一步預測在三趟一致。
  - 先前「後仰構型下設定點模型偏差」的推論不受支持，撤回。
  - TCP 側向偏離的來源未確定，候選是求解器追蹤取捨或預測與實際追蹤的不一致。
  - 依 Codex 指示不擬合構型偏差模型，也不據此宣稱機制。

## 2. 接近進度閘控（`door_coopen_approach_gate.py`，mixin 疊在 K3MP2Policy 上，不改父類）
- **G-A 對準確認**：ALIGN_HOLD → APPROACH 須「連續有效量測、TCP 對預抓取目標誤差 < pre_tol_m（5 mm）」累積 0.5 s（狀態時間）才放行。原版是單筆就放行。
- **G-B 推進凍結**：APPROACH 中，以本輪發布目標 T(s) 計 TCP 誤差，分解為沿接近軸（世界 H_R·a_H）、垂直分量與姿態角。
  - 垂直 > 3 mm ⇒ 暫停：本輪父類推進後把 s 還原，目標停在 T(s)。**控制照常發布、W-GMPC 繼續追蹤當前目標**，不凍結手臂設定點、不停止發布、不新增命令來源。
  - 暫停中，「垂直 ≤ 2 mm 且 |沿軸| ≤ 5 mm 且 姿態 ≤ 0.02 rad」以有效量測連續 0.5 s 才續進。
- **有效量測**：T_tcp／H 有限、state_age ≤ stale_s（0.2 s）、且狀態時間嚴格前進（重讀同一狀態不累積）；無效即清空累積窗。
- **APPROACH 總逾時 15 s**（K3AConfig）自進入 APPROACH 起算，暫停期間照計，不重置、不放寬；不收斂就失敗。
- **既有中止優先**：提前碰桿、非設計接觸、腕力、幾何、資料失效在父類 monitor 先行，閘控不遮蔽。

**登錄（`coopen_core_params_48_r6.json`＝r5＋approach_gate）**：

| 鍵 | 值 | 依據 |
|---|---|---|
| perp_pause_m | 0.003 | Codex 單一候選 |
| perp_resume_m | 0.002 | Codex 單一候選 |
| resume_hold_s | 0.5 | Codex 單一候選 |
| along_resume_m | 0.005 | K2Config.pre_tol_m |
| rot_resume_rad | 0.02 | 求解器 reach_rot_rad |
| align_confirm_s | 0.5 | K2Config.grip_settle_s |

## 3. 版本化接線
| 檔案 | 內容 |
|---|---|
| `door_coopen_core_task_node2.py`（v1 複本，只改五處） | 匯入閘控；策略換成 gated(K3MP2Policy)，閘控設定取自登錄（鍵不符拒跑）；task.json 記 coopen_gate（設定、模組 sha256）；trace 每列記 gate；approach_gate_stats |
| `run_door_coopen_core_g.sh`（原跑法複本） | 只換登錄 r6、任務節點 v2、凍結 g r1、判定器字樣；不新增節點或命令來源 |
| `door_coopen_check_core3.py`（r2 擴充） | J4 閘控登錄核對（chain_pass 條件）；core 另報 approach_gate_stats 與 pause／resume 事件 |

## 4. 測試
| 測試 | 內容 | 結果 |
|---|---|---|
| `test_door_coopen_approach_gate.py` | G1 偏離凍結 s 後恢復續進；G2 不收斂 ⇒ timeout:APPROACH；G3 重讀同一狀態不累積恢復窗；G4 暫停中碰桿仍立即中止；G5 對準確認連續 0.5 s，誤差閃爍會延後；G6 暫停期每輪仍發布目標＝T(s 凍結值)、solver_start；G7 無擾動照常 DONE；G8 未疊閘控的策略不受影響 | 8／8 |
| `test_door_coopen_core_g_nodes.py` | W1 節點 v2 以 r6 啟動並記閘控設定與模組雜湊；W2 r5（無閘控）拒跑；W3 判定器 J4 一致／不符／缺；W4 跑法只差四行 | 4／4 |

- 既有：核心段 policy 5、main ROS 3、判定器 r1 3、判定器 r2 4，以及 K2／K3A／w1b／w1e2 回歸。
- **揭露**：合成世界以「量測注入偏移」模擬追蹤偏差，不是真鏈端到端聯測。暫停期間「控制照常發布」在策略層（G6）驗證；節點層沿用 v1 迴圈「每輪發布非 None 目標」的既有路徑，未另做真鏈驗證。

## 5. 凍結
`freeze_door_coopen_core_g_dev_r1.sha256`＝core r3 清單＋本包新檔＋稽核掛勾實測執行時開啟檔（最後建立）。

## 6. 申請
單趟 `door_coopen_core_g_1`：跑法 `run_door_coopen_core_g.sh`、判定器 `door_coopen_check_core3.py`。執行條件同 draft-3 §8。
