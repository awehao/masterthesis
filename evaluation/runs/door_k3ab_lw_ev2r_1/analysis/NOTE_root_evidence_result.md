# door_k3ab_lw_ev2r_1：根與門扇三點讀取取證結果（單趟；2026-10-24）

**封存（Codex 20261024_002150）**：
> 單趟根／門扇執行邊界取證完成；任務未通過，閂鎖後覆蓋 10.11 s 仍未確認停止。零命令寫入後回讀為零，但物理步後根速度已改變；相鄰步 D、A 讀值一致。門扇角速度積分與姿態增量的差異，在 handle 有效的讀取條件下仍存在。未確認物理成因，不構成停止安全驗收。

診斷到此收斂。門速保護、5 mm 門檻、K3-M 進入條件不變。

**核可與執行**
- 依 Codex 20261023_235904 條件式核可。凍結 `freeze_door_k3ab_lw_ev2r_dev_r2`（166 檔）在起跑前與結束後都核對通過。
- run_guarded_v2：CPU 峰值 82 °C，熱中止 0 次。
- 控制、保護、物理設定都沒有改動。只跑一次，不補跑。

## 1. 任務判定（door_k3ab_check.py）
**未通過**。finger_contact_lost：策略中止 21.80 s、執行端閂鎖 21.83 s（步 2181）。判定理由與 lw_1 相同，含 base_drift。任務 rc=1，照實保留。

## 2. 觀察器保護完整性
**完整**。armed、未觸發；closing 後收 TERM 而 closed。LW_HEALTH：sim exited／0、obs exited／0。

## 3. 實際觀察長度與停止分析（door_k3ab_lw_analysis）
- 閂鎖後覆蓋 10.11 s，**未確認停止**（`not_stopped_within_covered_10.11_s`）。
- 窗末：門 3.12°、底盤位移 28.4 mm、偏航 −0.86°。
- 殼—橫桿取樣最小 4.43 mm；閂鎖後約 1.11 s 起至窗末持續 < 5 mm。手指—橫桿 −0.39 mm、自碰 9.85 mm。
- 與 lw_1 的停止行為相近。這是第二次觀察，兩趟不構成統計。

## 4. 取證完整性
**evidence_complete＝True**。依獨立核對：
- 列數 3193／room 3193；缺步、多步、重複、順序、時間皆 0；
- 欄位長度、型別、有限性無誤；
- 步進後實際時間 `D_time` 與 t＋dt 全部一致；
- 寫檔器 complete；
- 快照與模組來源皆有實質內容。

取證耗時中位數 0.34 ms／步（最大 1.8 ms）。任務節點牆鐘 28.3 s，lw_1 為 27.7 s。

## 5. 三點讀取（door_k3ab_root_evidence_analysis；只描述，不做根因主張）
**handle**：六個讀取點全部 3193／3193 為 True ⇒ 依有效性判定的預期分支是 PhysX tensor 路徑（不是分支的直接證據）。

**執行期模組來源**：
- RigidPrim、SingleArticulation、Articulation 來自 `extsDeprecated/isaacsim.core.prims`；World 來自 `extsDeprecated/isaacsim.core.api`。
- 門扇 physics view 類別來自 `omni.physics.tensors` 110.1.13。
- 套件 isaacsim、isaacsim-core 皆為 6.0.1.0。sha256 見 physics_snapshot.json。

**物理設定快照**（三分類）：
- **USD 宣告值**：
  - PhysicsScene：solverType TGS、timeStepsPerSecond 100、enableCCD True、enableStabilization False、broadphase MBP、GPU dynamics False；
  - 門扇：mass 8.0、physxRigidBody:angularDamping 2.0、linearDamping 0.0；
  - 鉸鏈：drive 剛度與阻尼 0、jointFriction 0、限位 0–110°；
  - articulation root 剛體：mass 0.001、diagonalInertia 1e-6（根連桿是極輕的連桿）。
- **執行期回讀值**：physics_dt 0.01、TGS、CCD True、穩定化 False、MBP、裝置 cpu、自碰 0、啟動時兩個 handle 都有效。
- **未知**：機器人求解迭代數（API 回傳 None）、門鉸鏈狀態（不在 articulation 內，未新增讀取）。

**根速度**：
| 比較 | 結果（max-abs，線 m/s 與角 rad/s 混合取最大） |
|---|---|
| api_args 對 cmd_body 經 yaw 轉換 | 中位數 4.4e-11、最大 4.5e-10（寫入參數＝命令的世界座標轉換） |
| B 回讀對 api_args | **全部 0**（寫入值原樣成為根速度） |
| D（步進後）對 B | 中位數 0.0015、p95 0.0107、最大 0.0239（物理步改變了根速度） |
| D_k 對 A_{k+1} | 位姿與速度**全部 0**（兩次讀取之間沒有其他寫入） |

**閂鎖後零命令段**（1012 步，cmd_body 全為 0）：
- 步進後根線速度：中位數 1.20 mm/s、p95 2.39、最大 3.00。
- 相鄰 D 位姿差分速度：中位數 2.69 mm/s、p95 4.17。
- 兩者比值中位數 0.46。**根速度參考點（根連桿原點或質心）未釐清，比值偏離 1 不稱物理失配。**
- 由此只能描述：寫入的零速度在每個物理步後不再是零，且根的位置在零命令下持續改變。

**門扇**：
- 執行端紀錄門角與門速（C_recorded）與同點 tensor 讀值**完全相同**（差 0）⇒ 門速欄位就是 tensor view 的 −ω_z。
- ω_x、ω_y 最大 8.7e-10，近乎 0。
- 步進後角速度 −ω_z(D) 的右端積分 ÷ D 姿態門角增量：OPEN→中止 2.21、閂鎖後 0–1 s 1.32、1 s–窗末 4.40。
  - 在 tensor 分支（handle 有效）與步進後讀取下，速度與姿態的不一致仍存在，數值與 lw_1、ev_1 的 r2 結果同量級。
  - 原因未查；不據此判定門速保護保守或失效。

## 6. 結論（收斂）
- 取證完整：寫入、回讀、讀取時序與分支條件都已核對到紀錄層。
- 零命令下，物理步後根速度已改變（D 對 B）。程式查核未見其他外部命令寫入，相鄰 D、A 邊界也未觀察到狀態差異（D_k＝A_{k+1}）；不據此排除物理步內部的作用。
- 門速欄位與門角增量的不一致，在 PhysX tensor 讀取下仍存在。
- 依 Codex 方向，診斷支線到此收斂，回到協同開門主線。K3-M 停止進入條件未解除；5 mm 門檻與門速保護不變。

影片：該趟實錄重播待重播修正核可後產出（`第七次進度報告素材/影片/README.md`）。
