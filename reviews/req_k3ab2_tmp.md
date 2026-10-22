# K3-A-B 小規格 draft-2（依你 20261022_133343 的三項）

全文：`evaluation/results/specs/K3_finger_balance_compliance_spec.md`（開頭附 draft-2 說明）。

1. **開門首筆連續**（§4）：OPEN 第一輪 Δ≡0、濾波不更新；DoorTargetK3 首輪不前進，所以首筆 OPEN 目標＝最後 HOLD 目標（≤1e-9）。第二輪起 Δ 才更新，每輪 ≤ v_max·Δt。
2. **END_HOLD 不動**（§4、§5）：Δ 在 OPEN 第二輪到 SETTLE 期間作用，進入 END_HOLD 那輪起凍結；END_HOLD 與 DONE 固定發布最後一筆已發布目標。策略分存 last_base 與 last_target。
3. **判定器**（§7）：另建 door_k3ab_check.py。
   - 基礎目標套原參考檢查（含純鉸鏈旋轉）；
   - 發布＝基礎＋Δ·e12_world（旋轉相等、平移差相符、e12_world＝R_base·(0,−1,0)）；
   - Δ 的上限與速率界限、首筆前 Δ＝0；
   - END_HOLD／DONE 固定；
   - 證據缺漏即不通過。
   - 每項都有完整判定流程的反例測試。

另一個待你裁定的點：SETTLE 期間 Δ 仍作用（SETTLE 的 q_ref 保持在 θ_goal，目標仍屬追蹤段）。若你認為 SETTLE 也應凍結，請指示。

請裁定可否進入實作（實作後送實作審查，開跑前再確認）。
