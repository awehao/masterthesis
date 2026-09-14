# 啟動失敗，非任務結果（逐連桿 TF 少兩個）

修好時鐘後，TF 新鮮度正常，但 **9/11**：
`uflite_finger1`、`uflite_finger2` 查不到。

原因：Isaac 執行端的 `/joint_states` 只發布 joint1–6，
`robot_state_publisher` 因此算不出兩個手指連桿的 TF。
不是場景問題，也不是 TF 樹接錯。

處置：執行端一併發布 `finger_joint1/2` 的實測狀態
（本測試不動夾爪，但它們是模型的真實自由度）。
**不降低「每個連桿 TF 都要合格」的條件。**

其餘各項本趟都通過：場景語意檢查 0 個外部碰撞體、
`obstacles_configured = 0`、下游速度框讀回與求解端一致。
本目錄不計為任務結果。
