# r3_s8_off_180328：啟動流程失敗，**不是** OFF 導航結果

中止點：就緒檢查 `!! lifecycle /planner_server active 逾時 60s`，`exit=2`。

Nav2 的生命週期管理員在啟動階段失敗：

```
[lifecycle_manager_navigation]: Activating planner_server
[lifecycle_manager_navigation]: Failed to change state for node: planner_server
[lifecycle_manager_navigation]: Failed to bring up all requested nodes. Aborting bringup.
```

從 `Configuring planner_server` 到失敗僅約 1.7 s，是啟動期的暫時性失敗。
其餘檢查正常：靜止檢查通過（訂閱配對 2.5 s，11 個主題皆靜止，
`/odom` twist 全零），bag 有資料（`/clock` 11052、`/odom` 3714、
`/scan` 1123、`/model/omni_bot/pose` 2210、`/cmd_vel` 2244，時長 121.3 s）。

**未進入任務，沒有發目標**，因此不列為 seed 8 r3 的 OFF 結果。
依 `FROZEN_v2_config.md` 的處置規則（只有啟動或資料無效才另行處理），
以**相同版本與設定**、另一個獨立 domain 重跑；不調參、不重建。
