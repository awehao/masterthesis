# N1 情境第二版（r2，2026-11-28）

Howard 11/28 00:40 指示：N1 正式第一輪（n1main）中原版 GMPC 沒到達的 15 條路線降低難度，4 方法都只重跑這 15 條（批次 n1r2，60 趟）；N1 最終結果＝其餘 45 條沿用 n1main＋這 15 條用 n1r2（Howard 00:45）。

- 選這 15 條的依據：**第一輪原版 GMPC 未到達**（bigarena k4,5,9,10,12,19；office k7,11,12,17,20；warehouse k6,11,15,17）。
- 改法（15 條同一組參數）：`--n-on-path 2 --n-corner 1 --n-near-path 1 --n-cross 2 --n-counter 1`（原 4／2／2／4／2）；起終點、route_try、名目路線不變；行人總數仍 10（多出的變背景行人）。
- 其餘 45 條情境不變。第一輪情境完整保存在 `../n1_scenarios_r1/`，第一輪結果在 `results/n1main/`，不刪。
