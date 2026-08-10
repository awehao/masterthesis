# Shield fallback fix: does it cost distance?

Paired on 18 routes. Old relaxation used np.minimum, which made distant returns forbid any approach; the fix uses np.maximum so only the violated rows relax.

| | 舊 fallback | 修正後 |
|---|---|---|
| 路徑中位 [m] | 22.1 | 20.6 |
| 到達時間中位 [s] | 100 | 99 |
| 間距中位 [m] | +0.196 | +0.191 |
| 到達 | 18/18 | 18/18 |
| 碰撞 | 0 | 0 |

## 路徑逐條變化

- 中位變化 **-0.42 m**，範圍 -8.9 … +3.6 m
- 變長超過 0.5 m 的路線：**4/18**
  - seed3: 24.7 → 28.3 m  (+3.6)
  - seed7: 20.0 → 23.6 m  (+3.6)
  - seed16: 22.7 → 25.5 m  (+2.8)
  - seed6: 21.7 → 23.7 m  (+2.0)
  - seed4: 16.2 → 16.6 m  (+0.4)
- Wilcoxon p = 0.2462

## Shield 行為

- **舊 fallback**: 週期 48,995，介入 1.15%，fallback 222 (0.45%)，**輸出零但輸入非零 116**，殘差 after 最大 +0.0459
- **修正後**: 週期 142,831，介入 1.05%，fallback 51 (0.04%)，**輸出零但輸入非零 36**，殘差 after 最大 +0.0183

**判準**：路徑中位變化若超過 +2 m，或變長的路線超過半數，這個修正就要改法 —
安全不能用普遍繞遠買。fallback 與「輸出零」的次數應該降到接近零，那是修正
本來要達成的目標。

