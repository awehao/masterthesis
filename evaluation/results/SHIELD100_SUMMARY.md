# Shield over 100 routes

| n | arrived | contacts | rate | median clr | worst | arrival | path | jerk_vx |
|---|---|---|---|---|---|---|---|---|
| 96 | 96/96 | **1** | 1.0% | +0.188 | -0.029 | 104 s | 22.0 m | 0.545 |

**1 contacts in 96 trials = 1.0% (95% CI 0.0%–3.1%).**

- static contacts 0, dynamic 1 (of 96 bags scored)
  - seed70: static +0.100, dynamic -0.029 (DYNAMIC)

## Shield behaviour

- cycles 236689, intervened 4195 (1.77%)
- |dv| when active: median 0.030 m/s
- clearance when active: median 0.106, min -1.000 m
- fallback engaged 554 (0.23%), max violation after +0.1598
- speed cost 0.213 -> 0.213 m/s (0.1%)
- stale cycles 27, scan age p99 0.100 s

`fallback` is the full barrier being infeasible, at which point the layer solves the weaker non-approach condition instead. That degradation is by design, but the diagnostic does not yet record whether the non-approach condition itself held -- a `non_approach_ok` field is still to be added.
