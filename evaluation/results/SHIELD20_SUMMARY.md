# Raw-scan shield on the full scenario

20 routes. Control is poseF: same routes, same pose source, same soft slack, same min_track_speed. The shield is the only difference.

| arm | n | arrived | contacts | static | dynamic | median clr | worst | arrival | path |
|---|---|---|---|---|---|---|---|---|---|
| control (no shield) | 19 | 19/19 | **3** | 0 | 3 | +0.161 | -0.018 | 103 s | 22.1 m |
| **shield** | 20 | 20/20 | **0** | 0 | 0 | +0.196 | +0.047 | 105 s | 22.5 m |

## Paired on the same 19 routes

- contacts: control 3, shield 0
- fixed by the shield: 3   introduced by it: 0
- McNemar p = 0.2500

## Shield behaviour

- cycles 48995, intervened 561 (1.1%)
- |dv| when active: median 0.034 m/s
- clearance when active: median 0.099, min 0.048 m
- **max violation after +0.0459**, unresolved 219 cycles, fallback 222
- speed cost: 0.214 -> 0.214 m/s (0.1%)
- stale cycles 0, scan age p99 0.100 s

**Scope**: 20 routes cannot settle a contact RATE -- the trial-to-trial spread
is 0.043 m median and 0.479 m worst on identical configurations, so only a
large effect is resolvable here. It answers whether the shield transfers off
seed27, not what the residual rate is.

