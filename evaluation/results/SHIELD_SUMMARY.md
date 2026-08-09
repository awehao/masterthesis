# Instantaneous-speed gate ablation (seed27, ten replays per arm)

`dyn_obs_5` is configured at 0.10 m/s and `min_track_speed` was 0.10, so noise decided each cycle whether it was a mover. Everything else is stock: no Mahalanobis gate, no fragment merge, no coasting.

| arm | runs | reached | track exists | **published** | est. speed | true speed | movers/cycle | contacts | worst |
|---|---|---|---|---|---|---|---|---|---|
| U0 baseline (gate 0.10, no shield) | 9 | 9/9 | 100% | **23%** | 0.049 | 0.097 | 0.5 | **9/9** | -0.300 |
| U1 gate 0.05, no shield | 10 | 10/10 | 98% | **32%** | 0.040 | 0.100 | 1.9 | **7/10** | -0.274 |
| U2 gate 0.05 + raw-scan shield | 10 | 10/10 | 100% | **30%** | 0.050 | 0.100 | 1.4 | **0/10** | +0.097 |

## Why the mover was withheld, by reason code

- **U0 baseline (gate 0.10, no shield)**: published 23%, instant speed 68%, net displacement 9%
- **U1 gate 0.05, no shield**: published 32%, instant speed 53%, net displacement 15%
- **U2 gate 0.05 + raw-scan shield**: published 31%, instant speed 49%, net displacement 20%

**Reading it**

- `published` is the causal metric. If it does not rise, nothing else matters.
- `instant speed` should vanish from the reason breakdown; if the rejections
  simply move to `net displacement`, the gate was not the binding constraint
  and this arm proves nothing.
- `movers/cycle` guards the other side: admitting statics would show up as more
  tracks called movers, and would cost corridor width rather than buy safety.
- Contacts come last, and at ten runs they cannot carry the claim on their own.


## Shield behaviour (U2)

- cycles 41392, intervened 2317 (5.6%)
- |dv| when active: median 0.183, p95 0.290 m/s
- clearance when active: median 0.144, min 0.100 m
- **max violation after: +0.0003** (> 1e-3 in 0 cycles)
- fallback engaged 0, unresolved 0
- iterations median 1, max 6
- scan age p99 0.100 s, stale cycles 0
- speed cost: median 0.176 -> 0.160 m/s (9.1%)
