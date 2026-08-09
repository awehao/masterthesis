# Instantaneous-speed gate ablation (seed27, ten replays per arm)

`dyn_obs_5` is configured at 0.10 m/s and `min_track_speed` was 0.10, so noise decided each cycle whether it was a mover. Everything else is stock: no Mahalanobis gate, no fragment merge, no coasting.

| arm | runs | reached | track exists | **published** | est. speed | true speed | movers/cycle | contacts | worst |
|---|---|---|---|---|---|---|---|---|---|
| min_track_speed 0.10 (control) | 9 | 9/9 | 100% | **23%** | 0.049 | 0.097 | 0.5 | **9/9** | -0.300 |
| min_track_speed 0.05 | 10 | 10/10 | 98% | **32%** | 0.040 | 0.100 | 1.9 | **7/10** | -0.274 |

## Why the mover was withheld, by reason code

- **min_track_speed 0.10 (control)**: published 23%, instant speed 68%, net displacement 9%
- **min_track_speed 0.05**: published 32%, instant speed 53%, net displacement 15%

**Reading it**

- `published` is the causal metric. If it does not rise, nothing else matters.
- `instant speed` should vanish from the reason breakdown; if the rejections
  simply move to `net displacement`, the gate was not the binding constraint
  and this arm proves nothing.
- `movers/cycle` guards the other side: admitting statics would show up as more
  tracks called movers, and would cost corridor width rather than buy safety.
- Contacts come last, and at ten runs they cannot carry the claim on their own.

