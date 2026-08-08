# Overnight results

Scenario: bigarena, 40 random routes, GMPC+CBF, mask 10 deg, fixed margins 0.60/0.38, hardware motion limits with wheel-speed coupling.

| arm | n | arrived | contacts | median clr | worst | arrival | jerk_vx | p95 solve |
|---|---|---|---|---|---|---|---|---|
| A/B: soft slack (old movers) | 4 | 4 | **0** | +0.075 | +0.000 | 108 s | 0.539 | 5.1 ms |
| A/B: hard static k0 (old movers) | 4 | 4 | **2** | +0.020 | -0.050 | 107 s | 0.612 | 6.0 ms |
| hard static k0 + mover cap 0.14 | 38 | 38 | **5** | +0.199 | -0.130 | 99 s | 0.520 | 2.1 ms |
| soft slack + mover cap 0.14 | 38 | 37 | **6** | +0.180 | -0.181 | 101 s | 0.537 | 2.5 ms |
| replicate of the cleaner arm | 39 | 39 | **4** | +0.213 | -0.129 | 101 s | 0.521 | 2.2 ms |
| soft slack + pose from EKF topic (no TF composition) | - | - | (not run) | | | | | |

## Barrier audit (per control cycle)

`A z - l < 0` means the command itself broke the CBF inequality; h < 0 alone does not, since a robot already inside the keep-out but leaving it still satisfies the condition.

- **A/B: soft slack (old movers)**: cycles 7626, wall points dropped before solve 0, barrier broken without slack 2861 (37.52%), of which slack kept feasible 852, worst residual -6.4185
- **A/B: hard static k0 (old movers)**: cycles 8131, wall points dropped before solve 0, barrier broken without slack 3180 (39.11%), of which slack kept feasible 585, worst residual -5.4901
- **hard static k0 + mover cap 0.14**: cycles 75673, wall points dropped before solve 0, barrier broken without slack 26394 (34.88%), of which slack kept feasible 5655, worst residual -12.5982
- **soft slack + mover cap 0.14**: cycles 79286, wall points dropped before solve 0, barrier broken without slack 27903 (35.19%), of which slack kept feasible 8355, worst residual -22.3783
- **replicate of the cleaner arm**: cycles 83124, wall points dropped before solve 0, barrier broken without slack 29720 (35.75%), of which slack kept feasible 6435, worst residual -20.6391

## Paired on the same 36 routes

- hard static k0: 4 contacts   soft: 5 contacts
- only hard contacts: 3   only soft contacts: 4
- McNemar p = 1.0000

## Replicate (hardC run twice, same 37 routes)

- contacts: 5 vs 4
- same-route clearance spread: median 0.043 m, max 0.479 m

This is the noise floor: any config difference smaller than it cannot be resolved at n=40.
