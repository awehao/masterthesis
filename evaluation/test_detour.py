"""Checks the committed detour before it goes near a benchmark.

The first gz round showed the mechanism works -- reversals 7.7 -> 3.6 per run,
matching MPPI's 3.2 -- but collided into walls four times in ten trials, because
the side choice picked "whichever side the obstacle is not on" with no check
that the side had room. These tests pin the fixes for that, and for two defects
that the scenario never exercised: in ten trials, 99% of the frames where the
detour was active had exactly ONE obstacle in the cone, so nothing multi-obstacle
was tested at all.

Run against the source tree (no build needed):
    python3 evaluation/test_detour.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/src/ammr_wholebody_mpc')

from ammr_wholebody_mpc.detour import (            # noqa: E402
    DetourConfig, DetourState, apply_offset, FREE, LOCK_LEFT, LOCK_RIGHT)

NAME = {FREE: 'FREE', LOCK_LEFT: 'LEFT', LOCK_RIGHT: 'RIGHT'}
R0 = (0.0, 0.0, 0.0)                    # robot at the origin, heading +x
OFF = 0.35


def cfg(**kw):
    return DetourConfig(enable=True, max_offset=OFF, **kw)


def mover(x, y, vx=0.0, vy=0.0):
    return dict(x=x, y=y, radius=0.25, vx=vx, vy=vy)


def wall(x, y):
    return dict(x=x, y=y, radius=0.05, vx=0.0, vy=0.0,
                static=True, margin=0.33)


def check(label, got, want):
    ok = got == want
    print(f'  {"OK  " if ok else "FAIL"} {label}: {got} (want {want})')
    return ok


def main():
    fails = 0

    print('\n-- side choice, single obstacle -------------------------------')
    s = DetourState(cfg())
    fails += not check('obstacle on the left -> go right',
                       NAME[s.update(R0, [mover(1.2, +0.5)])[0]], 'RIGHT')
    s = DetourState(cfg())
    fails += not check('obstacle on the right -> go left',
                       NAME[s.update(R0, [mover(1.2, -0.5)])[0]], 'LEFT')

    print('\n-- walls ------------------------------------------------------')
    s = DetourState(cfg())
    fails += not check('wall on the preferred side -> take the other side',
                       NAME[s.update(R0, [mover(1.2, +0.5),
                                          wall(0.6, -0.5)])[0]], 'LEFT')
    s = DetourState(cfg())
    fails += not check('walls both sides -> do not commit',
                       NAME[s.update(R0, [mover(1.2, 0.0), wall(0.6, -0.5),
                                          wall(0.6, +0.5)])[0]], 'FREE')
    s = DetourState(cfg())
    fails += not check('a wall alone must not trigger a detour',
                       NAME[s.update(R0, [wall(1.0, 0.0)])[0]], 'FREE')
    s = DetourState(cfg())
    s.update(R0, [mover(1.2, +0.5)])
    fails += not check('wall appears on the locked side -> release',
                       NAME[s.update(R0, [mover(1.2, +0.5),
                                          wall(0.6, -0.5)])[0]], 'FREE')

    print('\n-- other movers (never exercised by the gz scenario) ----------')
    s = DetourState(cfg())
    fails += not check('second mover parked on the preferred side',
                       NAME[s.update(R0, [mover(1.2, +0.5),
                                          mover(0.5, -0.5)])[0]], 'LEFT')
    # Currently 1.5 m clear on the right, but closing at 0.6 m/s: within the
    # 1.5 s lookahead it is at -0.6 m, inside 0.35 + 0.38. A check on present
    # positions alone would wrongly commit right and swing into it.
    s = DetourState(cfg())
    late = mover(0.5, -1.5, vy=+0.6)
    fails += not check('second mover ARRIVING on the preferred side',
                       NAME[s.update(R0, [mover(1.2, +0.5), late])[0]], 'LEFT')
    s = DetourState(cfg())
    fails += not check('  ...and it is genuinely clear right now',
                       abs(late['y']) > OFF + s.cfg.side_clear_dyn, True)

    print('\n-- identity across rebuilt dicts ------------------------------')
    # The node rebuilds every obstacle dict on each callback and the wire format
    # carries no id, so identity has to come from position association.
    s = DetourState(cfg())
    side0, _ = s.update(R0, [mover(1.2, +0.5)])
    held = True
    for i in range(1, 12):                      # obstacle creeps closer
        obs = [mover(1.2 - 0.05 * i, +0.5)]     # fresh dicts every step
        sd, _ = s.update(R0, obs)
        held &= (sd == side0)
    fails += not check('lock survives 11 rebuilt-dict steps', held, True)
    sd, _ = s.update(R0, [mover(-0.5, +0.5)])   # now behind
    fails += not check('releases once the obstacle is behind',
                       NAME[sd], 'FREE')

    print('\n-- lock must not jump to a different obstacle -----------------')
    s = DetourState(cfg())
    first, _ = s.update(R0, [mover(1.0, +0.4)])
    stuck = True
    for i in range(20):
        # the one we locked against stays put; a second mover drifts across the
        # cone from the other side, which is exactly what could steal the lock
        obs = [mover(1.0, +0.4), mover(1.8, -0.8 + 0.06 * i)]
        sd, _ = s.update(R0, obs)
        stuck &= (sd == first or sd == FREE)
    fails += not check('a passing second mover cannot flip the commitment',
                       stuck, True)

    print('\n-- regressions ------------------------------------------------')
    s = DetourState(cfg())
    first, sw = None, 0
    for i in range(60):
        sd, _ = s.update(R0, [mover(1.2, 0.5 * (1 if (i // 7) % 2 == 0 else -1))])
        if first is None and sd != FREE:
            first = sd
        elif sd != FREE and sd != first:
            sw += 1
    fails += not check('ping-pong obstacle, no walls: side switches', sw, 0)

    s = DetourState(cfg())
    for k in range(6):
        _, off = s.update(R0, [mover(1.2, +0.5)])
    X = np.stack([np.eye(3) for _ in range(20)])
    Y = apply_offset(X, off, OFF)
    dy = Y[:, 1, 2]
    fails += not check('arch peaks mid-horizon', int(np.argmax(np.abs(dy))) in
                       range(7, 13), True)
    fails += not check('arch returns to the plan at the end',
                       abs(dy[-1]) < 1e-9, True)

    s = DetourState(DetourConfig(enable=False, max_offset=OFF))
    sd, off = s.update(R0, [mover(1.2, +0.5)])
    fails += not check('enable=False leaves the reference untouched',
                       (NAME[sd], off, np.allclose(apply_offset(X, off, OFF), X)),
                       ('FREE', 0.0, True))

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILED"}\n')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
