"""Shield unit tests, run without ROS.

    python3 evaluation/test_shield.py

The cases that matter are the crowded ones. Two sign errors got through earlier
review because every hand-written case had one to five returns in clean
geometry: with so few constraints the feasible set stays large and a wrong sign
still produces a plausible-looking command. The scenes below reproduce what the
lidar actually delivers -- 200+ returns from every direction while hugging a
wall or crossing a doorway -- which is where a wrong relaxation collapses the
feasible set to the origin.
"""
import sys

import numpy as np

R, ALPHA, D0, TAU, ABRAKE, EPS = 0.30, 2.0, 0.05, 0.15, 6.25, 0.05
UMAX = np.array([0.2775, 0.2775, 1.1327])
ITERS, FB_ITERS = 6, 30


def rows(pts, u):
    pts = np.asarray(pts, float).reshape(-1, 2)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    d = rad - R
    n = pts / np.maximum(rad[:, None], 1e-9)
    Jr = np.stack([-n[:, 1], n[:, 0]], 1) * R
    a = np.concatenate([n, (n * Jr).sum(1, keepdims=True)], 1)
    v_app = np.maximum(0.0, a @ u)
    d_stop = D0 + v_app * TAU + v_app ** 2 / (2 * ABRAKE) + EPS
    return a, ALPHA * (d - d_stop), d


def sweep(A, B, x0, n_it):
    x = x0.copy()
    a2 = np.maximum((A * A).sum(1), 1e-9)
    for it in range(1, n_it + 1):
        v = A @ x - B
        i = int(np.argmax(v))
        if v[i] <= 1e-6:
            return x, it
        x = x - (v[i] / a2[i]) * A[i]
        x = np.clip(x, -UMAX, UMAX)
    return x, n_it


def limit(u, pts):
    box_a = np.zeros((6, 3)); box_b = np.zeros(6)
    for k in range(3):
        box_a[2*k, k] = 1.0;   box_b[2*k] = UMAX[k]
        box_a[2*k+1, k] = -1.0; box_b[2*k+1] = UMAX[k]
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return np.clip(u, -UMAX, UMAX), 0.0, False
    a, b, d = rows(pts, u)
    A = np.vstack([a, box_a]); B = np.concatenate([b, box_b])
    out, _ = sweep(A, B, u, ITERS)
    after = float(np.max(A @ out - B))
    fb = False
    if after > 1e-3:
        fb = True
        # relax the violated rows only; distant rows keep their generous bound
        B2 = np.concatenate([np.maximum(b, 0.0), box_b])
        out, _ = sweep(A, B2, u, FB_ITERS)
        if float(np.max(A @ out - B2)) > 1e-3:
            out = np.zeros(3)
        after = float(np.max(A @ out - B))
    return out, after, fb


def approach(out, pts):
    """Largest rate at which any body surface point closes on any return."""
    P = np.asarray(pts, float).reshape(-1, 2)
    if not len(P):
        return -9.9
    a, _, _ = rows(P, np.zeros(3))
    return float(np.max(a @ out))


def ring(radius, n=180, span=2 * np.pi, start=0.0):
    t = start + np.linspace(0, span, n, endpoint=False)
    return np.stack([radius * np.cos(t), radius * np.sin(t)], 1)


FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


print("--- sparse geometry ---")
o, af, fb = limit(np.array([0.2775, 0, 0]), [[0.50 + R, 0]])
check('far obstacle untouched', abs(o[0] - 0.2775) < 1e-3, f'vx={o[0]:+.3f}')
o, af, fb = limit(np.array([0.2775, 0, 0]), [[0.10 + R, 0]])
check('close obstacle reverses', o[0] < 0, f'vx={o[0]:+.3f}')
o, af, fb = limit(np.array([0, 0.2775, 0]), [[0.12 + R, 0]])
check('tangential preserved', abs(o[1] - 0.2775) < 1e-3, f'vy={o[1]:+.3f}')
o, af, fb = limit(np.array([-0.2775, 0, 0]), [[0.12 + R, 0]])
check('retreat preserved', abs(o[0] + 0.2775) < 1e-3, f'vx={o[0]:+.3f}')
o, af, fb = limit(np.array([0, 0, 1.1327]), [[0.12 + R, 0]])
check('disc spin untouched', abs(o[2] - 1.1327) < 1e-3, f'wz={o[2]:+.3f}')

print("\n--- crowded geometry (the cases that caught two sign errors) ---")
# A full ring at 1.8 m plus one return at 2 cm: the wall-hugging case. A
# relaxation that clamps DISTANT bounds down to zero freezes the robot here.
pts = np.vstack([[[R + 0.02, 0.0]], ring(R + 1.8, 24)])
u = np.array([-0.14, 0.09, -0.01])          # GMPC asking to retreat
o, af, fb = limit(u, pts)
check('surrounded: retreat survives', np.linalg.norm(o - u) < 0.02,
      f'in={u.round(3)} out={o.round(3)} fb={int(fb)}')
check('surrounded: not frozen', np.linalg.norm(o) > 0.05, f'|v|={np.linalg.norm(o):.3f}')

# 215 returns, the count measured in a real corridor
pts = np.vstack([ring(R + 0.35, 100), ring(R + 1.2, 115)])
o, af, fb = limit(np.array([0.2775, 0, 0]), pts)
check('215 returns: residual met', af <= 1e-3, f'resid={af:+.5f} fb={int(fb)}')
# Not "no approach": the barrier PERMITS closing at alpha*(d - d_stop), which
# at 0.35 m is 0.404 m/s -- far above anything the chassis can do. Requiring
# zero approach here would be asserting a behaviour the design deliberately
# does not have. What must hold is the barrier itself.
a_r, b_r, _ = rows(pts, np.array([0.2775, 0, 0]))
check('215 returns: barrier holds', float(np.max(a_r @ o - b_r)) <= 1e-3,
      f'max(a.v - b)={float(np.max(a_r @ o - b_r)):+.5f}')

# Doorway: walls either side at 0.15 m, nothing ahead. Must pass through.
gap = np.vstack([[[0.0, 0.15 + R]] * 1, [[0.0, -(0.15 + R)]] * 1,
                 [[x, 0.15 + R] for x in np.linspace(-1, 1, 40)],
                 [[x, -(0.15 + R)] for x in np.linspace(-1, 1, 40)]])
o, af, fb = limit(np.array([0.2775, 0, 0]), gap)
check('doorway: passes through', abs(o[0] - 0.2775) < 1e-3, f'vx={o[0]:+.3f}')

# Pinched front and back: the full barrier cannot hold, fallback must engage
# and the output must still not close on anything.
o, af, fb = limit(np.array([0.2775, 0, 0]),
                  [[0.05 + R, 0], [-(0.05 + R), 0]])
check('pinched: fallback engages', fb, f'fb={int(fb)}')
check('pinched: no approach', approach(o, [[0.05 + R, 0], [-(0.05 + R), 0]]) <= 1e-3,
      f'approach={approach(o, [[0.05+R,0],[-(0.05+R),0]]):+.4f}')

print("\n--- projection order independence ---")
post = [[0.10 + R, 0.02], [0.11 + R, -0.03], [0.13 + R, 0.06],
        [0.30, 0.30 + R], [0.35, -(0.32 + R)]]
rng = np.random.default_rng(0)
outs = []
for _ in range(8):
    p = list(post); rng.shuffle(p)
    outs.append(limit(np.array([0.2775, 0.10, 0]), p)[0])
spread = max(np.linalg.norm(a - b) for a in outs for b in outs)
check('order independent', spread < 1e-6, f'spread={spread:.2e}')

print("\n" + ("FAILURES: " + ", ".join(FAILS) if FAILS else "all shield tests pass"))
sys.exit(1 if FAILS else 0)
