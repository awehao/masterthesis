"""Load the REAL benchmark scenario (random_room) into the 2D sandbox.

Provides the actual walls (occupancy grid), the 4 unknown static pillars, the 4
ping-pong dynamic obstacles, an A* global path start->goal, and a distance-
transform nearest-wall field (for the map-based static-CBF). This makes the fast
2D sim a faithful top-down replica of the gz benchmark, so an RL residual trained
here transfers back.
"""
from __future__ import annotations
import heapq
import math
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt

MAP_PGM = '/home/howardchen/masterthesis/src/ammr_bringup/maps/random_room.pgm'
RES = 0.05
ORIGIN = np.array([-1.5, -1.5])            # world coord of pixel (row=H-1, col=0)
START = np.array([0.0, 0.0])
GOAL = np.array([17.0, 17.0])
ROBOT_R = 0.30

# unknown static pillars (not in prior map) : (x, y, r)
PILLARS = [(4.0, 3.0, 0.30), (9.0, 8.0, 0.30), (14.0, 13.0, 0.30), (2.2, 6.5, 0.30)]

# dynamic obstacles: ping-pong start<->end at speed, radius 0.25
DYN = [
    dict(name='dyn_obs_1', start=(0.0, 5.55), end=(0.0, 3.0),  speed=0.13, r=0.25),
    dict(name='dyn_obs_2', start=(10.3, 8.4), end=(11.9, 7.5), speed=0.30, r=0.25),
    dict(name='dyn_obs_0', start=(14.1, 7.1), end=(17.1, 7.1), speed=0.30, r=0.25),
    dict(name='dyn_obs_5', start=(5.0, 8.2),  end=(5.0, 9.8),  speed=0.30, r=0.25),
]


class Scenario:
    def __init__(self):
        img = np.array(Image.open(MAP_PGM))                 # (H, W), 0=black=wall
        p_norm = (255 - img) / 255.0
        self.occ = p_norm > 0.65                            # occupied mask (row, col)
        self.H, self.W = self.occ.shape
        # nearest-wall field on RAW walls (for static-CBF)
        dist, idx = distance_transform_edt(~self.occ, return_indices=True)
        self.wall_dist = dist * RES                         # metres
        self.wall_nn = idx                                  # [0]=row, [1]=col of nearest wall
        # Inflated occupancy for A* (keep robot radius clear).
        #
        # This used to be binary_dilation(occ, iterations=8). scipy's default
        # structuring element is the 4-connected cross, so N iterations grow the
        # obstacle by N cells in L1, i.e. a DIAMOND: 0.40 m along the axes but
        # only 0.40/sqrt(2) = 0.283 m diagonally. Measured on this map the A*
        # path came within 0.283 m of a wall, 0.017 m inside the 0.30 m robot,
        # and the smoother then cut that to 0.250 m. Thresholding the exact
        # Euclidean distance transform gives a true disc inflation instead.
        self.infl_radius = ROBOT_R + 0.10
        self.occ_infl = self.wall_dist < self.infl_radius
        self.path = self.astar(START, GOAL)

    # --- coordinate transforms (ROS map convention, y flipped) ---
    def world_to_cell(self, xy):
        col = int((xy[0] - ORIGIN[0]) / RES)
        row = int(self.H - 1 - (xy[1] - ORIGIN[1]) / RES)
        return row, col

    def cell_to_world(self, row, col):
        x = ORIGIN[0] + (col + 0.5) * RES
        y = ORIGIN[1] + (self.H - 1 - row + 0.5) * RES
        return np.array([x, y])

    def nearest_wall(self, xy):
        row, col = self.world_to_cell(xy)
        row = np.clip(row, 0, self.H - 1); col = np.clip(col, 0, self.W - 1)
        wr = int(self.wall_nn[0, row, col]); wc = int(self.wall_nn[1, row, col])
        return self.cell_to_world(wr, wc), float(self.wall_dist[row, col])

    def nearest_wall_batch(self, pts):
        """Vectorised nearest_wall for an (M,2) array -> (wall_pts (M,2), dist (M,))."""
        pts = np.asarray(pts, float)
        col = np.clip(((pts[:, 0] - ORIGIN[0]) / RES).astype(int), 0, self.W - 1)
        row = np.clip((self.H - 1 - (pts[:, 1] - ORIGIN[1]) / RES).astype(int),
                      0, self.H - 1)
        wr = self.wall_nn[0, row, col]
        wc = self.wall_nn[1, row, col]
        wx = ORIGIN[0] + (wc + 0.5) * RES
        wy = ORIGIN[1] + (self.H - 1 - wr + 0.5) * RES
        return np.stack([wx, wy], 1), self.wall_dist[row, col]

    def occ_with_pillars(self, pillars):
        """Wall-inflated occupancy PLUS discovered static pillars stamped in
        (robot-radius inflated). Dynamic obstacles are NOT included -> mimics the
        real /scan_filtered costmap the SmacPlanner sees."""
        occ = self.occ_infl.copy()
        for (x, y, r) in pillars:
            # circular stamp; a square one (the previous version) over-inflates
            # by sqrt(2) at the corners, which can wall off passages the robot
            # actually fits through and push the plan somewhere it need not go
            need = r + self.infl_radius
            rc = int(np.ceil(need / RES))
            row, col = self.world_to_cell((x, y))
            r0, r1 = max(0, row - rc), min(self.H, row + rc + 1)
            c0, c1 = max(0, col - rc), min(self.W, col + rc + 1)
            rr, cc = np.ogrid[r0:r1, c0:c1]
            wx = ORIGIN[0] + (cc + 0.5) * RES
            wy = ORIGIN[1] + (self.H - 1 - rr + 0.5) * RES
            occ[r0:r1, c0:c1] |= (np.hypot(wx - x, wy - y) < need)
        return occ

    # --- A* on the inflated grid (8-connectivity) ---
    def astar(self, start_w, goal_w, occ=None):
        occ = self.occ_infl if occ is None else occ
        s = self.world_to_cell(start_w); g = self.world_to_cell(goal_w)
        # math.hypot on scalars (np.hypot pays numpy-scalar overhead ~1e6 times
        # per plan); neighbour step costs are precomputed (1 or sqrt(2)).
        hyp = math.hypot
        gr, gc_ = g
        H, W = self.H, self.W
        nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142135623730951), (-1, 1, 1.4142135623730951),
                (1, -1, 1.4142135623730951), (1, 1, 1.4142135623730951))
        openq = [(hyp(s[0] - gr, s[1] - gc_), 0.0, s, None)]
        came = {}; gscore = {s: 0.0}
        while openq:
            _, gc, cur, par = heapq.heappop(openq)
            if cur in came: continue
            came[cur] = par
            if cur == g: break
            cr, cc = cur
            for dr, dc, step in nbrs:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W): continue
                if occ[nr, nc]: continue
                ng = gc + step
                key = (nr, nc)
                if ng < gscore.get(key, 1e18):
                    gscore[key] = ng
                    heapq.heappush(openq, (ng + hyp(nr - gr, nc - gc_), ng, key, cur))
        if g not in came:
            return np.array([start_w, goal_w])              # fallback straight line
        path = []; c = g
        while c is not None:
            path.append(self.cell_to_world(*c)); c = came[c]
        return np.array(path[::-1])

    def smooth_path(self, path, ds=0.15, w_data=0.2, w_smooth=0.25, iters=150,
                    min_clearance=None, w_obs=0.6):
        """Resample by arc length then gradient-descent smooth (= nav2 SimpleSmoother):
        minimise w_data*||p-orig||^2 + w_smooth*||p_{i-1}-2p_i+p_{i+1}||^2.
        Turns the jagged 8-connected A* staircase into a smooth reference path.
        Stability needs w_data + 2*w_smooth <= 1 (else the diffusion term diverges).

        The pure smoother CUTS CORNERS, and a corner cut across a doorway lands
        inside the wall: measured on this map it produced paths whose closest
        approach was 0.250 m from a wall, i.e. 0.05 m INSIDE the 0.30 m robot,
        with 12.6% of points inside the static-CBF keep-out. Real nav2 does not
        do that (min 0.350 m, 0% inside), so the 2D sandbox was handing the
        controller a reference no real planner would emit, and the CBF spent the
        whole run fighting it. Any conclusion drawn from that sandbox about
        smoothness was therefore measuring the wrong thing.

        Fix: after each smoothing step, push any point that came closer than
        `min_clearance` back out along the direction away from its nearest wall
        cell. `min_clearance` defaults to the robot radius plus the same 0.05 m
        real nav2 leaves."""
        # default: do not erode the clearance A* already guaranteed
        if min_clearance is None:
            min_clearance = self.infl_radius
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(s[-1] / ds))
        tgt = np.linspace(0, s[-1], n)
        rs = np.stack([np.interp(tgt, s, path[:, 0]), np.interp(tgt, s, path[:, 1])], 1)
        p = rs.copy(); orig = rs.copy()
        for _ in range(iters):
            p[1:-1] += (w_data * (orig[1:-1] - p[1:-1]) +
                        w_smooth * (p[:-2] + p[2:] - 2 * p[1:-1]))
            if min_clearance > 0.0 and len(p) > 2:
                wall, d = self.nearest_wall_batch(p[1:-1])
                too_close = d < min_clearance
                if too_close.any():
                    away = p[1:-1][too_close] - wall[too_close]
                    nrm = np.linalg.norm(away, axis=1, keepdims=True)
                    # a point sitting exactly on a wall cell has no direction to
                    # escape along; nudge it back toward its pre-smoothing spot
                    degenerate = nrm[:, 0] < 1e-9
                    if degenerate.any():
                        alt = orig[1:-1][too_close][degenerate] - wall[too_close][degenerate]
                        away[degenerate] = alt
                        nrm[degenerate, 0] = np.maximum(
                            np.linalg.norm(alt, axis=1), 1e-9)
                    push = (min_clearance - d[too_close])[:, None] * away / nrm
                    p[1:-1][too_close] += w_obs * push
        return p

    # --- dynamic obstacle position at time t (ping-pong) ---
    @staticmethod
    def dyn_at(o, t):
        a = np.array(o['start']); b = np.array(o['end'])
        L = np.linalg.norm(b - a)
        if L < 1e-6: return a, np.zeros(2)
        period = 2 * L / o['speed']
        phase = (t % period) / period
        if phase < 0.5:
            s = phase * 2; pos = a + (b - a) * s; vel = (b - a) / L * o['speed']
        else:
            s = (phase - 0.5) * 2; pos = b + (a - b) * s; vel = (a - b) / L * o['speed']
        return pos, vel


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP'], 'axes.unicode_minus': False})
    sc = Scenario()
    fig, ax = plt.subplots(figsize=(8, 8))
    ext = [ORIGIN[0], ORIGIN[0] + sc.W * RES, ORIGIN[1], ORIGIN[1] + sc.H * RES]
    ax.imshow(sc.occ, cmap='Greys', origin='upper', extent=ext, alpha=0.85)
    ax.plot(sc.path[:, 0], sc.path[:, 1], '-', color='#1565c0', lw=2, label='A* 全域路徑')
    ax.plot(*START, 'o', color='green', ms=12, label='起點')
    ax.plot(*GOAL, '*', color='red', ms=20, label='終點')
    for (x, y, r) in PILLARS:
        ax.add_patch(plt.Circle((x, y), r, color='gold', alpha=0.8))
    for o in DYN:
        a, b = np.array(o['start']), np.array(o['end'])
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='#c62828', lw=1.5)
        ax.add_patch(plt.Circle(a, o['r'], color='#37474f', alpha=0.7))
    ax.plot([], [], color='gold', lw=6, label='未知靜態柱')
    ax.plot([], [], color='#c62828', label='動態障礙 sweep')
    ax.set_title('真實地圖場景載入 2D(random_room)', fontsize=13, fontweight='bold')
    ax.set_aspect('equal'); ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    out = '/home/howardchen/masterthesis/evaluation/results/figs/rl_scenario_realmap.png'
    plt.savefig(out, dpi=140, bbox_inches='tight'); print('saved ->', out)
    print('path pts:', len(sc.path), ' length m:',
          float(np.sum(np.linalg.norm(np.diff(sc.path, axis=0), axis=1))))
