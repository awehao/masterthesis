"""Load the REAL benchmark scenario (random_room) into the 2D sandbox.

Provides the actual walls (occupancy grid), the 4 unknown static pillars, the 4
ping-pong dynamic obstacles, an A* global path start->goal, and a distance-
transform nearest-wall field (for the map-based static-CBF). This makes the fast
2D sim a faithful top-down replica of the gz benchmark, so an RL residual trained
here transfers back.
"""
from __future__ import annotations
import heapq
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
    dict(name='dyn_obs_1', start=(0.0, 5.55), end=(0.0, 3.0),  speed=0.15, r=0.25),
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
        # inflated occupancy for A* (keep robot radius clear)
        rad = int(round((ROBOT_R + 0.10) / RES))
        self.occ_infl = binary_dilation(self.occ, iterations=rad)
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

    def occ_with_pillars(self, pillars):
        """Wall-inflated occupancy PLUS discovered static pillars stamped in
        (robot-radius inflated). Dynamic obstacles are NOT included -> mimics the
        real /scan_filtered costmap the SmacPlanner sees."""
        occ = self.occ_infl.copy()
        for (x, y, r) in pillars:
            rc = int(round((r + ROBOT_R + 0.10) / RES))
            row, col = self.world_to_cell((x, y))
            r0, r1 = max(0, row - rc), min(self.H, row + rc + 1)
            c0, c1 = max(0, col - rc), min(self.W, col + rc + 1)
            occ[r0:r1, c0:c1] = True
        return occ

    # --- A* on the inflated grid (8-connectivity) ---
    def astar(self, start_w, goal_w, occ=None):
        occ = self.occ_infl if occ is None else occ
        s = self.world_to_cell(start_w); g = self.world_to_cell(goal_w)
        def h(a, b): return np.hypot(a[0] - b[0], a[1] - b[1])
        openq = [(h(s, g), 0.0, s, None)]
        came = {}; gscore = {s: 0.0}
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while openq:
            _, gc, cur, par = heapq.heappop(openq)
            if cur in came: continue
            came[cur] = par
            if cur == g: break
            for dr, dc in nbrs:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < self.H and 0 <= nc < self.W): continue
                if occ[nr, nc]: continue
                ng = gc + h(cur, (nr, nc))
                if ng < gscore.get((nr, nc), 1e18):
                    gscore[(nr, nc)] = ng
                    heapq.heappush(openq, (ng + h((nr, nc), g), ng, (nr, nc), cur))
        if g not in came:
            return np.array([start_w, goal_w])              # fallback straight line
        path = []; c = g
        while c is not None:
            path.append(self.cell_to_world(*c)); c = came[c]
        return np.array(path[::-1])

    @staticmethod
    def smooth_path(path, ds=0.15, w_data=0.2, w_smooth=0.25, iters=150):
        """Resample by arc length then gradient-descent smooth (= nav2 SimpleSmoother):
        minimise w_data*||p-orig||^2 + w_smooth*||p_{i-1}-2p_i+p_{i+1}||^2.
        Turns the jagged 8-connected A* staircase into a smooth reference path.
        Stability needs w_data + 2*w_smooth <= 1 (else the diffusion term diverges)."""
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(s[-1] / ds))
        tgt = np.linspace(0, s[-1], n)
        rs = np.stack([np.interp(tgt, s, path[:, 0]), np.interp(tgt, s, path[:, 1])], 1)
        p = rs.copy(); orig = rs.copy()
        for _ in range(iters):
            p[1:-1] += (w_data * (orig[1:-1] - p[1:-1]) +
                        w_smooth * (p[:-2] + p[2:] - 2 * p[1:-1]))
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
