"""Plot offline GMPC results.

For each saved logs/<name>.npz:
  - top-left:   x-y trajectory  (ref vs actual, with initial-offset marker)
  - top-right:  yaw vs time     (ref vs actual)
  - mid-left:   body-frame error (e_vx, e_vy, e_ω) vs time
  - mid-right:  applied vs ref body twist (vx, vy, ω) vs time
  - bottom:     OSQP solve-time histogram + summary text

Also writes a single summary figure across all 5 trajectories:
  - RMSE bar chart
  - solve-time bar chart (mean / p95 / max)

Usage:
    python3 plot.py              # plot everything in logs/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')                                   # headless
import matplotlib.pyplot as plt


LOG_DIR  = Path(__file__).parent / 'logs'
PLOT_DIR = Path(__file__).parent / 'plots'


# ---------------------------------------------------------------------------
# Per-trajectory figure
# ---------------------------------------------------------------------------

def plot_one(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    name = str(data['name'])

    X_actual = data['X_actual_xyth']        # (N+1, 3)
    X_ref    = data['X_ref_xyth']
    e        = data['e']                    # (N+1, 3) — body-frame error
    u        = data['u_applied']            # (N, 3)
    xi_ref   = data['xi_ref']               # (N, 3)
    t        = data['t']                    # (N+1,)
    solve_ms = data['solve_time_ms']        # (N,)

    fig = plt.figure(figsize=(13, 10))
    gs  = fig.add_gridspec(3, 2)

    # --- xy trajectory ---
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xy.plot(X_ref[:, 0],    X_ref[:, 1],    'k--', lw=1.5, label='reference')
    ax_xy.plot(X_actual[:, 0], X_actual[:, 1], 'C0-', lw=1.5, label='actual')
    ax_xy.scatter([X_actual[0, 0]], [X_actual[0, 1]],
                  marker='o', color='red', s=60, zorder=5, label='start (offset)')
    ax_xy.scatter([X_ref[0, 0]], [X_ref[0, 1]],
                  marker='x', color='black', s=60, zorder=5, label='ref start')
    ax_xy.set_xlabel('x [m]')
    ax_xy.set_ylabel('y [m]')
    ax_xy.set_title(f'{name} — xy trajectory')
    ax_xy.set_aspect('equal', adjustable='datalim')
    ax_xy.grid(alpha=0.3)
    ax_xy.legend(loc='best', fontsize=9)

    # --- yaw vs time ---
    ax_th = fig.add_subplot(gs[0, 1])
    ax_th.plot(t, X_ref[:, 2],    'k--', lw=1.5, label='ref θ')
    ax_th.plot(t, X_actual[:, 2], 'C0-', lw=1.5, label='actual θ')
    ax_th.set_xlabel('t [s]')
    ax_th.set_ylabel('θ [rad]')
    ax_th.set_title('yaw — note ref wraps in (-π, π]')
    ax_th.grid(alpha=0.3)
    ax_th.legend(loc='best', fontsize=9)

    # --- error vs time ---
    ax_e = fig.add_subplot(gs[1, 0])
    ax_e.plot(t, e[:, 0], 'C0-', label='$e_{vx}$ (body)')
    ax_e.plot(t, e[:, 1], 'C1-', label='$e_{vy}$ (body)')
    ax_e.plot(t, e[:, 2], 'C2-', label='$e_\\omega$ [rad]')
    ax_e.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax_e.set_xlabel('t [s]')
    ax_e.set_ylabel('geodesic error')
    ax_e.set_title('SE(2) error (no wrap)')
    ax_e.grid(alpha=0.3)
    ax_e.legend(loc='best', fontsize=9)

    # --- input vs ref ---
    ax_u = fig.add_subplot(gs[1, 1])
    tu = t[:-1]
    for i, lbl in enumerate(['vx', 'vy', 'ω']):
        ax_u.plot(tu, xi_ref[:, i],   f'C{i}--', alpha=0.5, label=f'{lbl}_ref')
        ax_u.plot(tu, u[:, i],        f'C{i}-',  label=f'{lbl}_app')
    ax_u.set_xlabel('t [s]')
    ax_u.set_ylabel('body twist')
    ax_u.set_title('applied vs reference twist')
    ax_u.grid(alpha=0.3)
    ax_u.legend(loc='best', fontsize=8, ncol=3)

    # --- solve time hist ---
    ax_st = fig.add_subplot(gs[2, :])
    ax_st.hist(solve_ms, bins=40, color='C3', alpha=0.7, edgecolor='k')
    ax_st.axvline(float(data['solve_mean_ms']), color='k', ls='-',
                  lw=1.5, label=f'mean {float(data["solve_mean_ms"]):.3f} ms')
    ax_st.axvline(float(data['solve_p95_ms']),  color='C0', ls='--',
                  lw=1.5, label=f'p95  {float(data["solve_p95_ms"]):.3f} ms')
    ax_st.axvline(float(data['solve_max_ms']),  color='C2', ls=':',
                  lw=1.5, label=f'max  {float(data["solve_max_ms"]):.3f} ms')
    ax_st.set_xlabel('OSQP solve time [ms]')
    ax_st.set_ylabel('steps')
    ax_st.set_title(
        f'solve time   |   RMSE_xy={float(data["rmse_pos"]):.4f} m,  '
        f'RMSE_yaw={float(data["rmse_yaw"]):.4f} rad,  '
        f'infeasible={int(data["n_infeasible"])} steps'
    )
    ax_st.legend(loc='best', fontsize=9)
    ax_st.grid(alpha=0.3)

    fig.suptitle(f'GMPC offline — {name}', fontsize=13, fontweight='bold')
    fig.tight_layout()

    out = PLOT_DIR / f'{name}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Summary across all trajectories
# ---------------------------------------------------------------------------

def plot_summary(npz_paths):
    names    = []
    rmse_pos = []
    rmse_yaw = []
    mean_ms  = []
    p95_ms   = []
    max_ms   = []
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        names.append(str(d['name']))
        rmse_pos.append(float(d['rmse_pos']))
        rmse_yaw.append(float(d['rmse_yaw']))
        mean_ms.append(float(d['solve_mean_ms']))
        p95_ms.append(float(d['solve_p95_ms']))
        max_ms.append(float(d['solve_max_ms']))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(names))

    # --- RMSE bars ---
    w = 0.4
    axes[0].bar(x - w/2, rmse_pos, width=w, label='RMSE xy [m]',   color='C0')
    axes[0].bar(x + w/2, rmse_yaw, width=w, label='RMSE yaw [rad]', color='C1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha='right')
    axes[0].set_ylabel('tracking RMSE')
    axes[0].set_title('SE(2)-GMPC tracking RMSE — 5 reference trajectories')
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].legend()

    # --- solve time bars ---
    axes[1].bar(x - w,   mean_ms, width=w, label='mean', color='C0')
    axes[1].bar(x,       p95_ms,  width=w, label='p95',  color='C1')
    axes[1].bar(x + w,   max_ms,  width=w, label='max',  color='C2')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha='right')
    axes[1].set_ylabel('OSQP solve time [ms]')
    axes[1].set_title('SE(2)-GMPC OSQP solve time')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].legend()

    fig.tight_layout()
    out = PLOT_DIR / '00_summary.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(LOG_DIR.glob('*.npz'))
    if not paths:
        print(f'No logs in {LOG_DIR}. Run run.py first.')
        return

    for p in paths:
        out = plot_one(p)
        print(f'  wrote {out.name}')
    out = plot_summary(paths)
    print(f'  wrote {out.name}')


if __name__ == '__main__':
    main()
