"""Summarise the overnight arms into one markdown page.

Reports per-cycle quantities alongside contact counts, because the measured
trial-to-trial spread (0.22-0.37 m) exceeds the between-config effect (0.11 m),
so a contact count at n=40 cannot on its own separate a real difference from
encounter phase. The barrier audit is the opposite: thousands of samples inside
each trial, unaffected by which mover the robot happened to meet.
"""
import csv, glob, math, os, sys, statistics as st

sys.path.insert(0, 'evaluation')
os.environ.setdefault('BIGARENA', '1')

ARMS = [('slackA', 'A/B: soft slack (old movers)'),
        ('slackB', 'A/B: hard static k0 (old movers)'),
        ('hardC',  'hard static k0 + mover cap 0.14'),
        ('softD',  'soft slack + mover cap 0.14'),
        ('repE',   'replicate of the cleaner arm')]

F = ['cycle', 'rx', 'ry', 'rx_seq', 'rx_n', 'held', 'appended', 'near_d',
     'near_r', 'near_m', 'h_entry', 'h_build_stat', 'h_build_dyn', 'min_h',
     'rows', 'msg_age', 'eps0', 'resid_noslack', 'resid_slack']


def csv_stats(path):
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path)) if r.get('min_clearance_m')]
    if not rows:
        return None
    clr = [float(r['min_clearance_m']) for r in rows]
    arr = sum(1 for r in rows if r['success'] == 'True')
    t = [float(r['arrival_time_s']) for r in rows
         if r['success'] == 'True' and r['arrival_time_s']]
    j = [float(r['jerk_vx']) for r in rows if r.get('jerk_vx')]
    p95 = [float(r['solve_time_p95_ms']) for r in rows if r.get('solve_time_p95_ms')]
    return dict(n=len(rows), arrived=arr, neg=sum(1 for c in clr if c < 0),
                med=st.median(clr), worst=min(clr),
                t=st.median(t) if t else float('nan'),
                jerk=st.median(j) if j else float('nan'),
                p95=st.median(p95) if p95 else float('nan'),
                per_run={r['run']: float(r['min_clearance_m']) for r in rows})


def barrier_audit(archive):
    """Cycles where the command itself broke the barrier and only the slack
    kept the QP feasible: A z - l < 0, eps > 0, A z + eps - l >= 0."""
    try:
        import numpy as np, rosbag2_py
        from rclpy.serialization import deserialize_message
        from std_msgs.msg import Float32MultiArray
    except Exception as e:
        return f"(rosbag unavailable: {e})"
    tot = viol = slack_saved = held_not_app = 0
    worst = 0.0
    for bag in sorted(glob.glob(f'{archive}/gmpc_cbf__scan_seed*')):
        if not os.path.exists(f'{bag}/metadata.yaml'):
            continue
        rd = rosbag2_py.SequentialReader()
        try:
            rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        except Exception:
            continue
        if '/gmpc/diag' not in {t.name for t in rd.get_all_topics_and_types()}:
            continue
        rd.set_filter(rosbag2_py.StorageFilter(topics=['/gmpc/diag']))
        while rd.has_next():
            _, buf, _ = rd.read_next()
            d = list(deserialize_message(buf, Float32MultiArray).data)
            if len(d) < len(F):
                continue
            v = dict(zip(F, d))
            tot += 1
            if v['held'] > 0 and v['appended'] == 0:
                held_not_app += 1
            rn, rs, e0 = v['resid_noslack'], v['resid_slack'], v['eps0']
            if math.isfinite(rn) and rn < -1e-6:
                viol += 1
                worst = min(worst, rn)
                if e0 > 1e-9 and math.isfinite(rs) and rs >= -1e-6:
                    slack_saved += 1
    if not tot:
        return "(no /gmpc/diag recorded)"
    return (f"cycles {tot}, wall points dropped before solve {held_not_app}, "
            f"barrier broken without slack {viol} ({viol/tot:.2%}), "
            f"of which slack kept feasible {slack_saved}, worst residual {worst:+.4f}")


print("# Overnight results\n")
print("Scenario: bigarena, 40 random routes, GMPC+CBF, mask 10 deg, "
      "fixed margins 0.60/0.38, hardware motion limits with wheel-speed "
      "coupling.\n")
print("| arm | n | arrived | contacts | median clr | worst | arrival | jerk_vx | p95 solve |")
print("|---|---|---|---|---|---|---|---|---|")
stats = {}
for key, label in ARMS:
    s = csv_stats(f'evaluation/results/{key}/batch.csv')
    stats[key] = s
    if not s:
        print(f"| {label} | - | - | (not run) | | | | | |")
        continue
    print(f"| {label} | {s['n']} | {s['arrived']} | **{s['neg']}** | "
          f"{s['med']:+.3f} | {s['worst']:+.3f} | {s['t']:.0f} s | "
          f"{s['jerk']:.3f} | {s['p95']:.1f} ms |")

print("\n## Barrier audit (per control cycle)\n")
print("`A z - l < 0` means the command itself broke the CBF inequality; h < 0 "
      "alone does not, since a robot already inside the keep-out but leaving "
      "it still satisfies the condition.\n")
for key, label in ARMS:
    if stats.get(key):
        print(f"- **{label}**: {barrier_audit(f'evaluation/bags/archive_{key}')}")

# paired comparison, same routes
if stats.get('hardC') and stats.get('softD'):
    a, b = stats['hardC']['per_run'], stats['softD']['per_run']
    common = sorted(set(a) & set(b), key=lambda s: int(s.replace('scan_seed', '')))
    if common:
        na = sum(1 for s in common if a[s] < 0)
        nb = sum(1 for s in common if b[s] < 0)
        print(f"\n## Paired on the same {len(common)} routes\n")
        print(f"- hard static k0: {na} contacts   soft: {nb} contacts")
        disc_a = [s for s in common if a[s] < 0 <= b[s]]
        disc_b = [s for s in common if b[s] < 0 <= a[s]]
        print(f"- only hard contacts: {len(disc_a)}   only soft contacts: {len(disc_b)}")
        try:
            from scipy.stats import binomtest
            n01, n10 = len(disc_b), len(disc_a)
            if n01 + n10:
                print(f"- McNemar p = {binomtest(n01, n01+n10).pvalue:.4f}")
        except Exception:
            pass

# replicate vs its original
if stats.get('repE'):
    src = 'hardC' if stats.get('hardC') and (
        not stats.get('softD') or stats['hardC']['neg'] <= stats['softD']['neg']) else 'softD'
    if stats.get(src):
        a, b = stats[src]['per_run'], stats['repE']['per_run']
        common = sorted(set(a) & set(b))
        if common:
            print(f"\n## Replicate ({src} run twice, same {len(common)} routes)\n")
            print(f"- contacts: {sum(1 for s in common if a[s]<0)} vs "
                  f"{sum(1 for s in common if b[s]<0)}")
            d = [abs(a[s] - b[s]) for s in common]
            print(f"- same-route clearance spread: median {st.median(d):.3f} m, "
                  f"max {max(d):.3f} m")
            print("\nThis is the noise floor: any config difference smaller "
                  "than it cannot be resolved at n=40.")
