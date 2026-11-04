# ===== Summary and pre-registered comparisons — v2 =====
import numpy as np
from atlas_lib import paired

HYPOTHESES = [
    ('G1 stronger learner beats random features (oracle means)',        ('A', 'O_full'), ('A', 'RC_LDA')),
    ('P1 per-class GREG beats global map (v1: +4.3)',                    ('A', 'GR_prev_buf'), ('A', 'CH_cv_buf')),
    ('P2 adding r as auxiliary helps',                                    ('A', 'GR_multi_buf'), ('A', 'GR_prev_buf')),
    ('P3 matched size: reserve bridge beats buffer bridge (= training bias is real)', ('A', 'GR_prev_res_m'), ('A', 'GR_prev_buf_m')),
    ('P3b matched size, raw means: reserve beats buffer',                 ('A', 'B_only_res_m'), ('A', 'B_only_buf_m')),
    ('P4 PoE beats random features alone',                                ('A', 'PoE[RC+GR_prev_buf]'), ('A', 'RC_LDA')),
    ('P4b PoE beats migrated learned readout alone',                      ('A', 'PoE[RC+GR_prev_buf]'), ('A', 'GR_prev_buf')),
    ('P5 equal bytes: PoE beats X2-style global map',                     ('E', 'PoE[RC+GR_prev_buf]'), ('E', 'CH_cv_buf')),
    ('P5b equal bytes: PoE beats random features alone',                  ('E', 'PoE[RC+GR_prev_buf]'), ('E', 'RC_LDA')),
    ('P6 X2 replication: global map beats buffer-only',                   ('A', 'CH_cv_buf'), ('A', 'B_only_buf')),
    ('P7 migration loss left inside the PoE (oracle means vs migrated)',  ('A', 'PoE[RC+O_full]'), ('A', 'PoE[RC+GR_prev_buf]')),
]

def table(results, cfg, metric):
    runs = results[cfg]; seeds = sorted(s for s in runs if 'error' not in runs[s])
    arms = list(runs[seeds[0]][metric].keys())
    rows = []
    for a in arms:
        v = np.array([runs[s][metric][a] for s in seeds])
        rows.append((a, float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')))
    rev = metric != 'nll'
    return sorted(rows, key=lambda r: -r[1] if rev else r[1]), seeds

def summarize(results, print_fn=print):
    out = {'tables': {}, 'hypotheses': [], 'r2': {}, 'checks': {}}
    for cfg in sorted(results):
        bad = [s for s in results[cfg] if 'error' in results[cfg][s]]
        if bad: print_fn(f'\nConfig {cfg}: runs that failed and are excluded: {bad}')
        for metric in ('acc', 'acc_old', 'acc_recent', 'nll'):
            rows, seeds = table(results, cfg, metric)
            out['tables'][f'{cfg}/{metric}'] = rows
            print_fn(f'\nConfig {cfg}, {metric}  (mean ± SE over seeds {seeds})')
            for a, m, s in rows: print_fn(f'   {a:28s} {m:8.3f} ± {s:.3f}')
        seeds = sorted(s for s in results[cfg] if 'error' not in results[cfg][s])
        viol = sum(sum(results[cfg][s]['mixture_bound_violations'].values()) for s in seeds)
        out['checks'][f'{cfg}/mixture_bound_violations'] = int(viol)
        print_fn(f'\nConfig {cfg}: ε-mixture bound violations on test items (must be 0): {viol}')
        print_fn(f'Config {cfg}: PoE weight on random-chart expert, per arm: ' +
                 str({a: round(float(np.mean([results[cfg][s]['poe_w'][a] for s in seeds])), 2) for a in results[cfg][seeds[0]]['poe_w']}))
        mn = [results[cfg][s]['matched_n'] for s in seeds if results[cfg][s]['matched_n'] is not None]
        if mn: print_fn(f'Config {cfg}: matched bridge items per class (mean): {np.mean(mn):.2f}')
        keys = list(results[cfg][seeds[0]]['r2'])
        if keys:
            out['r2'][cfg] = {k: (float(np.mean([results[cfg][s]['r2'][k] for s in seeds])),
                                  float(np.std([results[cfg][s]['r2'][k] for s in seeds], ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else float('nan'))
                              for k in keys}
            print_fn(f'Config {cfg}, within-class R² on held-out items:')
            for k, (m, s) in out['r2'][cfg].items(): print_fn(f'   {k:16s} {m:.3f} ± {s:.3f}')
    print_fn('\nPre-registered comparisons (test accuracy). Significant = |mean| > 2·SE and all seeds same sign. Not significant ≠ tie.')
    for name, (ca, aa), (cb, ab) in HYPOTHESES:
        if ca not in results or cb not in results: continue
        seeds = sorted(s for s in set(results[ca]) & set(results[cb]) if 'error' not in results[ca][s] and 'error' not in results[cb][s])
        if not seeds or aa not in results[ca][seeds[0]]['acc'] or ab not in results[cb][seeds[0]]['acc']: continue
        r = paired([results[ca][s]['acc'][aa] for s in seeds], [results[cb][s]['acc'][ab] for s in seeds])
        verdict = ('SUPPORTED' if r['mean'] > 0 else 'REVERSED') if r['sig'] else 'NOT SIGNIFICANT (not a tie)'
        out['hypotheses'].append(dict(name=name, a=f'{ca}/{aa}', b=f'{cb}/{ab}', result=r, verdict=verdict))
        print_fn(f'  {name}\n     {ca}/{aa} − {cb}/{ab}: {r["mean"]:+.2f} ± {r["se"]:.2f} ({r["n_pos"]}/{r["n"]} seeds positive) -> {verdict}')
    return out
