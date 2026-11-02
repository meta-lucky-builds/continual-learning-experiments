# ===== Summary and pre-registered hypothesis tests =====
import numpy as np
from atlas_lib import paired

HYPOTHESES = [  # (name, (config, arm), (config, arm), what a positive significant difference means)
    ('H1 multi-aux GREG (reserve bridge) beats X2-style CH_cv (buffer bridge)', ('A', 'GR_multi_res'), ('A', 'CH_cv_buf')),
    ('H2 reserve bridge beats buffer bridge, same estimator',                    ('A', 'GR_multi_res'), ('A', 'GR_multi_buf')),
    ('H3 per-class GREG with [r,f_prev] beats CH_cv, same reserve bridge',       ('A', 'GR_multi_res'), ('A', 'CH_cv_res')),
    ('H4 equal bytes: B multi-aux (M=1000+1000 reserve) beats A CH_cv_buf (M=2000)', ('B', 'GR_multi_res'), ('A', 'CH_cv_buf')),
    ('S1 replication of X2 direction: CH_cv_buf beats buffer-only',              ('A', 'CH_cv_buf'), ('A', 'B_only_buf')),
    ('S2 stable chart helps at all: GR_stable_res beats reserve-only means',     ('A', 'GR_stable_res'), ('A', 'B_only_res')),
]

def table(results, cfg, metric='acc'):
    runs = results[cfg]; seeds = sorted(runs)
    arms = list(runs[seeds[0]][metric].keys())
    rows = []
    for a in arms:
        v = np.array([runs[s][metric][a] for s in seeds])
        rows.append((a, v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')))
    return sorted(rows, key=lambda r: -r[1] if metric.startswith('acc') else r[1]), seeds

def summarize(results, print_fn=print):
    out = {'tables': {}, 'hypotheses': [], 'r2': {}}
    for cfg in sorted(results):
        for metric in ('acc', 'acc_oracleSigma', 'err_all', 'err_old'):
            rows, seeds = table(results, cfg, metric)
            out['tables'][f'{cfg}/{metric}'] = [(a, float(m), float(s)) for a, m, s in rows]
            if metric in ('acc', 'err_all'):
                print_fn(f'\nConfig {cfg}, {metric}  (mean ± SE over seeds {seeds})')
                for a, m, s in rows: print_fn(f'   {a:16s} {m:8.3f} ± {s:.3f}')
        seeds = sorted(results[cfg]); keys = [k for k in results[cfg][seeds[0]]['r2'] if k != 'hops']
        out['r2'][cfg] = {k: (float(np.mean([results[cfg][s]['r2'][k] for s in seeds])),
                              float(np.std([results[cfg][s]['r2'][k] for s in seeds], ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else float('nan'))
                          for k in keys}
        print_fn(f'\nConfig {cfg}, Test 1 within-class R² on reserve items:')
        for k, (m, s) in out['r2'][cfg].items(): print_fn(f'   {k:16s} {m:.3f} ± {s:.3f}')
    print_fn('\nPre-registered comparisons (accuracy, shared Σ). Significant = |mean| > 2·SE and all seeds same sign:')
    for name, (ca, aa), (cb, ab) in HYPOTHESES:
        if ca not in results or cb not in results: continue
        seeds = sorted(set(results[ca]) & set(results[cb]))
        r = paired([results[ca][s]['acc'][aa] for s in seeds], [results[cb][s]['acc'][ab] for s in seeds])
        r2 = paired([results[ca][s]['acc_oracleSigma'][aa] for s in seeds], [results[cb][s]['acc_oracleSigma'][ab] for s in seeds])
        verdict = ('SUPPORTED' if r['mean'] > 0 else 'REVERSED') if r['sig'] else 'NOT SIGNIFICANT (not a tie)'
        out['hypotheses'].append(dict(name=name, a=f'{ca}/{aa}', b=f'{cb}/{ab}', shared_sigma=r, oracle_sigma=r2, verdict=verdict))
        print_fn(f'  {name}\n     diff {r["mean"]:+.2f} ± {r["se"]:.2f} ({r["n_pos"]}/{r["n"]} seeds positive) -> {verdict};'
                 f'  with oracle Σ: {r2["mean"]:+.2f} ± {r2["se"]:.2f}')
    return out
