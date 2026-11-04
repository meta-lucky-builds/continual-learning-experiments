"""Summarise X3 (x3_plan.md readings R1-R3). Star = |mean| > 2*SE over seeds and 5/5 signs."""
import json, numpy as np
R = [json.load(open(f'results/x3/x3_{s}.json'))['rows'] for s in range(5)]
def per_seed(f, us=range(1, 20)): return np.array([np.mean([f(r) for r in rows if r['u'] in us]) for rows in R])
def fmt(x, pct=True):
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); sc = 100 if pct else 1
    star = ' *' if abs(m) > 2 * se and (np.all(x > 0) or np.all(x < 0)) else ''
    return f'{sc*m:+.2f} ± {sc*se:.2f}{star}' if pct else f'{m:.3f} ± {se:.3f}{star}'
out = ['X3 summary (5 seeds; 5-way within-task accuracy on test items of task u, u = 1..19)', '']
out.append('Arm means (%):')
for a in ['OLD', 'TRANS', 'REACT_1', 'REACT_5', 'REACT_all', 'BUF_1', 'BUF_5', 'BUF_all']:
    x = per_seed(lambda r: r[a]); out.append(f'  {a:10s} {100*x.mean():.2f} ± {100*x.std(ddof=1)/np.sqrt(5):.2f}')
out += ['', 'R1  REACT_k - BUF_k:']
for k in ['1', '5', 'all']: out.append(f'  k={k:3s} {fmt(per_seed(lambda r: r["REACT_"+k] - r["BUF_"+k]))}')
out += ['', 'R2  REACT_all / OLD (ratio of per-seed means), and REACT_all - OLD:']
for name, us in [('all u', range(1, 20)), ('u 1-6 (oldest)', range(1, 7)), ('u 7-12', range(7, 13)), ('u 13-19 (newest)', range(13, 20))]:
    ra = per_seed(lambda r: r['REACT_all'], us) / per_seed(lambda r: r['OLD'], us)
    out.append(f'  {name:18s} ratio {fmt(ra - 1, False)} (shown as ratio-1);  diff {fmt(per_seed(lambda r: r["REACT_all"] - r["OLD"], us))}')
for k in ['1', '5']:
    out.append(f'  REACT_{k} - OLD (all u): {fmt(per_seed(lambda r: r["REACT_"+k] - r["OLD"]))}')
out += ['', 'R3  REACT_all - TRANS:', f'  {fmt(per_seed(lambda r: r["REACT_all"] - r["TRANS"]))}']
for k in ['1', '5']: out.append(f'  REACT_{k} - TRANS: {fmt(per_seed(lambda r: r["REACT_"+k] - r["TRANS"]))}')
out += ['', 'Exploratory: TRANS - OLD by tercile']
for name, us in [('u 1-6', range(1, 7)), ('u 7-12', range(7, 13)), ('u 13-19', range(13, 20))]:
    out.append(f'  {name:8s} {fmt(per_seed(lambda r: r["TRANS"] - r["OLD"], us))}   REACT_1-OLD {fmt(per_seed(lambda r: r["REACT_1"] - r["OLD"], us))}')
open('results/x3/summary.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
