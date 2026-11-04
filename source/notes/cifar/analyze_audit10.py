"""A3 analysis (audit_plan.md): exact-transfer Sleep control vs ER. Usage: RESV=20 python3 analyze_audit10.py"""
import sys, os, math, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from analyze_e2 import per_seed
from analyze_cifar import paired
assert C.RESV == 20 and C.K == 10
E2, A, SL = 'results/e2', 'results/audit', 'results/sleep'; S = range(5); T = C.NTASK - 1
_, ytr, _, yte = C.load()
def log_of(p): return json.loads([l for l in open(p) if l.startswith('{')][-1])
md = []
for s in S:
    lg = log_of(f'{A}/slp_exact_{s}.log'); assert lg['cfg'] == {'sleep': 1, 'rank': 0, 'dream': 'buf'} and lg['resv'] == 20 and lg['mem'] == 500
    md += [max(x['maxdiff'].values()) for x in lg['sleeps']]; assert len(lg['sleeps']) == C.NTASK - 1
    z = np.load(f'{A}/slp_exact_{s}.npz'); er = np.load(f'{E2}/er_lr0.03_{s}.npz')
    n12 = int(sum(len(bi) for t, bi in C.batches(*C.make_stream(ytr, s)[1:]) if t < 2))
    assert np.array_equal(z['ylearn'], er['ylearn']) and np.array_equal(z['preq'][:n12], er['preq'][:n12])
assert max(md) <= 1e-4, max(md)
print(f'GATES PASS: configs; bit-identical to E2 ER through task 2; max |merged - teacher| logit over all sleeps = {max(md):.2e} (<= 1e-4)')
def ms(v): return f'{100 * np.mean(v):.1f} ± {100 * np.std(v, ddof=1) / math.sqrt(len(v)):.1f}'
ex = [per_seed(None, s, yte, f'{A}/slp_exact_{s}.npz', f'{E2}/rc_{s}.npz')['acc'][T] for s in S]
er = [per_seed(None, s, yte, f'{E2}/er_lr0.03_{s}.npz', f'{E2}/rc_{s}.npz')['acc'][T] for s in S]
fb = [per_seed(None, s, yte, f'{SL}/slp_full_buf_{s}.npz', f'{E2}/rc_{s}.npz')['acc'][T] for s in S]
for k in ['ER', 'ER+bc_buf', 'MIX_half']: print(f'{k:10s} exact-transfer {ms([x[k] for x in ex])}   ER {ms([x[k] for x in er])}   full-rank distilled {ms([x[k] for x in fb])}')
rows = lambda a, b, k: [{'k': {'a': x[k], 'b': y[k]}} for x, y in zip(a, b)]
print('H1 ', paired(rows(ex, er, 'ER+bc_buf'), 'a', 'b', 'k').replace('a - b', 'exact+bc - ER+bc'))
print('    ', paired(rows(fb, ex, 'ER+bc_buf'), 'a', 'b', 'k').replace('a - b', 'full-rank distilled+bc - exact+bc'))
