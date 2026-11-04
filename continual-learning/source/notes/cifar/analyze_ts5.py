"""T-S5 analysis (ts5_plan.md). Run: CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 MEM=2000 RESV=20 python analyze_ts5.py"""
import sys, os, math, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from analyze_e2 import per_seed, restrict
from analyze_cifar import softmax, acc, paired
assert C.K == 100 and C.NTASK == 20 and C.RESV == 20
O = 'results/ts5'; S = range(5); T = C.NTASK - 1; seen = list(range(C.K))
_, ytr, _, yte = C.load()
bud = json.load(open(f'{O}/budget.json')); ch = json.load(open('results/sleep_tune/choice.json'))
def log_of(nm, s): return json.loads([l for l in open(f'{O}/{nm}_{s}.log') if l.startswith('{')][-1])
want = {'erp': {'mem': 2000, 'promote': 1}, 'er_mmatch': {'mem': bud['M_match']}, 'slp': {'sleep': 1, 'rank': 16, 'dream': 'buf', 'dlr': ch['dlr'], 'dep': ch['dep']}}
for nm, w in want.items():
    for s in S: assert log_of(nm, s)['cfg'] == w, (nm, s, log_of(nm, s)['cfg'], w)
for s in S:  # same stream for every arm; the sleep arm equals ER (erp's learner) before its first sleep, task-1 end
    z = {nm: np.load(f'{O}/{nm}_{s}.npz') for nm in ['erp', 'er_mmatch', 'slp', 'rc']}
    assert all(np.array_equal(z[nm]['ylearn'], z['erp']['ylearn']) and np.array_equal(z[nm]['resv_y'], z['erp']['resv_y']) for nm in z)
    n1 = int(sum(len(bi) for t, bi in C.batches(*C.make_stream(ytr, s)[1:]) if t < 2))
    assert np.array_equal(z['slp']['preq'][:n1], z['erp']['preq'][:n1]) and np.array_equal(z['slp']['e1_T_test_logits'], z['erp']['e1_test_logits'])
print('GATES PASS: configs as declared; same stream in every arm; sleep arm bit-identical to ER through task 2')

def cmp(va, vb, tag, la, lb):
    rows = [{'k': {'a': a, 'b': b}} for a, b in zip(va, vb)]; print(f'{tag:4s}', paired(rows, 'a', 'b', 'k').replace('a - b', f'{la} - {lb}'))
def ms(v): return f'{100 * np.mean(v):.1f} ± {100 * np.std(v, ddof=1) / math.sqrt(len(v)):.1f}'
R = {}
for s in S:
    p = np.load(f'{O}/erp_{s}.npz'); rc = np.load(f'{O}/rc_{s}.npz')
    PA = restrict(softmax(p[f'e{T}_test_logits'].astype(np.float64) - p[f'e{T}_b_buf']), seen); PB = np.exp(rc[f'e{T}_test_lda_lp'].astype(np.float64))
    V = [np.exp(p[f'e{T}_ver{v}_test_lp'].astype(np.float64)) for v in range(C.NTASK - 1)]
    bank = lambda vs: acc((PA + PB + sum(V[v] for v in vs)) / (2 + len(vs)), yte, seen)
    mm = per_seed(None, s, yte, f'{O}/er_mmatch_{s}.npz', f'{O}/rc_{s}.npz')['acc'][T]; er = per_seed(None, s, yte, f'{O}/erp_{s}.npz', f'{O}/rc_{s}.npz')['acc'][T]
    sl = per_seed(None, s, yte, f'{O}/slp_{s}.npz', f'{O}/rc_{s}.npz')['acc'][T]
    R[s] = {'ER': er['ER'], 'ER+bc': er['ER+bc_buf'], 'RC': er['RC-LDA'], 'MIX_half': er['MIX_half'], 'BANK_all': bank(range(19)), 'BANK_4last': bank([15, 16, 17, 18]),
            'BANK_4spread': bank([3, 8, 13, 18]), 'ERmm+bc': mm['ER+bc_buf'], 'MIX_half(ERmm,RC)': mm['MIX_half'], 'SLP+bc': sl['ER+bc_buf'], 'MIX_half(SLP,RC)': sl['MIX_half'],
            'curve_recent': [bank(list(range(19 - k, 19))) for k in range(20)], 'versions_alone': [acc(V[v], yte, seen) for v in range(19)]}
    tv = np.load(f'{O}/slp_{s}.npz'); R[s]['SLP_teacher_final+bc'] = acc(tv[f'e{T}_T_test_logits'].astype(np.float64) - tv[f'e{T}_T_b_buf'], yte, seen)
g = lambda k: [R[s][k] for s in S]
print('\nFinal accuracy over 100 classes (%), mean ± SE')
for k in ['ER', 'ER+bc', 'RC', 'MIX_half', 'BANK_all', 'BANK_4last', 'BANK_4spread', 'ERmm+bc', 'MIX_half(ERmm,RC)', 'SLP+bc', 'SLP_teacher_final+bc', 'MIX_half(SLP,RC)']: print(f'  {k:22s} {ms(g(k))}')
print('\nPre-declared comparisons')
cmp(g('BANK_all'), g('MIX_half'), 'G1', 'BANK_all', 'MIX_half')
cmp(g('BANK_4last'), g('BANK_all'), 'G2', 'BANK_4last', 'BANK_all')
cmp(g('BANK_4spread'), g('BANK_all'), 'G3', 'BANK_4spread', 'BANK_all')
gain = np.array(g('BANK_all')) - np.array(g('MIX_half'))
for k in ['BANK_4last', 'BANK_4spread']: print(f'     {k} keeps {100 * np.mean((np.array(g(k)) - np.array(g("MIX_half")))) / max(1e-9, 100 * gain.mean()):.0%} of the G1 gain')
cmp(g('BANK_all'), g('MIX_half(ERmm,RC)'), 'G4', 'BANK_all', 'MIX_half(ER M=%d, RC)' % bud['M_match'])
cmp(g('BANK_all'), g('ERmm+bc'), 'G4b', 'BANK_all', 'ER+bc(M=%d)' % bud['M_match'])
cmp(g('SLP+bc'), g('ER+bc'), 'G5', 'SLP+bc', 'ER+bc')
cur = np.array(g('curve_recent')); mb = bud['bytes_per_version'] / 1e6
print('\nAccuracy vs number of versions kept (most recent first): k, extra MB, mean acc')
for k in [0, 1, 2, 4, 8, 12, 19]: print(f'  k={k:2d}  +{k * mb:5.1f} MB  {100 * cur[:, k].mean():.1f}')
print('Versions alone (LDA on each frozen encoder), mean over seeds:', [round(float(100 * x), 1) for x in np.mean(g('versions_alone'), 0)])
print('Compute (secs):', {nm: [round(log_of(nm, s)['secs']) for s in S] for nm in ['erp', 'er_mmatch', 'slp']})
