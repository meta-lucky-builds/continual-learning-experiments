"""A1 + A2 analysis (audit_plan.md). Usage: CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 MEM=2000 RESV=20 python3 analyze_audit100.py"""
import sys, os, math, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from analyze_e2 import per_seed, restrict
from analyze_cifar import softmax, acc, paired
assert C.K == 100 and C.NTASK == 20 and C.RESV == 20
O, P = 'results/ts5', 'results/ts5_probe'; S = range(5); T = C.NTASK - 1; seen = list(range(C.K))
_, ytr, _, yte = C.load()
def log_of(p): return json.loads([l for l in open(p) if l.startswith('{')][-1])
for s in S:  # g1
    z = np.load(f'{O}/erp_{s}.npz'); sl = np.load(f'{O}/slp_{s}.npz'); rc = np.load(f'{O}/rc_{s}.npz')
    n1 = int(sum(len(bi) for t, bi in C.batches(*C.make_stream(ytr, s)[1:]) if t < 2))
    assert np.array_equal(z['ylearn'], rc['ylearn']) and np.array_equal(z['preq'][:n1], sl['preq'][:n1]) and np.array_equal(z['e1_test_logits'], sl['e1_T_test_logits'])
    assert log_of(f'{O}/erp_{s}.log')['cfg'] == {'mem': 2000, 'promote': 1} and log_of(f'{O}/er_m3439_{s}.log')['cfg'] == {'mem': 3439}
print('g1 PASS: regenerated erp equals the T-S5 sleep arm through task 2 and shares the rc stream; configs as declared')
def cmp(va, vb, tag, la, lb):
    rows = [{'k': {'a': a, 'b': b}} for a, b in zip(va, vb)]; print(f'{tag:4s}', paired(rows, 'a', 'b', 'k').replace('a - b', f'{la} - {lb}'))
def ms(v): return f'{100 * np.mean(v):.1f} ± {100 * np.std(v, ddof=1) / math.sqrt(len(v)):.1f}'
# ---- A1
pr = [{e['name']: e for e in json.load(open(f'{P}/probe_{s}.json'))['enc']} for s in S]
names = [f'v{i}' for i in range(19)] + ['final', 'rc']
print('\nA1 probes, mean over 5 seeds (%). online = the T-S5 readout (buffer at promotion + later stream); PF/PS/PO/PO4 = same items for every encoder')
print(f"{'enc':6s} {'online':>7s} {'PF':>6s} {'PS':>6s} {'PO(5)':>6s} {'PO4(20)':>8s}")
for n in names:
    m = lambda k: (f'{100 * np.mean([p[n][k] for p in pr]):6.1f}' if k in pr[0][n] else '     -')
    print(f'{n:6s} {m("online"):>7s} {m("PF")} {m("PS")} {m("PO")} {m("PO4"):>8s}')
g = lambda n, k: [p[n][k] for p in pr]
print('\nPre-declared (v8 = observed online peak, declared as a post-hoc choice)')
cmp(g('v18', 'PF'), g('v8', 'PF'), 'D1', 'PF(v18)', 'PF(v8)')
cmp(g('v18', 'PF'), g('rc', 'PF'), 'D2', 'PF(v18)', 'PF(rc)')
cmp(g('v18', 'PO'), g('v8', 'PO'), 'D3', 'PO(v18)', 'PO(v8)')
cmp(g('v18', 'PS'), g('v8', 'PS'), 'D4', 'PS(v18)', 'PS(v8)')
gap = lambda n: np.array(g(n, 'online')) - np.array(g(n, 'PF'))
print(f'D5   online - PF: v8 {100 * gap("v8").mean():+.1f}, v18 {100 * gap("v18").mean():+.1f}   (online decline v8->v18 {100 * np.mean(np.array(g("v18", "online")) - g("v8", "online")):+.1f}; PF change {100 * np.mean(np.array(g("v18", "PF")) - g("v8", "PF")):+.1f})')
cmp(g('v18', 'PO4'), g('v8', 'PO4'), 'xPO4', 'PO4(v18)', 'PO4(v8)')
cmp(g('final', 'PF'), g('v18', 'PF'), 'x', 'PF(final learner)', 'PF(v18)')
# ---- A2
b4, mix, erbc = [], [], []
for s in S:
    p = np.load(f'{O}/erp_{s}.npz'); rc = np.load(f'{O}/rc_{s}.npz')
    PA = restrict(softmax(p[f'e{T}_test_logits'].astype(np.float64) - p[f'e{T}_b_buf']), seen); PB = np.exp(rc[f'e{T}_test_lda_lp'].astype(np.float64))
    V = [np.exp(p[f'e{T}_ver{v}_test_lp'].astype(np.float64)) for v in [3, 8, 13, 18]]
    b4.append(acc((PA + PB + sum(V)) / 6, yte, seen))
    a = per_seed(None, s, yte, f'{O}/er_m3439_{s}.npz', f'{O}/rc_{s}.npz')['acc'][T]; mix.append(a['MIX_half']); erbc.append(a['ER+bc_buf'])
print(f'\nA2  BANK_4spread {ms(b4)}   MIX_half(ER M=3439, RC) {ms(mix)}   ER+bc(M=3439) {ms(erbc)}')
cmp(b4, mix, 'E1', 'BANK_4spread', 'MIX_half(ER3439,RC)')
cmp(b4, erbc, 'E1b', 'BANK_4spread', 'ER+bc(M=3439)')
