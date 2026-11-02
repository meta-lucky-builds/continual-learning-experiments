"""Follow-up diagnostics on saved E1 outputs (no new training): where does MIX_half's gain come from?"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import cifar as C
from analyze_cifar import softmax
R = 'results/e1'; _, _, _, yte = C.load(); y = yte.numpy(); out = []
for s in range(5):
    er = np.load(f'{R}/er_lr0.03_{s}.npz'); rc = np.load(f'{R}/rc_{s}.npz'); tasks = er['tasks'].tolist()
    L = er['e4_test_logits'].astype(np.float64) - er['e4_b_buf']; PA = softmax(L); PB = np.exp(rc['e4_test_lda_lp'].astype(np.float64))
    pa, pb = PA.argmax(1), PB.argmax(1); r = {'seed': s}
    r['acc_A'], r['acc_B'] = float((pa == y).mean()), float((pb == y).mean())
    r['both_right'] = float(((pa == y) & (pb == y)).mean()); r['only_A'] = float(((pa == y) & (pb != y)).mean()); r['only_B'] = float(((pa != y) & (pb == y)).mean())
    r['oracle_either'] = float(((pa == y) | (pb == y)).mean())
    r['w_sweep'] = {f'{w:.1f}': float(((w * PA + (1 - w) * PB).argmax(1) == y).mean()) for w in np.linspace(0, 1, 11)}
    r['logit_avg'] = float((0.5 * np.log(PA + 1e-300) + 0.5 * np.log(PB + 1e-300)).argmax(1).__eq__(y).mean())
    task_of = np.zeros(10, int)
    for t, cl in enumerate(tasks):
        for c in cl: task_of[c] = t
    M = 0.5 * PA + 0.5 * PB
    r['by_task'] = {t: {k: float((P.argmax(1) == y)[task_of[y] == t].mean()) for k, P in [('ERbc', PA), ('RC', PB), ('MIX', M)]} for t in range(5)}
    r['conf_A'] = float(PA.max(1).mean()); r['conf_B'] = float(PB.max(1).mean())
    r['pred_newest_frac'] = {k: float(np.isin(P.argmax(1), tasks[4]).mean()) for k, P in [('ERbc', PA), ('RC', PB), ('MIX', M)]}
    out.append(r)
avg = lambda f: round(100 * float(np.mean([f(r) for r in out])), 1)
print('acc A(ERbc) B(RC):', avg(lambda r: r['acc_A']), avg(lambda r: r['acc_B']), ' both', avg(lambda r: r['both_right']), 'onlyA', avg(lambda r: r['only_A']), 'onlyB', avg(lambda r: r['only_B']), 'either(oracle)', avg(lambda r: r['oracle_either']))
print('w sweep (w on ERbc):', {w: avg(lambda r: r['w_sweep'][w]) for w in out[0]['w_sweep']})
print('per-seed best w:', [max(r['w_sweep'], key=r['w_sweep'].get) for r in out], ' logit-avg (product of experts):', avg(lambda r: r['logit_avg']))
print('by task of origin (final):'); [print('  task', t, {k: avg(lambda r: r['by_task'][t][k]) for k in ['ERbc', 'RC', 'MIX']}) for t in range(5)]
print('mean max-prob A, B:', avg(lambda r: r['conf_A']), avg(lambda r: r['conf_B']), ' frac predicted newest task (true 20%):', {k: avg(lambda r: r['pred_newest_frac'][k]) for k in ['ERbc', 'RC', 'MIX']})
an = json.load(open(f'{R}/analysis.json'))
print('consistency E1 RC-LDA final vs E0 init all-data LDA (seeds 0-2):', [(round(100 * a['acc']['4']['RC-LDA'], 1)) for a in an[:3]], [round(100 * json.loads(open(f'results/e0/e0_{s}.log').readline())['lda']['init']['all'], 1) for s in range(3)])
print('consistency E1 RP-LDA vs E0 rp512:', [(round(100 * a['acc']['4']['RP-LDA'], 1)) for a in an[:3]], [round(100 * json.loads(open(f'results/e0/e0_{s}.log').readline())['lda']['rp512']['all'], 1) for s in range(3)])
json.dump(out, open(f'{R}/fu_mix.json', 'w'), indent=0)
