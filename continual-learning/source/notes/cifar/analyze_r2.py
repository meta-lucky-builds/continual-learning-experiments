"""Round 2 analysis (plan.md, Round 2). Usage: python analyze_r2.py   (RESV=20). Missing runs are skipped."""
import sys, os, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from analyze_e2 import per_seed, restrict
from analyze_cifar import softmax, acc, paired
E2, R2 = 'results/e2', 'results/r2'; S = range(5)
have = lambda p: all(os.path.exists(p.format(s=s)) for s in S)
def table(name, er_t, rc_t):
    if not (have(er_t) and have(rc_t)): print('skip', name); return None
    res = [per_seed(None, s, yte, er_t.format(s=s), rc_t.format(s=s)) for s in S]; return [r['acc'][C.NTASK - 1] for r in res]
def show(label, F, keys=('ER+bc_buf', 'RC-LDA', 'MIX_half', 'POE_half', 'MIX_what_ahat')):
    print(label, {k: f"{100 * np.mean([f[k] for f in F]):.1f}±{100 * np.std([f[k] for f in F], ddof=1) / math.sqrt(5):.1f}" for k in keys})
def cmp(Fa, ka, Fb, kb, tag):
    rows = [{'k': {'a': a[ka], 'b': b[kb]}} for a, b in zip(Fa, Fb)]; print(tag, paired(rows, 'a', 'b', 'k').replace('a - b', f'{ka}[A] - {kb}[B]'))
_, _, _, yte = C.load(); assert C.RESV == 20
def cfg_check(name, want):
    # run settings live only in the printed JSON line of each run's log; verify them (reviewer finding 8)
    for s in S:
        p = f'{R2}/{name}_{s}.log'
        if os.path.exists(p):
            d = json.loads([l for l in open(p) if l.startswith('{')][-1]); assert d['cfg'] == want, (name, s, d['cfg'], want)
for nm, want in [('er_m680', {'mem': 680}), ('er_m1200', {'mem': 1200}), ('erp', {'promote': 1}), ('m2c', {'m2c': 1}), ('m2c2', {'m2c': 2}), ('er_e3', {'epochs': 3})]: cfg_check(nm, want)

base = table('base', E2 + '/er_lr0.03_{s}.npz', E2 + '/rc_{s}.npz'); show('E2 base (ER500, RC):', base)
# Step 2
m680 = table('m680', R2 + '/er_m680_{s}.npz', E2 + '/rc_{s}.npz'); m1200 = table('m1200', R2 + '/er_m1200_{s}.npz', E2 + '/rc_{s}.npz')
rcw = table('rcw', E2 + '/er_lr0.03_{s}.npz', R2 + '/rcw_{s}.npz')
if m680: show('ER M=680:', m680); cmp(base, 'MIX_half', m680, 'ER+bc_buf', 'R1')
if rcw: show('ER500 + RCW:', rcw); cmp(rcw, 'RC-LDA', base, 'RC-LDA', 'R3')
if m1200 and rcw: show('ER M=1200:', m1200); cmp(rcw, 'MIX_half', m1200, 'ER+bc_buf', 'R2'); cmp(rcw, 'MIX_what_ahat', m1200, 'ER+bc_buf', 'R2 (streaming-valid mix)')
# Step 3: promotion
if have(R2 + '/erp_{s}.npz'):
    Fp = table('erp', R2 + '/erp_{s}.npz', E2 + '/rc_{s}.npz'); show('erp ER path with RC (streaming-valid arms):', Fp)
    rows = []
    for s in S:
        p = np.load(f'{R2}/erp_{s}.npz'); rc = np.load(f'{E2}/rc_{s}.npz'); tasks = p['tasks'].tolist(); seen = list(range(C.K)); t = C.NTASK - 1
        PA = restrict(softmax(p[f'e{t}_test_logits'].astype(np.float64) - p[f'e{t}_b_buf']), seen); PB = np.exp(rc[f'e{t}_test_lda_lp'].astype(np.float64))
        V = [np.exp(p[f'e{t}_ver{v}_test_lp'].astype(np.float64)) for v in range(C.NTASK - 1)]
        r = {'ER+bc_buf': acc(PA, yte, seen), 'RC-LDA': acc(PB, yte, seen), 'P_last': acc(V[-1], yte, seen), 'MIX_half': acc(0.5 * PA + 0.5 * PB, yte, seen),
             'MIX3': acc((PA + PB + V[-1]) / 3, yte, seen), 'BANK': acc((PA + PB + sum(V)) / (2 + len(V)), yte, seen)}
        for v in range(len(V)): r[f'P{v}'] = acc(V[v], yte, seen)
        rows.append(r)
    show('promotion:', rows, tuple(rows[0])); cmp(rows, 'P_last', rows, 'RC-LDA', 'S1'); cmp(rows, 'MIX3', rows, 'MIX_half', 'S2'); cmp(rows, 'BANK', rows, 'MIX_half', 'S3')
# Step 4: protection
for nm in ['ewc', 'm2c', 'm2c2']:
    F = table(nm, R2 + '/' + nm + '_{s}.npz', E2 + '/rc_{s}.npz')
    if F: cfg_check(nm, {'ewc': float(json.load(open(f'{R2}/ewc_lambda.json'))['lam'])}) if nm == 'ewc' else None; show(nm + ':', F); cmp(F, 'ER+bc_buf', base, 'ER+bc_buf', f'U {nm}+bc vs ER+bc'); cmp(F, 'MIX_half', base, 'MIX_half', f'U MIX({nm},RC) vs MIX(ER,RC)')
# Step 5: epochs
F = table('e3', R2 + '/er_e3_{s}.npz', E2 + '/rc_{s}.npz')
if F:
    show('3 epochs:', F)
    for f in F: f['best_single'] = max(f['ER+bc_buf'], f['RC-LDA'])
    cmp(F, 'ER+bc_buf', F, 'RC-LDA', 'V1'); cmp(F, 'MIX_half', F, 'best_single', 'V2'); cmp(F, 'ER+bc_buf', base, 'ER+bc_buf', 'V3')
