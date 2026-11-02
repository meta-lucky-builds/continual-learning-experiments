"""Analysis of the sleep tests (sleep_plan.md). Usage: RESV=20 python analyze_sleep.py   (run from notes/cifar)."""
import sys, os, math, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from analyze_e2 import per_seed, restrict
from analyze_cifar import softmax, acc, paired
E2, R2, SL = 'results/e2', 'results/r2', 'results/sleep'; S = range(5); T = C.NTASK - 1
assert C.RESV == 20
_, ytr, _, yte = C.load()
choice = json.load(open('results/sleep_tune/choice.json'))
ARMS = {'slp_r16_buf': dict(sleep=1, rank=16, dream='buf'), 'slp_r4_buf': dict(sleep=1, rank=4, dream='buf'), 'slp_full_buf': dict(sleep=1, rank=0, dream='buf'),
        'slp_full_mix': dict(sleep=1, rank=0, dream='mix'), 'slp_full_noise': dict(sleep=1, rank=0, dream='noise'), 'erw': dict(width=64)}
def log_of(name, s): return json.loads([l for l in open(f'{SL}/{name}_{s}.log') if l.startswith('{')][-1])
def cmp(va, vb, tag, la='A', lb='B'):
    rows = [{'k': {'a': a, 'b': b}} for a, b in zip(va, vb)]; print(f'{tag:6s}', paired(rows, 'a', 'b', 'k').replace('a - b', f'{la} - {lb}'))
def ms(v): return f'{100 * np.mean(v):.1f} ± {100 * np.std(v, ddof=1) / math.sqrt(len(v)):.1f}'

# ---- gates: configs, and every arm identical to E2's ER through the end of task 2 (sleep arms) / same stream (erw)
for nm, want in ARMS.items():
    for s in S:
        cfg = log_of(nm, s)['cfg']; w = dict(want)
        if nm != 'erw': w.update(dlr=choice['dlr'], dep=choice['dep'])
        assert cfg == w, (nm, s, cfg, w)
er = {s: np.load(f'{E2}/er_lr0.03_{s}.npz') for s in S}; rc = {s: np.load(f'{E2}/rc_{s}.npz') for s in S}
for nm in ARMS:
    for s in S:
        z = np.load(f'{SL}/{nm}_{s}.npz'); assert np.array_equal(z['ylearn'], er[s]['ylearn']) and np.array_equal(z['resv_y'], er[s]['resv_y'])
        if nm == 'erw': continue
        n12 = int(sum(len(bi) for t, bi in C.batches(*C.make_stream(ytr, s)[1:]) if t < 2))
        assert np.array_equal(z['preq'][:n12], er[s]['preq'][:n12]) and np.array_equal(z['preq_lp'][:n12], er[s]['preq_lp'][:n12]), (nm, s, 'preq through task 2')
        assert np.array_equal(z['e0_test_logits'], er[s]['e0_test_logits']), (nm, s, 'task 1')
        assert np.array_equal(z['e1_T_test_logits'], er[s]['e1_test_logits']) and np.array_equal(z['e1_T_b_buf'], er[s]['e1_b_buf']), (nm, s, 'teacher at sleep 1')
print('GATES PASS: configs as declared; every sleep arm bit-identical to E2 ER through task 2 (stream predictions, task-1 logits, the teacher at the first sleep)')

# ---- final accuracies
F = {nm: [per_seed(None, s, yte, f'{SL}/{nm}_{s}.npz', f'{E2}/rc_{s}.npz')['acc'][T] for s in S] for nm in ARMS}
base = [per_seed(None, s, yte, f'{E2}/er_lr0.03_{s}.npz', f'{E2}/rc_{s}.npz')['acc'][T] for s in S]
print('\nFinal accuracy (%), mean ± SE over 5 seeds')
print(f"{'arm':16s} {'raw':>12s} {'+bc_buf':>12s} {'+bc(a-hat)':>12s} {'MIX_half(.,RC)':>15s} {'MIX(w,a)':>12s}")
for nm, rows in [('ER (E2)', base)] + list(F.items()):
    print(f"{nm:16s} {ms([r['ER'] for r in rows]):>12s} {ms([r['ER+bc_buf'] for r in rows]):>12s} {ms([r['ER+bc_ahat'] for r in rows]):>12s} {ms([r['MIX_half'] for r in rows]):>15s} {ms([r['MIX_what_ahat'] for r in rows]):>12s}")
# teacher at the final sleep (what the run would have without the last consolidation)
for nm in ARMS:
    if nm == 'erw': continue
    tv = []
    for s in S:
        z = np.load(f'{SL}/{nm}_{s}.npz'); tv.append(acc(z[f'e{T}_T_test_logits'].astype(np.float64) - z[f'e{T}_T_b_buf'], yte, list(range(C.K))))
    print(f'  {nm}: teacher before the final sleep, +bc_buf: {ms(tv)}')

# ---- BANK (round 2 promotion) per seed, for T1c
bank = []
for s in S:
    p = np.load(f'{R2}/erp_{s}.npz'); seen = list(range(C.K))
    PA = restrict(softmax(p[f'e{T}_test_logits'].astype(np.float64) - p[f'e{T}_b_buf']), seen); PB = np.exp(rc[s][f'e{T}_test_lda_lp'].astype(np.float64))
    V = [np.exp(p[f'e{T}_ver{v}_test_lp'].astype(np.float64)) for v in range(C.NTASK - 1)]
    bank.append(acc((PA + PB + sum(V)) / (2 + len(V)), yte, seen))

print('\nPre-declared comparisons')
g = lambda nm, k: [r[k] for r in F[nm]]
cmp(g('slp_r16_buf', 'ER+bc_buf'), [r['ER+bc_buf'] for r in base], 'T1', 'slp_r16_buf+bc', 'ER+bc')
cmp(g('slp_full_buf', 'ER+bc_buf'), [r['ER+bc_buf'] for r in base], 'T1b', 'slp_full_buf+bc', 'ER+bc')
cmp(g('slp_r16_buf', 'MIX_half'), bank, 'T1c', 'MIX_half(slp_r16,RC)', 'BANK')

# ---- per-sleep statistics (JSON logs) and sleep cost from saved logits
print('\nPer-sleep statistics, mean over the 4 sleeps, then mean ± SE over seeds')
st = {}
for nm in [a for a in ARMS if a != 'erw']:
    rows = []
    for s in S:
        L = log_of(nm, s)['sleeps']; z = np.load(f'{SL}/{nm}_{s}.npz'); tasks = z['tasks'].tolist(); cost = []
        for sl in L:
            t = sl['t']; seen = C.seen_of(tasks, t)
            stu = acc(z[f'e{t}_test_logits'].astype(np.float64) - z[f'e{t}_b_buf'], yte, seen)
            tea = acc(z[f'e{t}_T_test_logits'].astype(np.float64) - z[f'e{t}_T_b_buf'], yte, seen); cost.append(stu - tea)
        rows.append({'gap': np.mean([x['agree_buf'] - x['agree_resv'] for x in L]), 'agree_buf': np.mean([x['agree_buf'] for x in L]),
                     'agree_resv': np.mean([x['agree_resv'] for x in L]), 'agree_test': np.mean([x['agree_test'] for x in L]),
                     'kl_buf': np.mean([x['kl_buf'] for x in L]), 'kl_resv': np.mean([x['kl_resv'] for x in L]), 'cost': np.mean(cost), 'cost_each': cost,
                     'stu_imgs': sum(x['student_fwdbwd_imgs'] for x in L), 'tea_imgs': sum(x['teacher_fwd_imgs'] for x in L),
                     'sleep_secs': sum(x['sleep_secs'] for x in L), 'er_imgs': log_of(nm, s)['fwdbwd_imgs'], 'ranks': L[0]['expert_ranks'], 'eparams': L[0]['expert_params']})
    st[nm] = rows
    v = lambda k: [r[k] for r in rows]
    print(f"{nm:15s} agree buf {ms(v('agree_buf'))}  resv {ms(v('agree_resv'))}  test {ms(v('agree_test'))}  gap {ms(v('gap'))} (signs +{sum(x > 0 for x in v('gap'))}/5)"
          f"  KL buf {np.mean(v('kl_buf')):.3f} resv {np.mean(v('kl_resv')):.3f}  sleep cost {ms(v('cost'))}")
    print(f"{'':15s} expert params {rows[0]['eparams']} ranks {rows[0]['ranks']}; compute: ER imgs {rows[0]['er_imgs']}, + student fwd/bwd {rows[0]['stu_imgs']}, + teacher fwd {rows[0]['tea_imgs']}; sleep time {np.mean(v('sleep_secs')):.0f} s")
gaps = [r['gap'] for r in st['slp_r16_buf']]
print('T2 ', 'slp_r16_buf held-out gap', ms(gaps), 'LOSSY (pre-declared: mean > 5 points, 5/5 positive)' if np.mean(gaps) > 0.05 and all(x > 0 for x in gaps) else 'not lossy by the pre-declared rule')
for nm in st: print('T2 ', nm, 'gap', ms([r['gap'] for r in st[nm]]), 'lossy' if np.mean([r['gap'] for r in st[nm]]) > 0.05 and all(r['gap'] > 0 for r in st[nm]) else 'not lossy')
cmp([r['cost'] for r in st['slp_r4_buf']], [r['cost'] for r in st['slp_r16_buf']], 'T2b', 'cost r4', 'cost r16')
cmp([r['cost'] for r in st['slp_r16_buf']], [r['cost'] for r in st['slp_full_buf']], 'T2b', 'cost r16', 'cost full')
for nm in st:
    c = [r['cost'] for r in st[nm]]; print(f"T2b {nm} sleep cost {ms(c)} (signs -{sum(x < 0 for x in c)}/5)")
cmp([r['agree_resv'] for r in st['slp_full_mix']], [r['agree_resv'] for r in st['slp_full_buf']], 'T3', 'agree_resv mix', 'buf')
cmp([r['agree_resv'] for r in st['slp_full_noise']], [r['agree_resv'] for r in st['slp_full_buf']], 'T3', 'agree_resv noise', 'buf')
cmp(g('slp_full_mix', 'ER+bc_buf'), g('slp_full_buf', 'ER+bc_buf'), 'T3', 'final mix+bc', 'buf+bc')
cmp(g('slp_full_noise', 'ER+bc_buf'), g('slp_full_buf', 'ER+bc_buf'), 'T3', 'final noise+bc', 'buf+bc')
cmp([r['ER+bc_buf'] for r in base], g('erw', 'ER'), 'T4', 'ER+bc', 'erw raw')
d4 = np.array([r['ER+bc_buf'] for r in base]) - np.array(g('erw', 'ER')); print('T4 ', 'pre-declared: ER+bc > erw raw by more than 5 points:', bool(d4.mean() > 0.05 and (d4 > 0).all()), f'(mean {100 * d4.mean():.1f})')
cmp(g('erw', 'ER+bc_buf'), [r['ER+bc_buf'] for r in base], 'T4b', 'erw+bc', 'ER+bc')
print('     raw: ER', ms([r['ER'] for r in base]), ' erw', ms(g('erw', 'ER')), '; capacity gain raw', ms(np.array(g('erw', 'ER')) - np.array([r['ER'] for r in base])),
      '; bc gain at ER width', ms(np.array([r['ER+bc_buf'] for r in base]) - np.array([r['ER'] for r in base])))
ew = [log_of('erw', s) for s in S]; print('     erw compute: fwd/bwd imgs', ew[0]['fwdbwd_imgs'], '(same image count as ER; ~4x FLOPs per image at width 64), secs', [round(x['secs']) for x in ew])
