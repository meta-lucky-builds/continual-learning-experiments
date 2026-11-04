"""Summarise marathon v2 results (pre-registered rule: a difference counts only if the paired mean ± SE
excludes 0 AND all 5 seeds share its sign; n = 5 cannot give p < 0.0625 by any rank test).
Paired by seed. Arms that change the replay draws (buffer policy) are compared only with their own baseline."""
import json, os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def load(name):
    p = os.path.join(D, name + '.jsonl')
    if not os.path.exists(p): return None
    v = [json.loads(l) for l in open(p) if l.startswith('{')]
    return sorted(v, key=lambda x: x['seed']) if len(v) == 5 else None
def ms(x): x = np.asarray(x, float); return x.mean(), x.std(ddof=1) / np.sqrt(len(x))
def fmt(x): m, s = ms(x); return f'{100*m:5.1f}±{100*s:3.1f}'
def pdiff(a, b, scale=100):
    d = np.asarray(a, float) - np.asarray(b, float); m, s = ms(d)
    sig = (abs(m) > s) and ((d > 0).all() or (d < 0).all())
    return f'{scale*m:+6.1f}±{scale*s:4.1f} [{(d > 0).sum()}/5 >0]{" *" if sig else ""}'
A = lambda v, k='acc': [r[k] for r in v]

def dipstats(v):
    """T1: per task, min and mean old-class acc over steps 0..100 (step 0 = right after any init); averaged over tasks."""
    mn, au = [], []
    for r in v:
        per = {}
        for ti, st, a in r['curve']:
            if st <= 100: per.setdefault(ti, []).append(a)
        mn.append(np.mean([min(L) for L in per.values()])); au.append(np.mean([np.mean(L) for L in per.values()]))
    return mn, au

for st in ['toyA', 'toyB', 'mnist']:
    print(f'\n================ {st} ================')
    er = load(f'T2_er__{st}')
    if er is None: print('  ER not finished'); continue
    base = A(er)
    print(f"ER: acc {fmt(base)} old {fmt(A(er,'old_acc'))} last {fmt(A(er,'last_acc'))}")
    print('-- T2 readouts on the same ER weights --')
    for k in ['wa', 'biascorr', 'ncm_buf', 'refit_buf', 'refit_full']:
        x = [r['ro'][k] for r in er]; print(f'   {k:10s} {fmt(x)}  vs ER {pdiff(x, base)}' + ('   (oracle: refit on ALL training data)' if k == 'refit_full' else ''))
    print('   WA gamma', np.round([r['ro']['wa_gamma'] for r in er], 3))
    ers = load(f'T2_er_start__{st}')
    for name, ref, lab in [('T2_w9', base, 'ER'), ('F5_center', base, 'ER'), ('F5_cos', base, 'ER'), ('F5_cos_center', base, 'ER'),
                           ('F5_center_w9', base, 'ER'), ('T2_ace_start', A(ers) if ers else None, 'ER-start'),
                           ('F5_center_ace_start', A(ers) if ers else None, 'ER-start')]:
        v = load(f'{name}__{st}')
        if v is None or ref is None: print(f'   {name}: not finished'); continue
        print(f"   {name:20s} {fmt(A(v))} vs {lab:8s} {pdiff(A(v), ref)}  old {fmt(A(v,'old_acc'))} last {fmt(A(v,'last_acc'))}")
    w9, c = load(f'T2_w9__{st}'), load(f'F5_center__{st}')
    if w9 and c: print(f'   center vs w9 {pdiff(A(c), A(w9))}')
    print('-- offset mechanism (misclassified old items predicted as a newest class; conditional on the method\'s own errors) --')
    for name in ['F4_er_mech', 'F6_w9_mech', 'F6_center_ro']:
        v = load(f'{name}__{st}')
        if v is None: continue
        mk = {k: round(float(np.mean([r['mech'][k] for r in v])), 2) for k in ['n_items', 'gap_total', 'gap_common', 'gap_specific', 'gap_bias']}
        print(f'   {name:13s} {mk}')
        if 'ro' in v[0]: print('      readouts on centred model:', {k: round(100 * np.mean([r['ro'][k] for r in v]), 1) for k in ['biascorr', 'ncm_buf', 'refit_buf', 'refit_full']})
    print('-- T3 features vs readout, old classes after each task --')
    for name in ['T3_er', 'T3_ace_start', 'T2_er', 'T2_ace_start', 'F5_center']:
        v = load(f'{name}__{st}')
        if v is None or not v[0].get('probe_curve'): continue
        P = lambda k: np.round(100 * np.mean([[p.get(k, np.nan) for p in r['probe_curve']] for r in v], 0), 1)
        print(f"   {name:13s} model {P('model_old')}  probe_buf {P('probe_buf_old')}  probe_full {P('probe_full_old')}")
    print('-- T1 boundary shock (old-class acc over steps 0..100 of each task: min, mean; final) --')
    t1 = load(f'T1_er__{st}')
    if t1:
        m0, a0 = dipstats(t1); print(f'   ER            min {fmt(m0)} mean {fmt(a0)} final {fmt(A(t1))}  (T1_er acc must equal T2_er acc: {A(t1) == base})')
        for name in ['T1_cmean', 'T1_warm', 'T1_cmean_warm', 'T1_uncentred_mean']:
            v = load(f'{name}__{st}')
            if v is None: print(f'   {name}: not finished'); continue
            m1, a1 = dipstats(v)
            print(f"   {name:17s} min {pdiff(m1, m0)} mean {pdiff(a1, a0)} final {pdiff(A(v), A(t1))} old {pdiff(A(v,'old_acc'), A(t1,'old_acc'))}")
    if st in ('toyA', 'mnist'):
        print('-- T4 Fisher penalty on top of ER: stale (latest task-end, frozen) vs fresh (every 25 steps) vs stale_avg --')
        for lam in [0.1, 1, 10, 100]:
            s, f, sa = (load(f'T4_{k}_l{lam}__{st}') for k in ('stale', 'fresh', 'stale_avg'))
            if s is None or f is None: print(f'   lam {lam}: not finished'); continue
            print(f"   lam {lam:5}: stale {fmt(A(s))} fresh {fmt(A(f))} | fresh-stale {pdiff(A(f), A(s))} | stale-ER {pdiff(A(s), base)} | fresh-ER {pdiff(A(f), base)}"
                  + (f" | stale_avg {fmt(A(sa))}" if sa else ''))
        print('-- T5 recurrence of task 1 (oracle boundary). ttt = steps to 90% of reference (per-step for 50 steps); censored = 305 --')
        for sn in ['none', 'noop', 'full', 'rows', 'rows_wrong', 'rows_init', 'full_wrong']:
            v = load(f'T5_{sn}__{st}')
            if v is None: print(f'   {sn}: not finished'); continue
            R = [r['ret'] for r in v]
            print(f"   {sn:10s} ttt {[x['ttt'] for x in R]} cens {sum(x['censored'] for x in R)} | start {fmt([x['acc_task_start'] for x in R])} auc50 {fmt([x['auc50'] for x in R])} end {fmt([x['acc_task_end'] for x in R])} others_end {fmt([x['acc_others_end'] for x in R])} | ref {fmt([x['ref'] for x in R])}")
        fr, nn = load(f'T5_fresh__{st}'), load(f'T5_none__{st}')
        if fr and nn:
            tt, auc = [], []
            for r, n in zip(fr, nn):
                thr = 0.9 * n['ret']['ref']; hit = [s for s, a in r['fresh_curve'] if a >= thr]; tt.append(hit[0] if hit else 305)
                auc.append(np.mean([a for s, a in r['fresh_curve'] if s <= 50]))
            print(f'   fresh model (task 1 alone, its own init) ttt {tt} auc50 {fmt(auc)}')
