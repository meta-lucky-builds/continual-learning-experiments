"""Summarise wave-2 results (plan2.md). Same pre-registered rule as analyze.py: a difference counts only if the paired
mean ± SE excludes 0 AND all 5 seeds share its sign."""
import json, os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
def load(name):
    p = os.path.join(D, name + '.jsonl')
    if not os.path.exists(p): return None
    v = [json.loads(l) for l in open(p) if l.startswith('{')]
    return sorted(v, key=lambda x: x['seed']) if len(v) == 5 else None
def ms(x): x = np.asarray(x, float); return x.mean(), x.std(ddof=1) / np.sqrt(len(x))
def fmt(x, sc=100): m, s = ms(x); return f'{sc*m:5.1f}±{sc*s:3.1f}'
def pdiff(a, b, sc=100):
    d = np.asarray(a, float) - np.asarray(b, float); m, s = ms(d)
    sig = (abs(m) > s) and ((d > 0).all() or (d < 0).all())
    return f'{sc*m:+6.1f}±{sc*s:4.1f} [{(d > 0).sum()}/5 >0]{" *" if sig else ""}'

for st in ['toyA', 'toyB', 'mnist']:
    for arm in ['er', 'ace_start']:
        v = load(f'W_{arm}__{st}')
        if v is None: continue
        acc = [r['acc'] for r in v]; bc = [r['ro']['biascorr'] for r in v]
        print(f'\n======== W_{arm}__{st}: acc {fmt(acc)}  biascorr {fmt(bc)}  refit_buf {fmt([r["ro"]["refit_buf"] for r in v])}')
        print('-- W3/R1 floors (oracle readouts on all training data) --')
        F = {k: [r['floors'][k] for r in v] for k in v[0]['floors']}
        for k in F: print(f'   {k:11s} {fmt(F[k])}' + ('' if k == 'lda_now' else f'   now - this: {pdiff(F["lda_now" if k.startswith("lda") else "refit_now"], F[k])}'))
        print('   ER model acc vs streaming LDA on frozen random features (lda_init):', pdiff(acc, F['lda_init']), '| vs lda_rp:', pdiff(acc, F['lda_rp']), '| biascorr vs lda_rp:', pdiff(bc, F['lda_rp']))
        print('-- W1 means vs covariance --')
        S = {**{k: [r['stats'][k] for r in v] for k in v[0]['stats']}, **{k: [r['stats2'][k] for r in v] for k in v[0]['stats2']}}
        for k in S: print(f'   {k:24s} {fmt(S[k])}')
        print('-- W2 drift: E_pooled, resid_norm_frac, E by task, LDA(pred means, current cov) --')
        for k in v[0]['drift']:
            if k == 'rel_drift': continue
            Ep = [r['drift'][k]['E_pooled'] for r in v]; rn = [r['drift'][k]['resid_norm_frac'] for r in v]; la = [r['drift'][k]['lda_fullcov'] for r in v]
            bt = {t: round(float(np.mean([r['drift'][k]['E_by_task'][t] for r in v])), 2) for t in v[0]['drift'][k]['E_by_task']}
            print(f'   {k:22s} E {fmt(Ep, 1)[:9]} resid {np.mean(rn):.2f}  byT {bt}  LDA {fmt(la)}  vs oracle {pdiff(la, [r["drift"]["oracle_now"]["lda_fullcov"] for r in v])}')
        print('   rel_drift (|then-now|/|now|):', np.round([r['drift']['rel_drift'] for r in v], 2))
        print('-- W7 interference prediction --')
        for pk in ['sim_feat', 'sim_feat_b1', 'sim_raw', 'dist_feat']:
            rc, pick = [], []
            for r in v:
                rows = [e for e in r['interf'] if e.get(pk) is not None]
                sign = -1 if pk == 'dist_feat' else 1
                per_task = []
                for t in sorted(set(e['ti'] for e in rows)):
                    R = [e for e in rows if e['ti'] == t]
                    from scipy.stats import spearmanr
                    per_task.append(spearmanr([sign * e[pk] for e in R], [e['conf_end'] for e in R]).correlation)
                rc.append(np.nanmean(per_task))
                hits = []
                for t in sorted(set(e['ti'] for e in rows)):
                    for o in sorted(set(e['old'] for e in rows if e['ti'] == t)):
                        pr = [e for e in rows if e['ti'] == t and e['old'] == o]
                        if len(pr) == 2 and pr[0]['conf_end'] != pr[1]['conf_end']:
                            hits.append((sign * pr[0][pk] > sign * pr[1][pk]) == (pr[0]['conf_end'] > pr[1]['conf_end']))
                pick.append(np.mean(hits))
            print(f'   {pk:12s} within-task Spearman {fmt(rc,1)}  pick-the-absorber {fmt(pick)} (chance 50) vs 0.5: {pdiff(pick, [0.5]*5)}')
        print('-- R12 logit adjustment (tau=1 pre-registered; ER only) vs biascorr and ER --')
        for k in ['last_tau1.0', 'last_tau0.5', 'last_tau2.0', 'all_tau1.0', 'last_rep_tau1.0']:
            x = [r['la'][k] for r in v]; print(f'   {k:16s} {fmt(x)}  vs ER {pdiff(x, acc)}  vs biascorr {pdiff(x, bc)}')
        g = [(r['la']['last_tau1.0'] - a) / (b - a) for r, a, b in zip(v, acc, bc)]
        print('   fraction of biascorr gain recovered by LA(last, tau=1):', np.round(g, 2))
        print('-- R13 sleep (200 steps on the buffer only) --')
        for k in ['head', 'full']:
            x = [r['sleep'][k] for r in v]; print(f'   {k:5s} {fmt(x)}  vs ER {pdiff(x, acc)}  vs biascorr {pdiff(x, bc)}')
        print('   full - head', pdiff([r['sleep']['full'] for r in v], [r['sleep']['head'] for r in v]))
v = load('W_er_nmem50__mnist')
if v:
    print('\n======== W_er_nmem50__mnist: acc', fmt([r['acc'] for r in v]))
    for k in v[0]['drift']:
        if k != 'rel_drift': print(f"   {k:22s} E {np.mean([r['drift'][k]['E_pooled'] for r in v]):.3f} LDA {fmt([r['drift'][k]['lda_fullcov'] for r in v])}")
    for k in ['last_tau1.0', 'last_rep_tau1.0']: print('  ', k, fmt([r['la'][k] for r in v]))
for st in ['toyA', 'mnist']:
    for arm in ['er', 'ace_start']:
        v = load(f'C_{arm}__{st}')
        if v is None: continue
        print(f'\n======== W6 cyclic C_{arm}__{st} (15 tasks = 3 cycles of 0-4)')
        occ = sorted(set(e['occurrence'] for e in v[0]['rets']))
        for o in occ:
            R = [[e for e in r['rets'] if e['occurrence'] == o] for r in v]
            print(f"   occurrence {o}: retention at return {fmt([np.mean([e['acc_task_start'] for e in x]) for x in R])} "
                  f"ttt {np.round(np.mean([[e['ttt'] for e in x] for x in R]), 1)}  auc50 {fmt([np.mean([e['auc50'] for e in x]) for x in R])}")
        for cy in range(3):
            a = [r['cyc'][5 * cy + 4]['acc_seen'] for r in v]; b = [r['cyc'][5 * cy + 4]['bc_seen'] for r in v]
            print(f'   end of cycle {cy + 1}: acc_seen {fmt(a)} biascorr {fmt(b)}')
        print('   cycle3 - cycle1 end acc', pdiff([r['cyc'][14]['acc_seen'] for r in v], [r['cyc'][4]['acc_seen'] for r in v]),
              '| occurrence3 - occurrence2 retention', pdiff([np.mean([e['acc_task_start'] for e in r['rets'] if e['occurrence'] == 3]) for r in v],
                                                          [np.mean([e['acc_task_start'] for e in r['rets'] if e['occurrence'] == 2]) for r in v]))
