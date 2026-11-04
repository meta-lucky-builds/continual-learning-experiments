"""Follow-up FU-d: recurrence savings through raw vs bias-corrected readouts (same significance rule)."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
def load(n):
    p = os.path.join(D, n + '.jsonl')
    if not os.path.exists(p): return None
    v = [json.loads(l) for l in open(p) if l.startswith('{')]; return sorted(v, key=lambda x: x['seed']) if len(v) == 5 else None
def ms(x): x = np.asarray(x, float); return x.mean(), x.std(ddof=1) / np.sqrt(len(x))
def fmt(x, sc=100): m, s = ms(x); return f'{sc*m:5.1f}±{sc*s:3.1f}'
def pdiff(a, b, sc=100):
    d = np.asarray(a, float) - np.asarray(b, float); m, s = ms(d); sig = abs(m) > s and ((d > 0).all() or (d < 0).all())
    return f'{sc*m:+6.1f}±{sc*s:4.1f} [{(d > 0).sum()}/5]{" *" if sig else ""}'
for st in ['toyA', 'mnist']:
    R = {sn: load(f'FU5_{sn}__{st}') for sn in ['none', 'rows', 'rows_init', 'full']}
    if R['none'] is None: continue
    print(f'\n==== FU5 {st}: return of task 1 after tasks 1-5. raw = model argmax; bc = bias-corrected readout')
    for sn, v in R.items():
        if v is None: print('  ', sn, 'not finished'); continue
        r = [x['ret'] for x in v]; b = [x['ret']['bc'] for x in v]
        print(f"   {sn:9s} raw: start {fmt([x['acc_task_start'] for x in r])} ttt {[x['ttt'] for x in r]} | bc: ref {fmt([x['ref_bc'] for x in b])} start {fmt([x['start_bc'] for x in b])} "
              f"ttt {[x['ttt_bc'] for x in b]} auc50 {fmt([x['auc50_bc'] for x in b])} | all-seen bc at start {fmt([x['seen_bc_curve'][0] for x in b])} end {fmt([x['seen_bc_curve'][-1] for x in b])}")
    for sn in ['rows', 'full']:
        if R[sn] is None or R['rows_init'] is None: continue
        g = lambda v, k: [x['ret']['bc'][k] for x in v]
        print(f"   {sn} - rows_init (null): bc start {pdiff(g(R[sn],'start_bc'), g(R['rows_init'],'start_bc'))} bc auc50 {pdiff(g(R[sn],'auc50_bc'), g(R['rows_init'],'auc50_bc'))}"
              f" | raw start {pdiff([x['ret']['acc_task_start'] for x in R[sn]], [x['ret']['acc_task_start'] for x in R['rows_init']])}")
        print(f"   {sn} - none: bc start {pdiff(g(R[sn],'start_bc'), g(R['none'],'start_bc'))}  all-seen bc end {pdiff([x['ret']['bc']['seen_bc_curve'][-1] for x in R[sn]], [x['ret']['bc']['seen_bc_curve'][-1] for x in R['none']])}")
    v = load(f'FUC_er__{st}')
    if v:
        print(f'   cyclic ER, bias-corrected retention / ttt per occurrence:')
        for o in (2, 3):
            E = [[e for e in r['rets'] if e['occurrence'] == o] for r in v]
            print(f"     occ {o}: raw start {fmt([np.mean([e['acc_task_start'] for e in x]) for x in E])} raw ttt {np.mean([[e['ttt'] for e in x] for x in E]):.1f} | "
                  f"bc start {fmt([np.mean([e['bc']['start_bc'] for e in x]) for x in E])} bc ttt {np.mean([[e['bc']['ttt_bc'] for e in x] for x in E]):.1f} cens {sum(e['bc']['censored_bc'] for x in E for e in x)}")
        a3 = [np.mean([e['bc']['start_bc'] for e in r['rets'] if e['occurrence'] == 3]) for r in v]; a2 = [np.mean([e['bc']['start_bc'] for e in r['rets'] if e['occurrence'] == 2]) for r in v]
        print('     occ3 - occ2 bc retention', pdiff(a3, a2))
