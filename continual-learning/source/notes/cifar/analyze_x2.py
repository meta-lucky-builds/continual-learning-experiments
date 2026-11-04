"""Summary of X2 (x2_plan.md). Run from notes/cifar: python3 analyze_x2.py > results/x2/summary.txt"""
import json, numpy as np
R = [json.load(open(f'results/x2/x2_{s}.json'))['acc'] for s in range(5)]
a = lambda k: np.array([r[k] for r in R]) * 100
def st(x): m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); return f'{m:+.2f} ± {se:.2f} ' + ('*' if abs(m) > 2 * se and (np.all(x > 0) or np.all(x < 0)) else '')
for k in R[0]: print(f'{k:15s} {a(k).mean():6.2f} ± {a(k).std(ddof=1) / np.sqrt(5):.2f}')
print('gB: ER+bc_buf %.2f (want 22.7), BANK_4spread %.2f (want 31.2), RC %.2f (want 28.2)' % (a('ER+bc_buf').mean(), a('BANK_4spread').mean(), a('RC').mean()))
print('declared:')
for x, y in [('CH_cv', 'B_only'), ('CH_aff', 'CH_cv'), ('CH_shift', 'CH_cv'), ('MIX3_CH_cv', 'BANK_4spread'), ('O_mean', 'B_only'), ('O_full', 'O_mean'), ('CH_cv_tr', 'CH_cv')]: print(f'  {x} - {y}: {st(a(x) - a(y))}')
for k in ['CH_cv', 'CH_cv_tr', 'CH_aff', 'CH_shift', 'DI_cv', 'DI_aff']: print(f'  gap closed {k}: {((a(k) - a("B_only")) / (a("O_full") - a("B_only"))).mean():.2f}')
print('exploratory (not declared):')
for x, y in [('DI_aff', 'DI_cv'), ('DI_aff', 'CH_cv'), ('MIX3_CH_cv', 'MIX3_B_only'), ('CH_cv', 'BANK_4spread')]: print(f'  {x} - {y}: {st(a(x) - a(y))}')
