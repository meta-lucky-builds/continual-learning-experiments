"""E2 analysis (plan.md, E2 section): streaming-valid alpha-hat and w-hat chosen on reserve items of seen classes at each task end.
Usage: python analyze_e2.py [resultsdir]"""
import sys, os, math, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from analyze_cifar import softmax, acc, paired

AL = [round(0.1 * i, 1) for i in range(16)]; WS = [round(0.1 * i, 1) for i in range(11)]
def nll(P, y): return float(-np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)).mean())
def restrict(P, seen):
    """Renormalise probabilities over seen classes (reserve/test items of seen classes are scored among seen classes only)."""
    Q = np.zeros_like(P); Q[:, seen] = P[:, seen]; return Q / Q.sum(1, keepdims=True)

def per_seed(R, s, yte, er_path=None, rc_path=None):
    er = np.load(er_path or f'{R}/er_lr0.03_{s}.npz'); rc = np.load(rc_path or f'{R}/rc_{s}.npz'); tasks = er['tasks'].tolist(); ry = er['resv_y']
    assert tasks == rc['tasks'].tolist() and np.array_equal(ry, rc['resv_y']) and np.array_equal(er['ylearn'], rc['ylearn'])
    out = {'seed': s, 'acc': {}, 'sel': {}}
    for t in range(C.NTASK):
        seen = C.seen_of(tasks, t); m = np.isin(ry, seen); yr = ry[m]
        assert len(yr) == C.CPT * (t + 1) * C.RESV, (len(yr), C.RESV)
        L, b = er[f'e{t}_test_logits'].astype(np.float64), er[f'e{t}_b_buf'].astype(np.float64); Lr = er[f'e{t}_resv_logits'].astype(np.float64)[m]
        PB = np.exp(rc[f'e{t}_test_lda_lp'].astype(np.float64)); PBr = np.exp(rc[f'e{t}_resv_lda_lp'].astype(np.float64))[m]
        # alpha-hat: reserve NLL of ER with alpha * b_buf, over seen classes
        a_hat = min(AL, key=lambda a: (nll(restrict(softmax(Lr - a * b), seen), yr), abs(a - 1)))
        PAr = restrict(softmax(Lr - a_hat * b), seen); PBr_s = restrict(PBr, seen)
        w_hat = min(WS, key=lambda w: (nll(w * PAr + (1 - w) * PBr_s, yr), abs(w - 0.5)))
        PA1, PAa = restrict(softmax(L - b), seen), restrict(softmax(L - a_hat * b), seen); PBs = restrict(PB, seen)
        a = {'ER': acc(L, yte, seen), 'ER+bc_buf': acc(L - b, yte, seen), 'ER+bc_ahat': acc(L - a_hat * b, yte, seen), 'RC-LDA': acc(PB, yte, seen),
             'MIX_half': acc(0.5 * PA1 + 0.5 * PBs, yte, seen), 'MIX_what_ahat': acc(w_hat * PAa + (1 - w_hat) * PBs, yte, seen),
             'POE_half': acc(0.5 * np.log(PA1 + 1e-300) + 0.5 * np.log(PBs + 1e-300), yte, seen),
             'POE_half_ahat': acc(0.5 * np.log(PAa + 1e-300) + 0.5 * np.log(PBs + 1e-300), yte, seen)}
        # oracle references (test set; diagnostic only)
        a['oracle_alpha_acc'] = max(acc(L - al * b, yte, seen) for al in AL)
        a['oracle_w_acc'] = max(acc(w * PAa + (1 - w) * PBs, yte, seen) for w in WS)
        out['acc'][t] = a; out['sel'][t] = {'alpha_hat': a_hat, 'w_hat': w_hat}
    return out

if __name__ == '__main__':
    R = sys.argv[1] if len(sys.argv) > 1 else 'results/e2'; assert C.RESV == 20, 'run with RESV=20'
    _, _, _, yte = C.load(); res = [per_seed(R, s, yte) for s in range(5)]
    fin = [r['acc'][C.NTASK - 1] for r in res]
    for x in fin: x['best_single'] = max(x['ER+bc_buf'], x['RC-LDA'])
    print(f"{'arm':16s} {'final':>13s} {'avg-incr':>9s}")
    for k in fin[0]:
        f = [x[k] for x in fin]; inc = [np.mean([r['acc'][t][k] for t in range(C.NTASK)]) for r in res] if k in res[0]['acc'][0] else [float('nan')]
        print(f"{k:16s} {100 * np.mean(f):6.1f} ± {100 * np.std(f, ddof=1) / math.sqrt(5):4.1f} {100 * np.mean(inc):8.1f}")
    print('selected (alpha_hat, w_hat) per seed per task end:', [[(r['sel'][t]['alpha_hat'], r['sel'][t]['w_hat']) for t in range(C.NTASK)] for r in res])
    rows = [{'k': x} for x in fin]
    for a, b in [('ER+bc_ahat', 'ER+bc_buf'), ('MIX_what_ahat', 'MIX_half'), ('POE_half', 'MIX_half'), ('MIX_what_ahat', 'best_single'),
                 ('MIX_half', 'best_single'), ('POE_half', 'best_single'), ('POE_half_ahat', 'POE_half')]:
        print('Q', paired(rows, a, b, 'k'))
    json.dump(res, open(os.path.join(R, 'analysis_e2.json'), 'w'), default=float, indent=0)
