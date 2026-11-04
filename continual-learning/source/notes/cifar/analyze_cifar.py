"""Analysis of E1 (plan.md). Usage: python analyze_cifar.py <resultsdir> <er_lr> <ace_lr> [seeds]
Reads <method>_lr<lr>_<seed>.npz (er, ace, frozen, joint1) and rc_<seed>.npz, rp_<seed>.npz. All mixtures are computed here from saved per-item outputs."""
import sys, os, math, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C

def softmax(L): L = L - L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)
def acc(P, yte, seen): return C.acc_seen(torch.from_numpy(np.asarray(P, np.float64)), yte, seen)
def post_w(la, lb):
    """Posterior weight of A under uniform prior given total log-likelihoods la, lb."""
    return 1.0 / (1.0 + math.exp(max(-700, min(700, lb - la))))

def per_seed(R, s, er_lr, ace_lr, yte):
    g = lambda n: np.load(os.path.join(R, f'{n}_{s}.npz'))
    er, ace, rc, rp = g(f'er_lr{er_lr}'), g(f'ace_lr{ace_lr}'), g('rc'), g('rp')
    fz = g(f'frozen_lr{er_lr}') if os.path.exists(os.path.join(R, f'frozen_lr{er_lr}_{s}.npz')) else None
    j1 = g(f'joint1_lr{er_lr}') if os.path.exists(os.path.join(R, f'joint1_lr{er_lr}_{s}.npz')) else None
    tasks = er['tasks'].tolist(); assert tasks == rc['tasks'].tolist() == ace['tasks'].tolist()
    yl = er['ylearn']; assert np.array_equal(yl, rc['ylearn']) and np.array_equal(yl, ace['ylearn'])  # same stream, same labels
    ry = er['resv_y']; nper = len(yl) // C.NTASK
    la_s, lb_s = er['preq_lp'], rc['preq_lp']; assert la_s.dtype == np.float64 and np.isfinite(la_s).all() and np.isfinite(lb_s).all()
    lm_s, _, _ = C.bayes_mix_stream(la_s, lb_s); NL = lambda lp: float(-lp.sum())
    bound_ok = NL(lm_s) <= min(NL(la_s), NL(lb_s)) + math.log(2) + 1e-6
    out = {'seed': s, 'tasks': tasks, 'stream_nll': {'er': NL(la_s) / len(yl), 'rc': NL(lb_s) / len(yl), 'mix': NL(lm_s) / len(yl),
                                                      'ace': NL(ace['preq_lp']) / len(yl), 'rp': NL(rp['preq_lp']) / len(yl)},
           'stream_nll_by_task': {k: [float(-v[t * nper:(t + 1) * nper].mean()) for t in range(C.NTASK)] for k, v in [('er', la_s), ('rc', lb_s), ('mix', lm_s)]},
           'bound_ok': bound_ok, 'acc': {}, 'w': {}}
    for t in range(C.NTASK):
        seen = C.seen_of(tasks, t); a = {}; sm = np.isin(ry, seen)
        L = er[f'e{t}_test_logits'].astype(np.float64); b = er[f'e{t}_b_buf']; rl = er[f'e{t}_resv_logits'].astype(np.float64)
        a['ER'] = acc(L, yte, seen); a['ER+bc_buf'] = acc(L - b, yte, seen)
        b_r = C.balanced_mean_logit(torch.from_numpy(rl[sm]), torch.from_numpy(ry[sm])).numpy(); a['ER+bc_resv'] = acc(L - b_r, yte, seen)
        a['ER_bufLDA'] = float(er[f'e{t}_acc_buflda'])
        La = ace[f'e{t}_test_logits'].astype(np.float64); a['ACE'] = acc(La, yte, seen); a['ACE+bc_buf'] = acc(La - ace[f'e{t}_b_buf'], yte, seen)
        PB = np.exp(rc[f'e{t}_test_lda_lp'].astype(np.float64)); a['RC-LDA'] = acc(PB, yte, seen)
        a['RP-LDA'] = acc(np.exp(rp[f'e{t}_test_lda_lp'].astype(np.float64)), yte, seen)
        if fz is not None: a['FZ-LDA'] = acc(fz[f'e{t}_test_lda_lp'], yte, seen)
        # mixtures (restricted renormalisation over seen classes is unnecessary for argmax)
        PA = softmax(L); PAbc = softmax(L - b)
        n_t = (t + 1) * nper; w1 = post_w(float(la_s[:n_t].sum()), float(lb_s[:n_t].sum()))
        a['MIX_stream'] = acc(w1 * PA + (1 - w1) * PB, yte, seen)
        ra = softmax(rl - b)[sm, ry[sm]]; rb = np.exp(rc[f'e{t}_resv_lda_lp'].astype(np.float64))[sm, ry[sm]]
        w2 = post_w(float(np.log(np.maximum(ra, 1e-300)).sum()), float(np.log(np.maximum(rb, 1e-300)).sum()))
        a['MIX_resv'] = acc(w2 * PAbc + (1 - w2) * PB, yte, seen); a['MIX_half'] = acc(0.5 * PAbc + 0.5 * PB, yte, seen)
        a['NULL_w1'] = acc(1.0 * PAbc + 0.0 * PB, yte, seen); assert abs(a['NULL_w1'] - a['ER+bc_buf']) < 1e-12  # null control
        out['acc'][t] = a; out['w'][t] = {'stream_wER': w1, 'resv_wERbc': w2}
    if j1 is not None: out['acc'][C.NTASK - 1]['JOINT1'] = acc(j1[f'e{C.NTASK - 1}_test_logits'], yte, list(range(C.K)))
    return out

def paired(rows, a, b, key):
    d = np.array([r[key][a] - r[key][b] for r in rows]); se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float('nan')
    sig = (abs(d.mean()) > 2 * se) and (np.all(d > 0) or np.all(d < 0))
    return f'{a} - {b}: {100 * d.mean():+.1f} ± {100 * se:.1f} (signs +{int((d > 0).sum())}/-{int((d < 0).sum())}){" *" if sig else ""}'

if __name__ == '__main__':
    R, er_lr, ace_lr = sys.argv[1], sys.argv[2], sys.argv[3]; seeds = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '0,1,2,3,4').split(',')]
    _, _, _, yte = C.load(); res = [per_seed(R, s, er_lr, ace_lr, yte) for s in seeds]
    print('bound_ok all seeds:', all(r['bound_ok'] for r in res))
    fin = [{'f': r['acc'][C.NTASK - 1], 'i': {k: np.mean([r['acc'][t][k] for t in range(C.NTASK)]) for k in r['acc'][0]}} for r in res]
    keys = list(fin[0]['f'])
    print(f"{'arm':12s} {'final':>14s} {'avg-incr':>14s}")
    for k in keys:
        f = [x['f'][k] for x in fin]; i = [x['i'].get(k, float('nan')) for x in fin]
        print(f"{k:12s} {100 * np.mean(f):6.1f} ± {100 * np.std(f, ddof=1) / math.sqrt(len(f)):4.1f}  {100 * np.nanmean(i):6.1f}")
    print('stream NLL (mean over seeds):', {k: round(float(np.mean([r['stream_nll'][k] for r in res])), 3) for k in res[0]['stream_nll']})
    print('mixture weights at final:', [{k: round(v, 4) for k, v in r['w'][C.NTASK - 1].items()} for r in res])
    for x in fin:
        x['f']['best_single'] = max(x['f']['ER+bc_buf'], x['f']['RC-LDA']); x['f']['best_mix'] = max(x['f']['MIX_resv'], x['f']['MIX_half'])
    F = [x['f'] for x in fin]
    for a, b in [('RC-LDA', 'ER+bc_buf'), ('best_mix', 'best_single'), ('MIX_resv', 'best_single'), ('MIX_half', 'best_single'), ('ACE', 'ER+bc_buf'),
                 ('FZ-LDA', 'RC-LDA'), ('ER+bc_resv', 'ER+bc_buf'), ('ER_bufLDA', 'ER+bc_buf'), ('ER+bc_buf', 'ER'), ('RC-LDA', 'RP-LDA'), ('MIX_stream', 'ER')]:
        if a in F[0] and b in F[0]: print('P', paired([{'k': x} for x in F], a, b, 'k'))
    json.dump(res, open(os.path.join(R, 'analysis.json'), 'w'), default=float, indent=0)
