"""Audit follow-ups on FROZEN checkpoints (2026-10-29; answers DEEP_AUDIT_AND_REDESIGN.md sections 1 and 9B).

For each seed's saved ER checkpoint (results/ckpt/er_<setting>_<seed>.pt, written by mar2.py with save_model):

1. Sample-size sweep, features held fixed. LDA (means + pooled covariance, shrink 0.1) fitted on k random TRAINING
   items per class, k in KS, 10 draws per k (own RNG), plus k = all. Feature maps:
     now     - the trained network after the stream (diagnostic only: not streaming-valid; needs the old inputs)
     init    - the same seed's untrained network (same width, 256 / 64)
     rp<w>   - random ReLU expansion of the raw input, w = 256 and 1024 (MNIST) / 64 and 512 (toy)
     raw     - the input itself
   Separates readout sample size from feature quality on the same weights (the audit's request).
2. Width control for the random expansion with full statistics: w in WIDTHS.

Usage: python readout_checks.py <toyA|mnist> <seeds>   -> one JSON line per seed
       python readout_checks.py selftest
"""
import sys, os, json, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('mar2', os.path.join(HERE, 'mar2.py')); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
torch.set_num_threads(int(os.environ.get('NT', '1')))
KS = [5, 10, 20, 50, 100, 200]; NDRAW = 10; K = 10

def lda_from(F, y, classes):
    F = F.double(); mus = {k: F[y == k].mean(0) for k in classes}
    R = torch.cat([F[y == k] - mus[k] for k in classes]); cov = R.T @ R / max(1, len(R) - len(classes))
    return mus, cov

def sweep(fmap, Xtr, ytr, Xte, yte, rng):
    yn = ytr.numpy(); classes = list(range(K)); res = {}
    for k in KS:
        accs = []
        for _ in range(NDRAW):
            idx = np.concatenate([rng.choice(np.where(yn == c)[0], k, replace=False) for c in classes])
            mus, cov = lda_from(fmap(Xtr[idx]), ytr[idx], classes)
            accs.append(M.class_acc(M.lda_predict_chunked(mus, cov, fmap, Xte), yte, classes))
        res[str(k)] = float(np.mean(accs))
    mus, cov = M.lda_fit_chunked(fmap, Xtr, ytr, classes); res['all'] = M.class_acc(M.lda_predict_chunked(mus, cov, fmap, Xte), yte, classes)
    return res

def run(setting, seed):
    ck = torch.load(os.path.join(HERE, 'results', 'ckpt', f'er_{setting}_{seed}.pt'))
    Xtr, ytr, Xte, yte, m = M.setup(ck['cfg'], seed); _, _, _, _, m0 = M.setup(ck['cfg'], seed)
    m.load_state_dict(ck['state'])
    with torch.no_grad(): acc_model = M.class_acc(m(Xte).argmax(1), yte, range(K))
    mn = setting == 'mnist'; widths_sw = [256, 1024] if mn else [64, 512]; widths_full = [256, 1024, 4096] if mn else [64, 512, 2048]
    fm = {'now': lambda X: M.feats(m, X), 'init': lambda X: M.feats(m0, X), 'raw': lambda X: X}
    for w in widths_sw: fm[f'rp{w}'] = M.rp_map(Xtr.shape[1], w, seed + 777)
    out = {'setting': setting, 'seed': seed, 'acc_model': acc_model, 'sweep': {}}
    for name, f in fm.items(): out['sweep'][name] = sweep(f, Xtr, ytr, Xte, yte, np.random.default_rng(seed * 1000 + 5))
    out['width_full'] = {}
    for w in widths_full:
        f = M.rp_map(Xtr.shape[1], w, seed + 777); mus, cov = M.lda_fit_chunked(f, Xtr, ytr, list(range(K)))
        out['width_full'][str(w)] = M.class_acc(M.lda_predict_chunked(mus, cov, f, Xte), yte, list(range(K)))
    return out

def selftest():
    ok = True
    g = torch.Generator().manual_seed(3); Z = torch.randn(600, 5, generator=g, dtype=torch.float64); y = torch.arange(3).repeat(200)
    mus, cov = lda_from(Z, y, [0, 1, 2]); mu2, cov2 = M.lda_fit_chunked(lambda X: X, Z, y, [0, 1, 2], chunk=77)
    c1 = all(torch.allclose(mus[k], mu2[k]) for k in range(3)) and torch.allclose(cov, cov2); print(('PASS' if c1 else 'FAIL'), 'R1 small-sample LDA stats equal chunked stats'); ok &= c1
    # R2 planted: two well-separated Gaussian classes in 2-D, LDA from 5 per class must be ~perfect; from 'all' too
    Xa = torch.cat([torch.randn(500, 2, generator=g) + 6, torch.randn(500, 2, generator=g) - 6]); ya = torch.cat([torch.zeros(500), torch.ones(500)]).long()
    mus, cov = lda_from(Xa[torch.tensor(list(range(5)) + list(range(500, 505)))], ya[torch.tensor(list(range(5)) + list(range(500, 505)))], [0, 1])
    acc = float((M.lda_predict(mus, cov, Xa) == ya).float().mean()); c2 = acc > 0.999; print(('PASS' if c2 else 'FAIL'), 'R2 planted separable: 5-shot LDA ~ 100%', acc); ok &= c2
    print('SELFTEST', 'ALL PASS' if ok else 'FAILURES'); return ok

if __name__ == '__main__':
    if sys.argv[1] == 'selftest': sys.exit(0 if selftest() else 1)
    for s in [int(x) for x in sys.argv[2].split(',')]: print(json.dumps(run(sys.argv[1], s)), flush=True)
