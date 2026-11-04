"""Audit follow-up: on the same frozen ER checkpoint, is the replay buffer a worse source of class statistics than
the same number of FRESH (never-trained-on) items per class? Oracle diagnostic: fresh items are not available in CL."""
import sys, os, json, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('rc', os.path.join(HERE, 'readout_checks.py')); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
M = R.M; K = 10
def run(setting, seed, ndraw=20):
    ck = torch.load(os.path.join(HERE, 'results', 'ckpt', f'er_{setting}_{seed}.pt')); buf = np.array(ck['buf'])
    Xtr, ytr, Xte, yte, m = M.setup(ck['cfg'], seed); m.load_state_dict(ck['state']); yn = ytr.numpy(); f = lambda X: M.feats(m, X)
    old, last = list(range(8)), [8, 9]
    def acc(idx):
        mus, cov = R.lda_from(f(Xtr[idx]), ytr[idx], list(range(K))); p = M.lda_predict_chunked(mus, cov, f, Xte)
        return M.class_acc(p, yte, range(K)), M.class_acc(p, yte, old), M.class_acc(p, yte, last)
    out = {'buffer': acc(buf)}; nper = len(buf) // K; rng = np.random.default_rng(seed + 99); fr = []
    for _ in range(ndraw):
        idx = np.concatenate([rng.choice(np.setdiff1d(np.where(yn == c)[0], buf), nper, replace=False) for c in range(K)]); fr.append(acc(idx))
    out['fresh'] = np.mean(fr, 0).tolist()
    # where do buffer features sit? distance of each class's buffer mean from its full-data mean, vs fresh-sample means
    Fall = f(Xtr).double(); mu = {c: Fall[ytr == c].mean(0) for c in range(K)}; Fb = f(Xtr[buf]).double(); yb = ytr[buf]
    out['bufmean_err_old'] = float(np.mean([float((Fb[yb == c].mean(0) - mu[c]).norm() / mu[c].norm()) for c in old]))
    fe = []
    for _ in range(ndraw):
        e = []
        for c in old:
            i = rng.choice(np.setdiff1d(np.where(yn == c)[0], buf), nper, replace=False); e.append(float((Fall[i].mean(0) - mu[c]).norm() / mu[c].norm()))
        fe.append(np.mean(e))
    out['freshmean_err_old'] = float(np.mean(fe)); out['nper'] = int(nper); out['seed'] = seed
    return out
if __name__ == '__main__':
    for s in [int(x) for x in sys.argv[2].split(',')]: print(json.dumps(run(sys.argv[1], s)), flush=True)
