"""T7: context detection with real (toy) label evidence under conditional shift.

Label models: shared-covariance Gaussian (LDA) fitted on the toy's training set gives p(y|x) for context A.
Context B relabels a fraction q of the classes (cyclic permutation among them): same inputs, different P(y|x).
Context C (3-state runs) relabels a different set. Label models are the TRUE per-context rules applied to
LDA posteriors, smoothed: p_eps = (1-eps) p + eps/K, eps = 0.05 (spec B8.1). Temperature kappa = n_eff/|X|, n_eff = 4.
Filter: sticky HMM, h = 0.02, switch threshold pi = 0.9 (spec B8.2-B8.3).
Stream: 200 batches of A, then 300 of B. Batches of 16 test items.
Reported per q: mean realised per-batch log-likelihood ratio log L (B vs A, on B batches), detection rate and delay,
false switches during A; compared with the two-state L_min (0.166) and the simulated multi-state threshold.
S9 (selftest): with constant evidence L, the two-state simulator settles above 0.9 iff log L > log L_min.
"""
import os, sys, json, math, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
H, PI, EPS, NEFF, BS, K = 0.02, 0.9, 0.05, 4, 16, 10

def lmin(h=H, pi=PI):
    O = pi / (1 - pi); return O * (h * O + 1 - h) / ((1 - h) * O + h)

def transition(S, h=H):
    T = np.full((S, S), h / (S - 1)); np.fill_diagonal(T, 1 - h); return T

def filt(loglik_seq, S, start=0, h=H):
    """Forward filter over S states; loglik_seq: (n, S) tempered log-likelihoods. Returns posteriors (n, S)."""
    T = transition(S, h); p = np.zeros(S); p[start] = 1; out = []
    for ll in loglik_seq:
        p = p @ T; lw = np.log(np.maximum(p, 1e-300)) + ll; lw -= lw.max(); p = np.exp(lw); p /= p.sum(); out.append(p.copy())
    return np.array(out)

def selftest():
    ok = True
    L0 = math.log(lmin())
    for dl, expect in [(-0.01, False), (0.01, True)]:
        n = 20000; ll = np.zeros((n, 2)); ll[:, 1] = L0 + dl
        post = filt(ll, 2)[:, 1]; got = post.max() > PI
        print(('PASS' if got == expect else 'FAIL'), f'S9 two-state constant evidence log L = L_min{dl:+}: max posterior {post.max():.5f}, crosses 0.9: {got}')
        ok &= got == expect
    print('log L_min =', round(L0, 4)); return ok

def lda(Xtr, ytr):
    X = Xtr.numpy().astype(np.float64); y = ytr.numpy()
    mu = np.stack([X[y == k].mean(0) for k in range(K)]); R = X - mu[y]; S = R.T @ R / len(X) + 1e-6 * np.eye(X.shape[1])
    P = np.linalg.inv(S)
    def post(Z):
        Z = Z.numpy().astype(np.float64); s = Z @ P @ mu.T - 0.5 * np.einsum('kd,de,ke->k', mu, P, mu)
        s -= s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
    return post

def relabel(perm_classes):
    m = np.arange(K)
    if len(perm_classes) >= 2: m[perm_classes] = np.roll(perm_classes, 1)
    return m

def run(q, seed, S=2, nA=200, nB=300, temp=1.0):
    os.environ['SEP'] = '0.6'
    spec = importlib.util.spec_from_file_location('tc', os.path.join(HERE, '..', 'v13-checks', 'toy_cl3.py')); tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
    Xtr, ytr, Xte, yte = tc.make_data(seed); post = lda(Xtr, ytr)
    rng = np.random.default_rng(seed + 99); n_perm = int(round(q * K))
    order = rng.permutation(K); maps = [np.arange(K), relabel(order[:n_perm])]
    if S == 3:                                                    # a different rule: disjoint set if possible, else the opposite cycle
        if 2 * n_perm <= K: maps.append(relabel(order[n_perm:2 * n_perm]))
        else:
            mm = np.arange(K); pc = order[:n_perm]; mm[pc] = np.roll(pc, -1); maps.append(mm)
        assert not np.array_equal(maps[2], maps[1]) and not np.array_equal(maps[2], maps[0])
    Pte = post(Xte)                                               # p(y_true_class | x) under LDA
    # p(label | x, c): context c labels true class k as maps[c][k]
    def p_label(idx, c):
        Pc = np.zeros((len(idx), K)); Pc[:, maps[c]] = Pte[idx]; return (1 - EPS) * Pc + EPS / K
    true_ctx = [0] * nA + [1] * nB; LL = []; lr = []
    for b, c in enumerate(true_ctx):
        idx = rng.choice(len(yte), BS, replace=False); lab = maps[c][yte.numpy()[idx]]
        ll = temp * np.array([NEFF / BS * np.log(p_label(idx, s)[np.arange(BS), lab]).sum() for s in range(S)])
        LL.append(ll); lr.append(ll[1] - ll[0])
    P = filt(np.array(LL), S)
    false_sw = int((P[:nA, 1:].max(1) > PI).any())
    det = np.where(P[nA:, 1] > PI)[0]
    return {'q': q, 'S': S, 'seed': seed, 'temp': temp, 'mean_logL_B_batches': float(np.mean(lr[nA:])), 'sd_logL': float(np.std(lr[nA:])),
            'detected': bool(len(det)), 'delay': int(det[0]) + 1 if len(det) else None, 'false_switch_in_A': false_sw}

if __name__ == '__main__':
    if sys.argv[1] == 'selftest': sys.exit(0 if selftest() else 1)
    if sys.argv[1] == 'scan':   # T7b: scale the evidence into the threshold region (q = 0.2), 2 and 3 states, 20 seeds
        for S in (2, 3):
            for temp in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
                for seed in range(20):
                    print(json.dumps(run(0.2, seed, S, temp=temp)), flush=True)
        sys.exit(0)
    for S in (2, 3):
        for q in [0.0, 0.2, 0.3, 0.5, 1.0]:   # q = 0 is the null control (B == A)
            if S == 3 and q == 0: continue
            for seed in range(5):
                print(json.dumps(run(q, seed, S)), flush=True)
