# ===== atlas_lib: pure-numpy analysis code (estimators, LDA, statistics) =====
import numpy as np

def ledoit_wolf(Xc):
    """Ledoit-Wolf shrinkage covariance of already-centred data Xc (n x p).
    Same formula as sklearn.covariance.ledoit_wolf (assume_centered=True)."""
    Xc = np.asarray(Xc, dtype=np.float64)
    n, p = Xc.shape
    X2 = Xc ** 2
    emp_cov_trace = X2.sum(0) / n
    mu = emp_cov_trace.sum() / p
    beta_ = (X2.T @ X2).sum()
    delta_ = ((Xc.T @ Xc) ** 2).sum() / n ** 2
    beta = 1.0 / (p * n) * (beta_ / n - delta_)
    delta = (delta_ - 2.0 * mu * emp_cov_trace.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    s = 0.0 if beta == 0 else beta / delta
    S = Xc.T @ Xc / n
    return (1 - s) * S + s * mu * np.eye(p), s

def class_center(X, y):
    """Subtract each sample's class mean. Returns centred X and dict of class means."""
    Xc = np.empty_like(X, dtype=np.float64)
    means = {}
    for c in np.unique(y):
        m = y == c
        means[c] = X[m].mean(0)
        Xc[m] = X[m] - means[c]
    return Xc, means

def pooled_cov(X, y):
    Xc, _ = class_center(X, y)
    return ledoit_wolf(Xc)[0]

def lda_predict(X, M, Sigma):
    """Uniform-prior LDA. M: (C x d) class means (rows), Sigma: (d x d). Returns argmax index into rows of M."""
    P = np.linalg.solve(Sigma, M.T)            # d x C  = Sigma^-1 mu_c
    b = -0.5 * np.einsum('dc,cd->c', P, M)     # -1/2 mu^T Sigma^-1 mu
    S = X @ P + b
    return S.argmax(1)

EIGH = np.linalg.eigh   # the notebook may swap in a GPU float64 eigh with identical results (checked in a cell)

def _ridge_factory(Xs, Y):
    """Returns a function lam -> ridge coefficients for standardised Xs, using the smaller Gram matrix."""
    n, p = Xs.shape
    if n >= p:
        ev, V = EIGH(Xs.T @ Xs); ev = np.clip(ev, 0, None)
        VtXtY = V.T @ (Xs.T @ Y)
        return lambda lam: V @ (VtXtY / (ev + lam)[:, None])
    ev, U = EIGH(Xs @ Xs.T); ev = np.clip(ev, 0, None)
    UtY = U.T @ Y
    return lambda lam: Xs.T @ (U @ (UtY / (ev + lam)[:, None]))

def ridge_cv(X, Y, rng, k=5, ngrid=16, split=None, kappas=(0.0, 0.1, 0.3, 1.0)):
    """Ridge regression Y ~ X @ B on centred data (no intercept), penalty chosen by k-fold CV on (X,Y) only.
    Columns of X are standardised internally; returned B is in original units.
    split=p1: columns [:p1] form a separate block (the stable chart r) whose penalty is chosen separately,
    by scaling that block by kappa (kappa=0 drops it). This stops a wide noisy block from forcing one
    shared penalty onto the informative block."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    n = X.shape[0]
    sd = X.std(0) + 1e-8
    Xs0 = X / sd
    ks = kappas if split else (1.0,)
    k = max(2, min(k, n))
    folds = rng.permutation(n) % k
    best = (np.inf, None, None)
    for kap in ks:
        scale = np.ones(X.shape[1])
        if split: scale[:split] = kap
        Xs = Xs0 * scale if not (split and kap == 0) else Xs0[:, split:]
        smax2 = np.linalg.norm(Xs, 2) ** 2 + 1e-12
        lams = smax2 * np.logspace(-9, 1, ngrid)
        err = np.zeros(ngrid)
        for f in range(k):
            tr, te = folds != f, folds == f
            coef = _ridge_factory(Xs[tr], Y[tr])
            for jj, lam in enumerate(lams):
                err[jj] += ((Xs[te] @ coef(lam) - Y[te]) ** 2).sum()
        jj = int(err.argmin())
        if err[jj] < best[0]:
            best = (err[jj], kap, lams[jj])
    _, kap, lam = best
    if split and kap == 0:
        B = np.zeros((X.shape[1], Y.shape[1]))
        B[split:] = _ridge_factory(Xs0[:, split:], Y)(lam) / sd[split:, None]
        return B, lam, kap
    scale = np.ones(X.shape[1])
    if split: scale[:split] = kap
    B = _ridge_factory(Xs0 * scale, Y)(lam) * (scale / sd)[:, None]
    return B, lam, kap

def greg_means(classes, mu_aux, A_br, Y_br, y_br, rng, crossfit=False, split=None):
    """Regression (control-variate / GREG) estimate of class means of Y.
    classes : list of class ids to estimate
    mu_aux  : dict class -> population mean of auxiliary features (known or carried)
    A_br, Y_br, y_br : bridge items' auxiliary features, target features, labels
    Estimate: mu_Y,c = mean_B,c(Y) + (mu_aux,c - mean_B,c(A)) @ B, B = pooled within-class slope.
    Default: one pooled slope fitted on all bridge items (crossfit=False). Optional 2-fold cross-fitting;
    on synthetic data it lost badly with small bridges (5 items/class), so it is off (decided before any CIFAR run).
    Classes with no bridge items fall back to a global affine map (flagged).
    Returns dict class -> estimate, set of fallback classes, info."""
    A_br = np.asarray(A_br, np.float64); Y_br = np.asarray(Y_br, np.float64)
    fold = np.zeros(len(y_br), int)
    for c in np.unique(y_br):
        idx = np.where(y_br == c)[0]
        fold[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % 2
    out, fallback = {}, set()
    present = set(np.unique(y_br).tolist())
    if crossfit:
        Bs = []
        for f in (0, 1):
            tr = fold != f
            Ac, _ = class_center(A_br[tr], y_br[tr]); Yc, _ = class_center(Y_br[tr], y_br[tr])
            Bs.append(ridge_cv(Ac, Yc, rng, split=split)[0])
    Ac_all, _ = class_center(A_br, y_br); Yc_all, _ = class_center(Y_br, y_br)
    B_all = ridge_cv(Ac_all, Yc_all, rng, split=split)[0]
    for c in classes:
        if c not in present:
            fallback.add(c)
            out[c] = Y_br.mean(0) + (mu_aux[c] - A_br.mean(0)) @ B_all
            continue
        if not crossfit:
            m = y_br == c
            out[c] = Y_br[m].mean(0) + (mu_aux[c] - A_br[m].mean(0)) @ B_all
            continue
        est, w = 0.0, 0
        for f in (0, 1):
            m = (y_br == c) & (fold == f)
            if m.sum() == 0:
                continue
            est = est + m.sum() * (Y_br[m].mean(0) + (mu_aux[c] - A_br[m].mean(0)) @ Bs[1 - f])
            w += m.sum()
        out[c] = est / w
    return out, fallback

def global_affine_means(classes, mu_aux, A_br, Y_br, y_br, rng, split=None):
    """X2-style chained map: pooled within-class-centred ridge slope, global intercept."""
    A_br = np.asarray(A_br, np.float64); Y_br = np.asarray(Y_br, np.float64)
    Ac, _ = class_center(A_br, y_br); Yc, _ = class_center(Y_br, y_br)
    B = ridge_cv(Ac, Yc, rng, split=split)[0]
    a0, y0 = A_br.mean(0), Y_br.mean(0)
    return {c: y0 + (mu_aux[c] - a0) @ B for c in classes}

def within_class_r2(A, Y, y, rng, split=None):
    """Out-of-fold within-class R^2 of Y predicted from A (2-fold cross-fit, centring by training-fold class means)."""
    A = np.asarray(A, np.float64); Y = np.asarray(Y, np.float64)
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        fold[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % 2
    sse = sst = 0.0
    for f in (0, 1):
        tr, te = fold != f, fold == f
        Ac, ma = class_center(A[tr], y[tr]); Yc, my = class_center(Y[tr], y[tr])
        B = ridge_cv(Ac, Yc, rng, split=split)[0]
        for c in np.unique(y[te]):
            if c not in ma:
                continue
            m = te & (y == c)
            pred = my[c] + (A[m] - ma[c]) @ B
            sse += ((Y[m] - pred) ** 2).sum()
            sst += ((Y[m] - my[c]) ** 2).sum()
    return 1 - sse / sst

def paired(a, b):
    """Pre-registered rule: diff = a-b over seeds; significant iff |mean| > 2*SE and all seeds share the sign."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    sig = bool(n > 1 and abs(d.mean()) > 2 * se and (np.all(d > 0) or np.all(d < 0)))
    return dict(mean=float(d.mean()), se=float(se), n_pos=int((d > 0).sum()), n=n, sig=sig)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

# ---------- v2 additions: scores, calibration, product of experts, epsilon-mixture ----------
def lda_scores(X, M, Sigma):
    """Uniform-prior LDA discriminant scores (n x C); argmax equals lda_predict."""
    P = np.linalg.solve(Sigma, M.T)
    return X @ P - 0.5 * np.einsum('dc,cd->c', P, M)

def log_softmax(S):
    S = S - S.max(1, keepdims=True)
    return S - np.log(np.exp(S).sum(1, keepdims=True))

TAUS = np.exp(np.linspace(np.log(0.05), np.log(50), 41))

def fit_temperature(S, yi):
    """Temperature tau minimising NLL of softmax(S/tau) on (S, yi). Grid search (robust, no optimiser state)."""
    nll = [-log_softmax(S / t)[np.arange(len(yi)), yi].mean() for t in TAUS]
    return float(TAUS[int(np.argmin(nll))])

def fit_poe(L1, L2, yi, grid=np.linspace(0, 1, 21)):
    """Weight w for w*L1 + (1-w)*L2 (log-probs of two calibrated experts), chosen on control data by accuracy,
    ties broken by NLL after temperature calibration. Returns w, tau of the combination."""
    best = None
    for w in grid:
        Z = w * L1 + (1 - w) * L2
        acc = (Z.argmax(1) == yi).mean()
        tau = fit_temperature(Z, yi)
        nll = -log_softmax(Z / tau)[np.arange(len(yi)), yi].mean()
        key = (acc, -nll)
        if best is None or key > best[0]:
            best = (key, float(w), tau)
    return best[1], best[2]

def eps_mixture_logp(LA, LP, eps):
    """log( eps*pA + (1-eps)*pP ) computed stably from log-probs."""
    a = np.log(eps) + LA; b = np.log1p(-eps) + LP
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))
