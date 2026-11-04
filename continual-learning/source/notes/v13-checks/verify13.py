"""Independent checks of the second audit's claims and of the v1.3 fixes. Algebraic/toy only."""
import numpy as np, math, json
from math import comb
from scipy.optimize import minimize
rng = np.random.default_rng(7)
R = {}

# 1. Group rates applied after the solve -> ascent; rates inside B -> descent (Woodbury).
Kinv = np.array([[.5, -.4], [-.4, .5]]); E = np.diag([1., 100.]); a = np.array([1., .1])
R['post_scale_ascent'] = float(a @ (-E @ Kinv @ a))
bad = 0; err = 0
for _ in range(500):
    N, r = 30, 5
    Bd = np.exp(rng.uniform(-12, -2, N))            # eta_i/(sqrt v + eps), group rates spanning decades
    C = rng.normal(size=(N, r)); aa = rng.normal(size=N); lam = 10 ** rng.uniform(-2, 6)
    dense = np.linalg.solve(np.diag(1 / Bd) + lam * C @ C.T, -aa)
    S = np.eye(r) / lam + C.T @ (Bd[:, None] * C)
    wood = -Bd * aa + (Bd[:, None] * C) @ np.linalg.solve(S, C.T @ (Bd * aa))
    err = max(err, np.linalg.norm(wood - dense) / np.linalg.norm(dense)); bad += (aa @ wood >= 0)
R['woodbury_group_B'] = {'max_rel_err': err, 'non_descent': int(bad)}

# 2. No-effect condition is H Delta0 = 0, not H a = 0.
H = np.ones((2, 2)); D = np.diag([1., 2.]); m = np.array([1., -1.])
d0 = -D @ m; dw = np.linalg.solve(np.linalg.inv(D) + H, -m)
R['nullspace_condition'] = {'Hm': (H @ m).tolist(), 'HDelta0': (H @ d0).tolist(), 'protected': dw.tolist()}

# 3. Sketch: over- and under-protection; and how much worse is the step chosen with a sketch,
#    measured on the EXACT pool metric, in a random setting (not only the adversarial 2-D case).
xi = np.array([2., 0.]); R['sketch_eigs_minus_exact'] = np.linalg.eigvalsh(np.outer(xi, xi) - np.eye(2)).tolist()
ratios = []
for _ in range(200):
    N, n, K = 200, 20, 10
    J = rng.normal(size=(n, K, N)) / math.sqrt(N)          # per-item logit Jacobians
    Hex = sum(J[i].T @ J[i] for i in range(n)) / n
    G = np.stack([J[i].T @ rng.normal(size=K) for i in range(n)], 1)   # one projection per item
    Hsk = G @ G.T / n
    aa = rng.normal(size=N); lam = 50.
    de = np.linalg.solve(np.eye(N) + lam * Hex, -aa); ds = np.linalg.solve(np.eye(N) + lam * Hsk, -aa)
    ratios.append((ds @ Hex @ ds) / (de @ Hex @ de))
R['sketch_vs_exact_damage_ratio_median'] = float(np.median(ratios))   # >1 means the sketch-chosen step damages more

# 4. Conditional novelty: broad 'new' input model vs label evidence; and the shared-input fix.
d, neff, s = 768, 4, 2
R['broad_new_input_logratio'] = neff * d * ((1 - s ** -2) / 2 - math.log(s))
R['max_label_adv_floor'] = neff * math.log(1000)
# with new_cond sharing p_in(X|c), the comparison is labels only: a permuted-label batch of 16, kappa=1/4
p_old_correct = .9; K = 10
eps = .05; base = 1 / K
ll_old = math.log((1 - eps) * (1 - p_old_correct) / (K - 1) + eps * base)   # old head on new rule: label it puts mass ~0.011 on
ll_new = math.log(base)                                                      # new_cond starts at base distribution
R['new_cond_per_item_log_adv'] = ll_new - ll_old

# 5. Sticky HMM: minimum per-batch likelihood ratio to ever reach posterior pi (symmetric 2-state).
def Lmin(h, pi):
    O = pi / (1 - pi); return O * (h * O + 1 - h) / ((1 - h) * O + h)
def run(L, h, steps=20000):
    O = h / (1 - h)
    for _ in range(steps): O = L * ((1 - h) * O + h) / (h * O + 1 - h)
    return O / (1 + O)
h = .02; lm = Lmin(h, .9)
R['hmm'] = {'logLmin_h02_pi09': math.log(lm), 'post_at_e^0.1': run(math.exp(.1), h),
            'post_just_below': run(lm * 0.999, h), 'post_just_above': run(lm * 1.001, h),
            'static_formula_coeff_kappa_n_4': (math.log(9) + math.log(49)) / 4}

# 6. Certificates: translation false alarm; centered certificate m > sqrt2 * ||P_perp d||; exactness check.
viol = 0; tested = 0
for _ in range(20000):
    K = rng.integers(2, 12); fr = rng.normal(size=K) * 3; y = int(np.argmax(fr))
    dlt = rng.normal(size=K) * rng.uniform(0, 3) + rng.normal() * 10      # includes big common shifts
    fn = fr + dlt; mref = fr[y] - np.max(np.delete(fr, y))
    Pd = dlt - dlt.mean()
    if mref > math.sqrt(2) * np.linalg.norm(Pd):
        tested += 1; viol += (int(np.argmax(fn)) != y)
R['centered_cert'] = {'certified': tested, 'violations': int(viol)}

# 7. Power with regressions (exact multinomial), n=53, same net improvement.
def pw(n, pp, pm, alpha=.05):
    tot = 0.
    for b in range(n + 1):
        for c in range(n - b + 1):
            nd = b + c
            if nd == 0: continue
            tail = sum(comb(nd, k) for k in range(b, nd + 1)) / 2 ** nd
            if tail < alpha: tot += comb(n, b) * comb(n - b, c) * pp ** b * pm ** c * (1 - pp - pm) ** (n - b - c)
    return tot
R['power_regr'] = {str(k): pw(53, *k) for k in [(.125, 0.), (.25, .125), (.5, .375)]}
# worst case over discordance for net 0.125 at n: smallest n with envelope power >= 0.8 for d in [0.125, 0.4]
def env(n): return min(pw(n, (dd + .125) / 2, (dd - .125) / 2) for dd in np.linspace(.125, .4, 12))
R['n_for_envelope_0.8_d_le_0.4'] = next(n for n in range(53, 400, 5) if env(n) >= .8)

# 8. Margin-constrained step = B^{-1}-projection of the Adam step onto {C^T Delta >= -s}, s >= 0.
def qp_dual(Bd, C, d0, s):
    G = C.T @ (Bd[:, None] * C); r = C.T @ d0 + s
    f = lambda mu: (.5 * mu @ G @ mu + mu @ r, G @ mu + r)
    res = minimize(f, np.zeros(len(s)), jac=True, bounds=[(0, None)] * len(s), method='L-BFGS-B',
                   options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 5000})
    mu = res.x
    for _ in range(50):                      # exact active-set polish: solve KKT on the free set
        F = mu > 1e-14
        muF = np.zeros_like(mu)
        if F.any(): muF[F] = np.linalg.solve(G[np.ix_(F, F)], -r[F])
        grad = G @ muF + r
        if (muF >= -1e-13).all() and (grad[~F] >= -1e-11).all(): mu = np.maximum(muF, 0); break
        mu = np.maximum(muF, 0); mu[(~F) & (grad < 0)] = 1e-12  # free a violating constraint
    return d0 + Bd * (C @ mu)
worst = {'descent_fail': 0, 'nonexp_fail': 0, 'feas_viol': 0., 'primal_gap': 0.}
for _ in range(200):
    N, n = 25, 8
    Bd = np.exp(rng.uniform(-3, 3, N)); C = rng.normal(size=(N, n)); aa = rng.normal(size=N)
    s = np.abs(rng.normal(size=n)) * rng.uniform(0, 1); d0 = -Bd * aa
    dq = qp_dual(Bd, C, d0, s)
    worst['feas_viol'] = max(worst['feas_viol'], float(np.max(-(C.T @ dq) - s)))
    worst['descent_fail'] += int(aa @ dq > -np.sum(dq * dq / Bd) + 1e-9)
    worst['nonexp_fail'] += int(np.sum(dq * dq / Bd) > np.sum(d0 * d0 / Bd) + 1e-9)
    # primal check with SLSQP
    obj = lambda x: .5 * np.sum((x - d0) ** 2 / Bd)
    cons = {'type': 'ineq', 'fun': lambda x: C.T @ x + s, 'jac': lambda x: C.T}
    pr = minimize(obj, np.zeros(N), jac=lambda x: (x - d0) / Bd, constraints=[cons], method='SLSQP',
                  options={'ftol': 1e-14, 'maxiter': 1000})
    worst['primal_gap'] = max(worst['primal_gap'], abs(obj(pr.x) - obj(dq)) / max(1e-12, obj(dq)))
R['margin_qp'] = worst

# 3b. Damage measured as first-order MARGIN decrease on pool items (what retention needs), and progress <a,Delta>.
def margins_setup(N=200, n=20, K=10):
    J = rng.normal(size=(n, K, N)) / math.sqrt(N); logits = rng.normal(size=(n, K)) * 2
    y = logits.argmax(1); top = []
    for i in range(n):
        o = np.argsort(-logits[i]); top.append([j for j in o if j != y[i]][:2])
    return J, y, top
res = {k: [] for k in ['exact_quad', 'random_sketch', 'rival_sketch', 'margin_qp', 'adam']}
prog = {k: [] for k in res}
for _ in range(100):
    N, n, K = 200, 20, 10
    J, y, top = margins_setup(N, n, K)
    Mg = np.stack([J[i][y[i]] - J[i][j] for i in range(n) for j in range(K) if j != y[i]], 1)  # all margin grads
    aa = rng.normal(size=N); Bd = np.ones(N); lam = 50.; d0 = -aa
    Hex = sum(J[i].T @ (np.eye(K) - 1 / K) @ J[i] for i in range(n)) / n
    Gr = np.stack([J[i].T @ ((lambda e: e - e.mean())(rng.normal(size=K))) for i in range(n)], 1)
    Gm = np.stack([J[i][y[i]] - J[i][j] for i in range(n) for j in top[i]], 1)
    steps = {'adam': d0,
             'exact_quad': np.linalg.solve(np.eye(N) + lam * Hex, -aa),
             'random_sketch': np.linalg.solve(np.eye(N) + lam * Gr @ Gr.T / n, -aa),
             'rival_sketch': np.linalg.solve(np.eye(N) + lam * Gm @ Gm.T / n, -aa),
             'margin_qp': qp_dual(Bd, Gm, d0, np.zeros(Gm.shape[1]))}
    for k, dl in steps.items():
        res[k].append(np.sum(np.maximum(0, -(Mg.T @ dl)))); prog[k].append(-(aa @ dl))
R['margin_damage_mean'] = {k: float(np.mean(v)) for k, v in res.items()}
R['progress_mean'] = {k: float(np.mean(v)) for k, v in prog.items()}

# 9. Cumulative first-order guarantee: slack s = k(m - m_floor) keeps linearized margins above m_floor forever.
k = .1; m = 2.; mf = .5
for t in range(1000): m = m - k * (m - mf)        # worst case: every step uses the full slack
R['geometric_slack_margin_after_1000'] = m

print(json.dumps(R, indent=1))
assert R['post_scale_ascent'] > 0 and R['woodbury_group_B']['non_descent'] == 0
assert R['hmm']['post_just_below'] < .9 < R['hmm']['post_just_above'] and R['hmm']['post_at_e^0.1'] < .9
assert R['centered_cert']['violations'] == 0
assert R['margin_qp']['descent_fail'] == 0 and R['margin_qp']['nonexp_fail'] == 0 and R['margin_qp']['feas_viol'] < 1e-6
print('ALL CHECKS PASSED')
