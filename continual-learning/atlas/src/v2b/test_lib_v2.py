import numpy as np, sys; sys.path.insert(0, '.')
from atlas_lib import *
rng = np.random.default_rng(5); fails = []
def check(n, c, i=''):
    print('PASS' if c else 'FAIL', n, i)
    if not c: fails.append(n)
d, C = 6, 5; M = rng.normal(size=(C, d)) * 2; A = rng.normal(size=(d, d)); Sig = A @ A.T + np.eye(d); X = rng.normal(size=(200, d))
check('scores_argmax_eq_predict', np.array_equal(lda_scores(X, M, Sig).argmax(1), lda_predict(X, M, Sig)))
L = log_softmax(rng.normal(size=(50, 7)) * 30)
check('log_softmax_normalised', np.allclose(np.exp(L).sum(1), 1) and np.isfinite(L).all())
# temperature: data generated from softmax(S/2) -> fitted tau near 2
S = rng.normal(size=(20000, 10)) * 3; P = np.exp(log_softmax(S / 2.0)); yi = np.array([rng.choice(10, p=p) for p in P])
t = fit_temperature(S, yi); check('temperature_recovered', 1.6 < t < 2.5, f'tau={t:.2f}')
# PoE: expert 1 informative, expert 2 noise -> weight near 1; symmetric experts -> weight near 0.5
n = 3000; yi = rng.integers(0, 10, n)
def expert(q):
    S = rng.normal(size=(n, 10)); S[np.arange(n), yi] += q; return log_softmax(S)
w, _ = fit_poe(expert(3.0), expert(0.0), yi); check('poe_prefers_informative', w >= 0.8, f'w={w}')
E1, E2 = expert(1.5), expert(1.5); w, _ = fit_poe(E1, E2, yi)
accs = (E1.argmax(1) == yi).mean(), (E2.argmax(1) == yi).mean(), ((0.5 * E1 + 0.5 * E2).argmax(1) == yi).mean()
check('poe_independent_experts_help', 0.25 <= w <= 0.75 and accs[2] > max(accs[:2]), f'w={w} accs={np.round(accs,3)}')
# epsilon mixture: exact against direct computation, and the bound holds
LA, LP = expert(1.0), expert(-5.0)
lm = eps_mixture_logp(LA, LP, 0.05)
check('mixture_exact', np.allclose(lm, np.log(0.05 * np.exp(LA) + 0.95 * np.exp(LP))))
idx = np.arange(n)
check('mixture_bound', np.all(-lm[idx, yi] <= -LA[idx, yi] - np.log(0.05) + 1e-12))
print('FAILS:', fails)
if fails: sys.exit(1)
