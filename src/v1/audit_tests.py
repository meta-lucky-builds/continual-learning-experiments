# ===== Audit design tests (synthetic, known answers). Each checks a property ARCH-v2 relies on. =====
import numpy as np, sys; sys.path.insert(0, '.')
from atlas_lib import greg_means, global_affine_means, lda_predict, pooled_cov
rng = np.random.default_rng(123)
afails = []
def acheck(n, c, i=''):
    print('PASS' if c else 'FAIL', n, i)
    if not c: afails.append(n)

# U1 Re-anchoring does not reset inherited error: an exact hop carries the old error forward as delta @ W.
d, C = 8, 6
mu0 = {c: rng.normal(size=d) * 3 for c in range(C)}
W = rng.normal(size=(d, d)); b = rng.normal(size=d)
yb = np.repeat(np.arange(C), 30); A = np.stack([mu0[c] for c in yb]) + rng.normal(size=(len(yb), d)); Y = A @ W + b
delta = {c: rng.normal(size=d) * 0.5 for c in range(C)}
carried = {c: mu0[c] + delta[c] for c in range(C)}
est, _ = greg_means(list(range(C)), carried, A, Y, yb, rng)
err_after = np.array([np.linalg.norm(est[c] - (mu0[c] @ W + b)) for c in range(C)])
pred_err = np.array([np.linalg.norm(delta[c] @ W) for c in range(C)])
acheck('U1_inherited_error_persists', np.allclose(err_after, pred_err, rtol=1e-3) and err_after.min() > 0.1,
       f'after={err_after.round(3)} predicted={pred_err.round(3)}')

# U2 A rotation (no shift) is not fixed by a mean shift; a full linear transport fixes it.
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
Yr = A @ Q
shift = {c: carried_c + (Yr.mean(0) - A.mean(0)) for c, carried_c in mu0.items()}
full = global_affine_means(list(range(C)), mu0, A, Yr, yb, rng)
e_shift = max(np.linalg.norm(shift[c] - mu0[c] @ Q) for c in range(C)); e_full = max(np.linalg.norm(full[c] - mu0[c] @ Q) for c in range(C))
acheck('U2_rotation_needs_full_transport', e_shift > 1.0 and e_full < 1e-3, f'shift_err={e_shift:.2f} full_err={e_full:.1e}')

# U3 Reused evidence: fusing two estimates built from the SAME items as if independent halves the claimed
#    variance but not the real one. The design rule (never count reused evidence as new) is needed.
n, trials = 20, 4000
fused = []
for _ in range(trials):
    x = rng.normal(size=n); e1 = x.mean(); e2 = x.mean() + 0.0  # second "source" re-reads the same items
    fused.append((e1 + e2) / 2)
claimed = 1 / (n + n); real = np.var(fused)
acheck('U3_reused_evidence_overconfident', real / claimed > 1.8, f'real/claimed variance={real/claimed:.2f} (expect 2)')

# U4 A dormant class whose items fall inside active classes' support is still retrieved by the stable chart.
C2, dd = 50, 32
M = rng.normal(size=(C2, dd)) * 0.6       # crowded: classes overlap
Sig = np.eye(dd)
dormant = np.arange(10, 50)
Xq = np.repeat(dormant, 40); Zq = M[Xq] + rng.normal(size=(len(Xq), dd))
P = np.linalg.solve(Sig, M.T); S = Zq @ P - 0.5 * np.einsum('dc,cd->c', P, M)
top5 = np.argsort(-S, 1)[:, :5]; recall = (top5 == Xq[:, None]).any(1).mean()
acheck('U4_dormant_retrieved_top5', recall > 0.5, f'top-5 recall={recall:.2f} (chance 0.10)')

# U5 A label-only change (classes 0 and 1 swap labels) is caught by prequential error, not by input novelty.
C3 = 10; M3 = rng.normal(size=(C3, dd)) * 3
def novelty(Z): return np.min(((Z[:, None, :] - M3[None]) ** 2).sum(-1), 1)
lab = rng.integers(0, C3, 4000); Z = M3[lab] + rng.normal(size=(4000, dd))
before, after = slice(0, 2000), slice(2000, 4000)
obs = lab.copy(); sw = (np.arange(4000) >= 2000) & np.isin(lab, [0, 1]); obs[sw] = 1 - lab[sw]
pred = lda_predict(Z, M3, np.eye(dd))
err_b = (pred[before] != obs[before])[np.isin(obs[before], [0, 1])].mean(); err_a = (pred[after] != obs[after])[np.isin(obs[after], [0, 1])].mean()
nov = novelty(Z); dn = abs(nov[after].mean() - nov[before].mean()) / nov[before].std()
acheck('U5_label_shift_seen_by_error_not_novelty', err_a - err_b > 0.8 and dn < 0.1, f'err {err_b:.2f}->{err_a:.2f}, novelty shift={dn:.3f} sd')

# U6 The epsilon-mixture bound holds against a confidently wrong expert.
eps = 0.05; K = 20; N = 1000
y6 = rng.integers(0, K, N)
pA = rng.dirichlet(np.ones(K), N); pA[np.arange(N), y6] += 0.5; pA /= pA.sum(1, keepdims=True)
wrong = np.full((N, K), 1e-12); wrong[np.arange(N), (y6 + 1) % K] = 1; wrong /= wrong.sum(1, keepdims=True)
p = eps * pA + (1 - eps) * wrong
lhs = -np.log(p[np.arange(N), y6]); rhs = -np.log(pA[np.arange(N), y6]) - np.log(eps)
acheck('U6_eps_mixture_bound', np.all(lhs <= rhs + 1e-12) and -np.log(wrong[np.arange(N), y6]).mean() > 20,
       f'max(lhs-rhs)={np.max(lhs-rhs):.2e}, wrong-expert loss={-np.log(wrong[np.arange(N), y6]).mean():.1f}')
print('AUDIT FAILS:', afails)
if afails: sys.exit(1)
