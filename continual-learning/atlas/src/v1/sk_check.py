# ===== Cross-check our numpy code against scikit-learn =====
import numpy as np, sys; sys.path.insert(0, '.')
import atlas_lib
from sklearn.covariance import ledoit_wolf as sk_lw
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
rng = np.random.default_rng(7)
for n, p in ((300, 40), (30, 80)):
    X = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)); Xc = X - X.mean(0)
    C1, s1 = atlas_lib.ledoit_wolf(Xc); C2, s2 = sk_lw(Xc, assume_centered=True)
    assert np.allclose(C1, C2, atol=1e-8 * np.abs(C2).max()) and abs(s1 - s2) < 1e-10, ('LW mismatch', n, p)
    Y = rng.normal(size=(n, 3)); lam = 2.5
    B1 = atlas_lib._ridge_factory(X, Y)(lam); B2 = Ridge(alpha=lam, fit_intercept=False, solver='svd').fit(X, Y).coef_.T
    assert np.allclose(B1, B2, atol=1e-6 * np.abs(B2).max()), ('ridge mismatch', n, p)
K, d = 7, 10
yl = np.repeat(np.arange(K), 60); Xl = rng.normal(size=(K, d))[yl] * 1.5 + rng.normal(size=(len(yl), d)) @ rng.normal(size=(d, d)) * 0.5
lda = LinearDiscriminantAnalysis(solver='lsqr', priors=np.ones(K) / K, store_covariance=True).fit(Xl, yl)
Xt = rng.normal(size=(2000, d)) * 2
agree = (atlas_lib.lda_predict(Xt, lda.means_, lda.covariance_) == lda.predict(Xt)).mean()
assert agree > 0.999, ('LDA mismatch', agree)
print('sklearn cross-check PASS: Ledoit-Wolf, ridge, LDA (agreement %.4f)' % agree)
