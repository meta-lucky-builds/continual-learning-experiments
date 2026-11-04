"""Check the NNLS M2c projection against the original L-BFGS-B + polish solver (2026-10-29)."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C, cifar2 as K
ok = True
def chk(c, name, *info):
    global ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
rg = np.random.default_rng(1); worst = 0
for trial in range(200):
    m = int(rg.integers(2, 60)); Cr = rg.normal(size=(500, m))
    if trial % 3 == 0: Cr[:, -1] = Cr[:, 0] + Cr[:, 1]  # dependent columns
    d0 = rg.normal(size=500); a = K.qp_project_zero(0.03, Cr, d0); b = K.qp_project_zero_nnls(0.03, Cr, d0)
    if a is None or b is None: assert a is None and b is None; continue
    worst = max(worst, np.linalg.norm(a - b) / np.linalg.norm(a)); assert (Cr.T @ (d0 + b) >= -1e-8 * np.abs(Cr.T @ d0).max()).all()
chk(worst < 1e-6, 'Q1 random instances (incl. dependent columns): NNLS correction = L-BFGS-B correction, feasible', worst)
# real instances from an M2c run: wrap the solver, compare both on every projected step
rec = []
def both(lr, Cm, d0):
    t0 = time.time(); a = K.qp_project_zero(lr, Cm, d0); t1 = time.time(); b = K.qp_project_zero_nnls(lr, Cm, d0); t2 = time.time()
    if a is not None: rec.append((np.linalg.norm(a - b) / np.linalg.norm(a), t1 - t0, t2 - t1, float((Cm.T @ (d0 + b)).min() / np.abs(Cm.T @ d0).max())))
    return a
K.QP = both; D = C.load(); K.run(0, D, max_steps=120, eval_tasks=False, m2c=1); R = np.array(rec)
chk(len(R) > 50 and R[:, 0].max() < 1e-5 and R[:, 3].min() > -1e-8, 'Q2 real CIFAR instances: NNLS = original solver, feasible', len(R), R[:, 0].max(), R[:, 3].min())
print('mean seconds per solve: L-BFGS-B %.4f  NNLS %.5f' % (R[:, 1].mean(), R[:, 2].mean()))
print('QPCHECK', 'ALL PASS' if ok else 'FAILURES')
