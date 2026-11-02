"""Self-tests for m2c=2 (past-task-only pool), added 2026-10-29 before any m2c=2 result. Run: RESV=20 LR=0.03 python m2c2_check.py"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C, cifar2 as K
ok = True
def chk(c, name, *info):
    global ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
D = C.load(); tasks = C.make_stream(D[1], 0)[0]; tb0 = sum(1 for t, _ in C.batches(*C.make_stream(D[1], 0)[1:]) if t == 0)
seen = []; orig = K.margin_cols
def spy(net, X, y, extra): seen.append((y.tolist(), list(extra))); return orig(net, X, y, extra)
K.margin_cols = spy
er = K.run(0, D, max_steps=tb0 + 60, eval_tasks=False); m2 = K.run(0, D, max_steps=tb0 + 60, eval_tasks=False, m2c=2)
n0 = tb0 * C.B
chk(np.array_equal(er['preq'][:n0 + C.B], m2['preq'][:n0 + C.B]), 'M1 m2c=2 equals ER through task 1 (no past items to protect)')
chk(m2['stats']['proj_steps'] > 0 and not np.array_equal(er['preq'][n0 + C.B:], m2['preq'][n0 + C.B:]) and m2['stats']['max_violation'] < 1e-6,
    'M2 active and feasible in task 2', m2['stats']['proj_steps'], m2['stats']['max_violation'])
chk(len(seen) > 0 and all(all(l in tasks[0] for l in ys) and sorted(ex) == sorted(tasks[1]) for ys, ex in seen), 'M3 pool labels are past-task only; rivals include current classes', len(seen))
seen.clear(); m1 = K.run(0, D, max_steps=60, eval_tasks=False, m2c=1)
chk(m1['stats']['proj_steps'] > 0 and len(seen) > 0, 'M4 m2c=1 unchanged: still active within task 1', m1['stats']['proj_steps'])
print('CHECKS', 'ALL PASS' if ok else 'FAILURES'); sys.exit(0 if ok else 1)
