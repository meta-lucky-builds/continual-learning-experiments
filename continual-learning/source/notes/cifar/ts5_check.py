"""T-S5 gates (ts5_plan.md). Run: CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 MEM=2000 RESV=20 LR=0.03 python ts5_check.py"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C, cifar2 as K2, sleep as Z
ok = True
def chk(c, name, *info):
    global ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
assert C.K == 100 and C.NTASK == 20 and C.MEM == 2000 and C.RESV == 20
D = C.load(); Xtr, ytr, Xte, yte = D
chk(Xtr.shape == (50000, 3, 32, 32) and Xte.shape == (10000, 3, 32, 32) and (torch.bincount(ytr) == 500).all() and (torch.bincount(yte) == 100).all(), 'T5a CIFAR-100 shapes and 500/100 per class')
for s in range(5):
    tasks, order, tid, resv = C.make_stream(ytr, s); bl = C.batches(order, tid, resv); y = ytr.numpy()
    allc = sorted(c for t in tasks for c in t)
    good = (allc == list(range(100)) and all(len(t) == 5 for t in tasks) and len(order) == 50000 and len(set(order.tolist())) == 50000
            and (np.bincount(y[order[resv]], minlength=100) == 20).all() and not set(np.concatenate([b for _, b in bl]).tolist()) & set(order[resv].tolist())
            and all(np.isin(y[b], tasks[t]).all() for t, b in bl) and sum(len(b) for _, b in bl) == 48000 and [t for t, _ in bl] == sorted(t for t, _ in bl))
    chk(good, f'T5b seed {s}: 20 disjoint tasks of 5 classes, 20 reserve/class, reserve never in a batch, batches in task order, 48,000 learning items')
# smoke runs (30 steps) of every arm's code path at K=100
r = K2.run(0, D, mem=2000, promote=1, max_steps=30, eval_tasks=False); chk(r['buf_len'] == 300 and len(r['preq'][0]) == 100, 'T5c erp smoke (K=100 head, buffer fills)')
r = K2.run(0, D, mem=8836, max_steps=30, eval_tasks=False); chk(len(r['preq']) == 300, 'T5c er_mmatch smoke')
r = Z.run(0, D, sleep=1, rank=16, max_steps=30, eval_tasks=False); chk(r['net'].head.out_features == 100, 'T5c slp smoke')
e = C.run_expert('rc', 0, D, max_steps=30); chk(e['preq'].shape == (300, 100), 'T5c rc smoke')
print('CHECKS', 'ALL PASS' if ok else 'FAILURES'); sys.exit(0 if ok else 1)
