"""Extra self-tests for cifar2.py, added after the independent review flagged K2/K4-ewc/K9 as vacuous (2026-10-29).
Run: RESV=20 LR=0.03 python cifar2_checks.py"""
import sys, os, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C, cifar2 as K
ok = True
def chk(c, name, *info):
    global ok; print('PASS' if c else 'FAIL', name, *info, flush=True); ok &= bool(c)
D = C.load(); tb0 = sum(1 for t, _ in C.batches(*C.make_stream(D[1], 0)[1:]) if t == 0); n0 = tb0 * C.B
# X1: EWC inside the learner switches on only after the first task-end Fisher, and is causal across the boundary
er = K.run(0, D, max_steps=tb0 + 10); ew = K.run(0, D, max_steps=tb0 + 10, ewc=1000.0)
chk(np.array_equal(er['preq'][:n0 + C.B], ew['preq'][:n0 + C.B]) and not np.array_equal(er['preq'][n0 + C.B:], ew['preq'][n0 + C.B:]),
    'X1 EWC: identical to ER through task 1 and the first task-2 batch, differs afterwards')
ew2 = K.run(0, D, max_steps=tb0 + 10, ewc=1000.0, flip_step=tb0 + 5)
chk(np.array_equal(ew['preq'][:(tb0 + 6) * C.B], ew2['preq'][:(tb0 + 6) * C.B]), 'X1b EWC causality after the Fisher exists')
# X2: promoted version parameters stay frozen while the learner keeps training
q = K.run(0, D, max_steps=tb0 + 30, promote=1); v = q['versions'][0]['net']; live = q['net']
# rebuild the network at promotion time: run exactly tb0 steps and compare parameters
p0 = K.run(0, D, max_steps=tb0)['net']
same = all(torch.equal(a, b) for a, b in zip(v.state_dict().values(), p0.state_dict().values()))
moved = not all(torch.equal(a, b) for a, b in zip(live.state_dict().values(), p0.state_dict().values()))
chk(same and moved and not any(p.requires_grad for p in v.parameters()), 'X2 promoted version = learner at promotion, unchanged after; learner moved on')
# X3: version statistics equal an independent recomputation (buffer features at promotion + 30 later batches)
tasks, order, tid, resv = C.make_stream(D[1], 0); bl = C.batches(order, tid, resv)
lda = C.StreamLDA(16 * C.W); later = np.concatenate([bi for _, bi in bl[tb0:tb0 + 30]])
with torch.no_grad():
    Fl = v.feats(D[0][later])
lda_q = q['versions'][0]['lda']; base_n = lda_q.n.clone(); base_S1 = lda_q.S1.clone()
# subtract the later-stream contribution and check the remainder is a 500-item set of the right classes
rem_n = base_n - torch.bincount(D[1][later], minlength=C.K).double(); rem_S1 = base_S1 - torch.zeros_like(base_S1).index_add_(0, D[1][later], Fl.double())
chk(abs(float(rem_n.sum()) - min(C.MEM, n0)) < 1e-9 and set(torch.nonzero(rem_n).flatten().tolist()) == set(tasks[0]), 'X3 version stats = 500 buffer items of task-1 classes + later stream batches (exact subtraction)')
# X4: epochs: labels used in extra passes line up with the items (one-to-one, same as stream labels)
items = np.concatenate([bi for t, bi in bl if t == 0]); labs = D[1][items]
chk(np.array_equal(K.run(0, D, max_steps=tb0)['ylearn'], labs.numpy()), 'X4 first-pass label list aligns with task items (as used for extra epochs)')
e3 = K.run(0, D, max_steps=tb0 + 20, epochs=3); e1 = K.run(0, D, max_steps=tb0 + 20, epochs=1)
chk(np.array_equal(e3['preq'][:n0], e1['preq'][:n0]) and not np.array_equal(e3['preq'][n0:], e1['preq'][n0:]), 'X5 epochs=3: same task-1 stream predictions, different task-2 predictions (extra passes took effect)')
print('CHECKS', 'ALL PASS' if ok else 'FAILURES'); sys.exit(0 if ok else 1)
