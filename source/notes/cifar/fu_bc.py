"""Diagnostic (test-set sweeps are oracle, not streaming-valid): is buffer bias correction over-correcting the newest task?"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import cifar as C
from analyze_cifar import softmax
R = 'results/e1'; _, _, _, yte = C.load(); y = yte.numpy(); rows = []
for s in range(5):
    er = np.load(f'{R}/er_lr0.03_{s}.npz'); tasks = er['tasks'].tolist(); L = er['e4_test_logits'].astype(np.float64); ry = er['resv_y']
    b = er['e4_b_buf']; br = C.balanced_mean_logit(torch.from_numpy(er['e4_resv_logits'].astype(np.float64)), torch.from_numpy(ry)).numpy()
    newc = tasks[4]; isnew = np.isin(y, newc); r = {}
    for name, LL in [('raw', L), ('bc_buf', L - b), ('bc_resv', L - br)]:
        p = LL.argmax(1); r[name] = (100 * (p == y).mean(), 100 * np.isin(p, newc).mean(), 100 * (p == y)[isnew].mean(), 100 * (p == y)[~isnew].mean())
    r['alpha'] = {a: 100 * ((L - a * b).argmax(1) == y).mean() for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    # buffer vs reserve mean logit gap for newest classes (memorisation signature): b - br on newest-class logits
    r['gap_new'] = float((b - br)[newc].mean()); r['gap_old'] = float((b - br)[[c for c in range(10) if c not in newc]].mean())
    # per-class mean logit on TEST (oracle balanced) vs buffer
    bt = C.balanced_mean_logit(torch.from_numpy(L), torch.from_numpy(y)).numpy(); r['gap_test_new'] = float((b - bt)[newc].mean()); r['gap_test_old'] = float((b - bt)[[c for c in range(10) if c not in newc]].mean())
    r['bc_test'] = 100 * ((L - bt).argmax(1) == y).mean()
    rows.append(r)
for k in ['raw', 'bc_buf', 'bc_resv']:
    print(k, 'acc, %pred newest (true 20), acc newest, acc old:', np.round(np.mean([r[k] for r in rows], 0), 1))
print('alpha*b_buf (oracle sweep):', {a: round(np.mean([r['alpha'][a] for r in rows]), 1) for a in rows[0]['alpha']})
print('b_buf - b_resv  newest classes %.2f  old %.2f' % (np.mean([r['gap_new'] for r in rows]), np.mean([r['gap_old'] for r in rows])))
print('b_buf - b_test  newest classes %.2f  old %.2f   (test-mean bc, oracle: %.1f)' % (np.mean([r['gap_test_new'] for r in rows]), np.mean([r['gap_test_old'] for r in rows]), np.mean([r['bc_test'] for r in rows])))
print('per-seed gap_test_new:', [round(r['gap_test_new'], 2) for r in rows])
