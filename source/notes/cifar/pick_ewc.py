"""Pick EWC lambda on seed 100 by MIX_half final accuracy with seed 100's RC expert (plan.md step 4). RESV=20."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar as C
from analyze_e2 import per_seed
_, _, _, yte = C.load(); R = 'results/r2'; out = {}
for lam in ['1', '10', '100', '1000']:
    if not os.path.exists(f'{R}/ewc_l{lam}_100.npz'): print(lam, 'run failed (see .err); excluded'); continue
    r = per_seed(None, 100, yte, f'{R}/ewc_l{lam}_100.npz', f'{R}/rc_100.npz')['acc'][C.NTASK - 1]
    out[lam] = {k: r[k] for k in ['ER+bc_buf', 'MIX_half', 'MIX_what_ahat']}; print(lam, out[lam])
best = max(out, key=lambda l: out[l]['MIX_half']); json.dump({'lam': float(best), 'grid': out}, open(f'{R}/ewc_lambda.json', 'w'), indent=1); print('chosen', best)
