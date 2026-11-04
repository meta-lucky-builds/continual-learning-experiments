"""Choose dlr x dep for the sleep arms on seed 100 (sleep_plan.md): final reserve accuracy of ER+bc_buf, reserve held out of all training.
Run: RESV=20 python pick_sleep.py results/sleep_tune  -> writes results/sleep_tune/choice.json"""
import sys, os, json, glob, numpy as np
d = sys.argv[1]; rows = []
for f in sorted(glob.glob(os.path.join(d, 'tune_*_100.npz'))):
    z = np.load(f); L = z['e4_resv_logits'] - z['e4_b_buf']; acc = float((L.argmax(1) == z['resv_y']).mean())
    lr_, dep = os.path.basename(f)[5:-8].split('_')  # tune_<dlr>_<dep>_100.npz
    cfg = json.loads([l for l in open(f[:-4] + '.log') if l.startswith('{')][-1])['cfg']  # cross-check settings against the filename
    assert cfg == {'sleep': 1, 'rank': 16, 'dream': 'buf', 'dlr': float(lr_), 'dep': int(dep)}, (f, cfg)
    rows.append((acc, float(lr_), int(dep))); print(os.path.basename(f), 'reserve acc', acc, 'n', len(z['resv_y']))
assert len(rows) >= 4 and all(len(np.load(f)['resv_y']) == 200 for f in glob.glob(os.path.join(d, 'tune_*_100.npz')))
best = sorted(rows, key=lambda r: (-r[0], r[2], r[1]))[0]
json.dump({'dlr': best[1], 'dep': best[2], 'reserve_acc': best[0], 'all': rows}, open(os.path.join(d, 'choice.json'), 'w')); print('CHOICE', best)
