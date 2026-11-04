"""Main runs. Hyperparameters tau were tuned on seed 100 (n_dis=40) only; test seeds 0-4."""
import os, sys, json, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pkmsleep import run, summarize
CONDS = {'single': {}, 'single2N': {}, 'oracle': {}, 'count_gate': {}, 'none': {},
         'copy_all': {}, 'copy_sel': {'tau': 0.4}, 'copy_soft': {'tau': 3.0}, 'replay': {}, 'replay_all': {}}
SETTINGS = {'dis0': {'n_dis': 0}, 'dis40': {'n_dis': 40}, 'dis100': {'n_dis': 100}}
SEEDS = [0, 1, 2, 3, 4]
def job(a):
    st, c, sd = a
    M, C = run(c, sd, **SETTINGS[st], **CONDS[c])
    s = summarize(M); s['final_cos'] = float(np.nanmean(C[-1]))
    return st, c, sd, {k: (float(v) if not isinstance(v, list) else v) for k, v in s.items()}
if __name__ == '__main__':
    jobs = [(st, c, sd) for st in SETTINGS for c in CONDS for sd in SEEDS]
    with Pool(4) as p: res = p.map(job, jobs)
    out = {}
    for st, c, sd, s in res: out.setdefault(st, {}).setdefault(c, {})[sd] = s
    json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    for st in SETTINGS:
        print('==', st)
        for c in CONDS:
            r = out[st][c]; f = np.array([r[s]['final'] for s in SEEDS]); a = np.array([r[s]['acq'] for s in SEEDS])
            h = np.array([r[s]['half_life'] for s in SEEDS])
            print(f"{c:11s} final {f.mean():.3f}±{f.std(ddof=1):.3f}  acq {a.mean():.3f}  half-life {h.mean():.0f} (min {h.min()})")
