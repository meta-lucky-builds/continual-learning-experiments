import json
rd = lambda f: open(f).read().rstrip('\n')
cells = []
def md(s): cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n')))
def code(s): cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip('\n')))

md(r'''
# Atlas tests v2: stronger learner, matched bridge test, random-features PoE, equal bytes

**How to run:** Runtime → Change runtime type → **T4 GPU** → Runtime → **Run all**. Expect about 1.5 to 2 hours. Keep the tab open.
At the end `atlas_v2b_results.json` downloads. Send it back. If any cell fails, paste the error.

**Why v2 exists (from the v1 results):**
- v1's learner was too weak. Its head got 8.1% vs ewt's 17.4%, and its learned features with perfect means (25.1%) lost to fixed random features (35.1%).
  - So v2 first **calibrates the learner** on a separate seed and picks it using held-out control items, never the test set.
- v1 compared a 10-per-class reserve against a ~20-per-class buffer. That mixed up *how many* items with *whether they were trained on*.
  - v2 adds a **matched-size** test with the same count per class from each source.
- v1 never combined the random-features expert with the learned one. v2 adds the **product of experts (PoE)** and the **ε-mixture**, and checks the mixture's log-loss bound on every test item.
- v1's "equal bytes" config spent half the memory on a reserve. v2's config **E** uses 1800 buffer + 2 control items per class = 2000 images, like ewt's ER baseline.

**Honesty notes**
- These are fresh runs of my re-implementation, not ewt's T-S5 runs. Comparisons inside the notebook are clean; absolute numbers are not comparable to ewt's.
- The random chart's exact statistics are not free. A 1024×1024 covariance plus 100 means is about 4.6 MB in float32, which is roughly 1500 CIFAR images' worth. So config E is equal in **images**, not in total bytes. This is reported, not hidden.
- Every learned arm uses the same LDA covariance within a config. Temperatures, PoE weights and the learner choice are all fitted on **control** items. The test set is used only for the final numbers.
''')
code(r'''
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # before CUDA starts
import sys, json, time, gc, importlib, subprocess, platform
import numpy as np, torch
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
import warnings; warnings.filterwarnings('once')
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print('torch', torch.__version__, '| numpy', np.__version__, '| python', platform.python_version(), '| GPU', GPU)
if GPU is None:
    print('WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU. On CPU this takes many hours.')
VERSION = 'atlas-tests-v2b'
''')
md(r'''
## Pre-registration (written before any v2 result)

**Learner calibration** (seed 100, which is not one of the test seeds). Six candidates:
- v1's recipe;
- momentum;
- several gradient steps per incoming batch;
- flip + crop augmentation (training only);
- a small learning-rate range.

The pick is the candidate with the highest **learned-feature ceiling on control items** (LDA with true means). The gate "learned ceiling > random-chart LDA on control" is reported. If the gate fails, the learned-arm rankings are provisional.

**Configs** (5 seeds each, 20 tasks × 5 classes, single online pass):
- **A**: buffer 2000 + reserve 20/class (10 bridge, 10 control). Σ comes from reserve items.
- **E**: buffer 1800 + 2 control/class = 2000 images. No bridge. Σ comes from the buffer plus control.

**Arms.**
- `O_full`: oracle means.
- `B_only_buf`: buffer means.
- `CH_cv_buf`: X2 global map.
- `GR_prev_buf`: per-class GREG on the previous chart.
- `GR_multi_buf`: GREG with [r, previous].
- `GR_stable_buf`: one hop from r.
- Matched pairs (A only), with the same k per class at every hop: `GR_prev_res_m` vs `GR_prev_buf_m`, and `B_only_res_m` vs `B_only_buf_m`.
- `RC_LDA`: the random chart with exact statistics.
- `PoE[RC+x]`: w·log p_RC + (1−w)·log p_x, with w and temperatures fitted on control.
- `Mix[RC,PoE+x]`: 0.05·p_RC + 0.95·p_PoE.

**Comparisons (rule: |mean| > 2·SE and 5/5 seeds same sign; not significant ≠ tie). My predictions, with confidence:**

| | Comparison | Prediction |
|---|---|---|
| G1 | A: O_full > RC_LDA | yes, 60% |
| P1 | A: GR_prev_buf > CH_cv_buf | yes, 80% |
| P2 | A: GR_multi_buf > GR_prev_buf | yes, 40% |
| P3 | A: GR_prev_res_m > GR_prev_buf_m (training bias is real) | yes, 35% |
| P3b | A: B_only_res_m > B_only_buf_m | yes, 40% |
| P4 | A: PoE[RC+GR_prev_buf] > RC_LDA | yes, 80% |
| P4b | A: PoE[RC+GR_prev_buf] > GR_prev_buf | yes, 85% |
| P5 | E: PoE[RC+GR_prev_buf] > CH_cv_buf | yes, 85% |
| P5b | E: PoE[RC+GR_prev_buf] > RC_LDA | yes, 70% |
| P6 | A: CH_cv_buf > B_only_buf (X2 replication) | yes, 40% |
| P7 | A: PoE[RC+O_full] > PoE[RC+GR_prev_buf] (migration loss remains) | yes, 70% |

**Hard check:** the ε-mixture bound −log p ≤ −log p_RC − log ε must hold on 100% of test items. It is a mathematical identity, so any violation means a code bug.

**Edge cases and what happens**

| What if | Handling |
|---|---|
| A learner candidate diverges (NaN loss) | It is marked failed in calibration and cannot be picked. |
| The picked learner diverges on a test seed | The run is recorded as an error and excluded. Comparisons use only the seeds every arm has. |
| The learner still loses to random features | The gate is printed. The learned rankings are labelled provisional; the PoE results still stand. |
| Tuning leaks into the test set | It can't: the learner is picked on seed 100 using control items; temperatures and PoE weights use control items; test is read once. |
| A class has no items in the buffer at some hop | It falls back to the global map, and the fallback is counted and reported. |
| The matched test isn't really matched | Each class gets the same k from each source at every hop by construction; the mean k is printed. |
| Colab disconnects | Calibration and every run are saved as they finish. Run all again and finished work is skipped. |
| A results file from another version | It is detected by the version and learner settings and not mixed in. |
| Many comparisons | There are 11. Under no effect, a 5/5 same-sign result has probability 1/16 per test, before the 2·SE requirement. Expect at most one false positive. Look at sizes, not just verdicts. |
| Augmentation contaminates features | Augmentation is training only. All features are extracted from clean images. |
''')
md('## 1. Code (pure-numpy analysis; PyTorch learner)')
for f in ('atlas_lib.py', 'atlas_analysis.py', 'atlas_summary.py', 'atlas_torch.py'):
    code(f'%%writefile {f}\n' + rd(f))
code(r'''
sys.path.insert(0, '.')
import atlas_lib, atlas_analysis, atlas_summary, atlas_torch
for m in (atlas_lib, atlas_analysis, atlas_summary, atlas_torch): importlib.reload(m)
from atlas_analysis import run_analysis
from atlas_summary import summarize
from atlas_torch import load_cifar100, run_one, rc_control_acc, Diverged
''')
md('## 2. Checks with known answers (the notebook stops if anything fails)')
for f in ('test_lib.py', 'test_lib_v2.py', 'test_analysis.py', 'sk_check.py', 'audit_tests.py'):
    code(f'%%writefile {f}\n' + rd(f))
code(r'''
def run_check(script):
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(f'===== {script} =====\n' + r.stdout + (r.stderr[-3000:] if r.returncode else ''))
    if r.returncode != 0:
        raise SystemExit(f'{script} FAILED: stop here and send me this output.')
for s in ('test_lib.py', 'test_lib_v2.py', 'test_analysis.py', 'sk_check.py', 'audit_tests.py'):
    run_check(s)
print('All checks passed.')
''')
code(r'''
# GPU eigendecomposition for speed, used only if it gives the same ridge answers as numpy
def gpu_eigh(A):
    ev, V = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(A)).to('cuda', torch.float64))
    return ev.cpu().numpy(), V.cpu().numpy()
atlas_lib.EIGH = np.linalg.eigh
if torch.cuda.is_available():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 1152)); Y = rng.normal(size=(2000, 128))
    t = time.time(); B_cpu = atlas_lib.ridge_cv(X, Y, np.random.default_rng(1), split=1024)[0]; t_cpu = time.time() - t
    atlas_lib.EIGH = gpu_eigh
    t = time.time(); B_gpu = atlas_lib.ridge_cv(X, Y, np.random.default_rng(1), split=1024)[0]; t_gpu = time.time() - t
    same = np.allclose(B_cpu, B_gpu, rtol=1e-6, atol=1e-9 * np.abs(B_cpu).max())
    if not (same and t_gpu < t_cpu): atlas_lib.EIGH = np.linalg.eigh
    print(f'cpu {t_cpu:.1f}s, gpu {t_gpu:.1f}s, identical={same} -> using', 'GPU' if atlas_lib.EIGH is gpu_eigh else 'CPU')
''')
md('## 3. Data and smoke tests (every code path, small and fast)')
code(r'''
data = load_cifar100()
SM = dict(n_classes=15, n_tasks=3, lr=0.03, momentum=0.9, batch=10, iters=2, aug=True, stream_per_class=60)
SMOKE_A = dict(SM, M=200, bridge_per_class=10, control_per_class=10, sigma_from='reserve')
SMOKE_E = dict(SM, M=180, bridge_per_class=0, control_per_class=2, sigma_from='buffer')
for name, cfg in (('A', SMOKE_A), ('E', SMOKE_E)):
    D = run_one(0, cfg, data)
    o = run_analysis(D, 0)
    assert all(np.isfinite(v) and 0 <= v <= 100 for v in o['acc'].values()), o['acc']
    assert sum(o['mixture_bound_violations'].values()) == 0
    assert ('GR_prev_res_m' in o['acc']) == (name == 'A')
    print(f'smoke {name} ok:', {k: round(v, 1) for k, v in o['acc'].items()})
    if name == 'A':
        D2 = run_one(0, cfg, data, log=lambda *a: None)
        dmax = float(np.abs(D['F'][-1] - D2['F'][-1]).max())
        print('determinism: max feature difference between identical runs =', dmax,
              '(OK)' if dmax < 1e-3 else '(WARNING: not bit-reproducible on this GPU; still valid as a 5-seed statistic)')
        del D2
    del D; gc.collect()
c = run_one(0, SMOKE_A, data, full=False); assert 0 <= c['oracle_ctl'] <= 100
print('smoke calibration path ok; random-chart control acc', round(rc_control_acc(0, SMOKE_A, data), 1))
''')
md('## 4. Learner calibration (seed 100, control items only)')
code(r'''
BASE_A = dict(n_classes=100, n_tasks=20, batch=10, stream_per_class=None, M=2000, bridge_per_class=10, control_per_class=10, sigma_from='reserve')
CANDIDATES = {
    'L0_v1':        dict(lr=0.03, momentum=0.0, iters=1, aug=False),
    'L1_mom':       dict(lr=0.03, momentum=0.9, iters=1, aug=False),
    'L2_mom_it3':   dict(lr=0.01, momentum=0.9, iters=3, aug=True),
    'L3_mom_it3_hi': dict(lr=0.03, momentum=0.9, iters=3, aug=True),
    'L4_mom_it5':   dict(lr=0.01, momentum=0.9, iters=5, aug=True),
    'L5_sgd_it3':   dict(lr=0.1, momentum=0.0, iters=3, aug=True),
}
CAL_SEED = 100
CAL_PATH = 'atlas_v2b_calibration.json'
cal = json.load(open(CAL_PATH)) if os.path.exists(CAL_PATH) else {}
if cal.get('version') != VERSION: cal = {'version': VERSION, 'runs': {}}
if 'rc_ctl' not in cal:
    cal['rc_ctl'] = rc_control_acc(CAL_SEED, BASE_A, data)
print(f'random-chart LDA on control items: {cal["rc_ctl"]:.1f}  (the bar)')
for name, lc in CANDIDATES.items():
    if name in cal['runs']: print(name, 'done:', cal['runs'][name]); continue
    print(name, lc)
    try:
        cal['runs'][name] = run_one(CAL_SEED, dict(BASE_A, **lc), data, full=False)
    except Diverged as e:
        cal['runs'][name] = {'error': str(e)}; print('   diverged:', e)
    json.dump(cal, open(CAL_PATH, 'w'))
ok = {k: v for k, v in cal['runs'].items() if 'error' not in v}
assert ok, 'every learner candidate diverged: send me this output'
PICK = max(ok, key=lambda k: (ok[k]['oracle_ctl'], ok[k]['head_ctl']))
LEARNER = CANDIDATES[PICK]
GATE = ok[PICK]['oracle_ctl'] > cal['rc_ctl']
print(f'\nPICKED {PICK} {LEARNER}: learned ceiling {ok[PICK]["oracle_ctl"]:.1f} vs random chart {cal["rc_ctl"]:.1f} -> gate',
      'PASSED' if GATE else 'FAILED (learned-arm rankings will be provisional)')
''')
md('## 5. The experiment (configs A and E × 5 seeds). Saved after every run.')
code(r'''
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = {'A': dict(BASE_A, **LEARNER),
           'E': dict(BASE_A, **LEARNER, M=1800, bridge_per_class=0, control_per_class=2, sigma_from='buffer')}
RESULTS_PATH = 'atlas_v2b_partial.json'
results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
if results.get('_version') != VERSION or results.get('_learner') != PICK:
    results = {'_version': VERSION, '_learner': PICK}
t_all = time.time()
for cname, cfg in CONFIGS.items():
    results.setdefault(cname, {})
    for seed in SEEDS:
        if str(seed) in results[cname]: print(f'{cname} seed {seed}: done, skipping'); continue
        print(f'{cname} seed {seed}: running ...')
        try:
            D = run_one(seed, cfg, data)
        except Diverged as e:
            results[cname][str(seed)] = {'error': str(e)}; print('   diverged:', e)
            json.dump(results, open(RESULTS_PATH, 'w')); continue
        out = run_analysis(D, seed); out['config'] = cfg
        results[cname][str(seed)] = out
        del D; gc.collect()
        json.dump(results, open(RESULTS_PATH, 'w'))
        a = out['acc']
        print(f'   analysis {out["seconds"]:.0f}s | O_full {a["O_full"]:.1f} | RC {a["RC_LDA"]:.1f} | GR_prev_buf {a["GR_prev_buf"]:.1f}'
              f' | PoE {a["PoE[RC+GR_prev_buf]"]:.1f} | CH_cv_buf {a["CH_cv_buf"]:.1f} | head {a["ER_head"]:.1f} | {(time.time()-t_all)/60:.0f} min')
''')
md('## 6. Results and pre-registered comparisons')
code(r'''
runs = {k: v for k, v in results.items() if not k.startswith('_')}
print(f'Learner: {PICK} {LEARNER} | gate {"PASSED" if GATE else "FAILED"} (learned ceiling {ok[PICK]["oracle_ctl"]:.1f} vs random chart {cal["rc_ctl"]:.1f} on control)')
print('Calibration:', {k: (v.get('oracle_ctl'), v.get('head_ctl')) if 'error' not in v else 'diverged' for k, v in cal['runs'].items()})
summary = summarize(runs)
final = dict(meta=dict(version=VERSION, torch=torch.__version__, numpy=np.__version__, gpu=GPU, seeds=SEEDS, configs=CONFIGS,
                       learner=PICK, gate=bool(GATE), calibration=cal, finished=time.strftime('%Y-%m-%d %H:%M:%S')),
             summary=summary, runs=runs)
json.dump(final, open('atlas_v2b_results.json', 'w'), indent=1)
try:
    from google.colab import files; files.download('atlas_v2b_results.json')
except Exception as e:
    print('Download atlas_v2b_results.json from the Files panel on the left.', e)
''')
nb = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='T4'),
          kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')),
          nbformat=4, nbformat_minor=0)
for c in nb['cells']:
    c['source'] = [l + '\n' for l in c['source'].split('\n')]; c['source'][-1] = c['source'][-1].rstrip('\n')
json.dump(nb, open('atlas_v2b_tests.ipynb', 'w'), indent=1)
print(len(cells), 'cells')
