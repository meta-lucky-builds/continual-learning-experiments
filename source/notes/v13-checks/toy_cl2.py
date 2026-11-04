"""Small nonlinear class-incremental toy: does the margin-constrained step help beyond ER?
Synthetic data only. Harness checks: disjoint splits by id, test-never-trained assertion,
oracle/joint upper bound and finetune lower bound reported, fixed seeds, per-method identical streams."""
import torch, math, json, sys, time
import numpy as np
from scipy.optimize import minimize
torch.set_num_threads(4)
import os
SEP = float(os.environ.get("SEP", "1.6")); NMEM = int(os.environ.get("NMEM", "20")); REP = int(os.environ.get("REP", "16"))
D_IN, HID, K, TASKS = 20, 64, 10, 5
PER_CLASS_TRAIN, PER_CLASS_TEST = 300, 200

def make_data(seed):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(K, D_IN, generator=g) * SEP
    mix = torch.randn(D_IN, D_IN, generator=g) / math.sqrt(D_IN)   # shared correlated structure
    def sample(n, c): return (means[c] + torch.randn(n, D_IN, generator=g)) @ (torch.eye(D_IN) + mix)
    Xtr, ytr, Xte, yte = [], [], [], []
    for c in range(K):
        Xtr.append(sample(PER_CLASS_TRAIN, c)); ytr += [c] * PER_CLASS_TRAIN
        Xte.append(sample(PER_CLASS_TEST, c)); yte += [c] * PER_CLASS_TEST
    Xtr, Xte = torch.cat(Xtr), torch.cat(Xte)
    return Xtr, torch.tensor(ytr), Xte, torch.tensor(yte)

def model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(D_IN, HID), torch.nn.ReLU(), torch.nn.Linear(HID, HID),
                               torch.nn.ReLU(), torch.nn.Linear(HID, K))

def flat(m): return torch.cat([p.reshape(-1) for p in m.parameters()])
def set_flat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n

def margin_grads(m, X, y, k):
    """Per-item gradients of (f_y - f_j) for the k currently strongest rivals j; returns (N_P x n*k) and margins."""
    params = {n: p.detach() for n, p in m.named_parameters()}
    f = torch.func.functional_call
    def logit_fn(pr, x): return f(m, pr, (x.unsqueeze(0),)).squeeze(0)
    with torch.no_grad(): L = m(X)
    cols, mars = [], []
    jac = torch.func.vmap(torch.func.jacrev(logit_fn), in_dims=(None, 0))(params, X)   # dict: name -> (n,K,*shape)
    Jf = torch.cat([jac[n].reshape(X.shape[0], K, -1) for n, _ in m.named_parameters()], 2)  # (n,K,N_P)
    for i in range(X.shape[0]):
        Li = L[i].clone(); yi = int(y[i]); Li[yi] = -1e9
        riv = torch.topk(Li, k).indices
        for j in riv:
            cols.append(Jf[i, yi] - Jf[i, int(j)]); mars.append(float(L[i, yi] - L[i, int(j)]))
    return torch.stack(cols, 1).double().numpy(), np.array(mars)

def qp_project(Bd, C, d0, s):
    G = C.T @ (Bd[:, None] * C); r = C.T @ d0 + s
    if np.all(r >= 0): return d0                       # already feasible
    res = minimize(lambda mu: (.5 * mu @ G @ mu + mu @ r, G @ mu + r), np.zeros(len(s)), jac=True,
                   bounds=[(0, None)] * len(s), method='L-BFGS-B', options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 2000})
    return d0 + Bd * (C @ res.x)

def run(method, seed, n_mem=None, steps_per_task=300, bs=16, rep=None, lr=1e-3, k_riv=3, n_pool=32, kappa=0.2, mfloor=0.5):
    n_mem = NMEM if n_mem is None else n_mem; rep = REP if rep is None else rep
    Xtr, ytr, Xte, yte = make_data(seed)
    m = model(seed + 1000); rng = np.random.default_rng(seed)
    beta1, beta2, eps = .9, .999, 1e-8
    Np = flat(m).numel(); mvec = torch.zeros(Np, dtype=torch.float64); vvec = torch.zeros(Np, dtype=torch.float64); t = 0
    buf = []                                               # indices into training set (replay set)
    cbuf = []                                              # constraint-only set (margin_split): never trained on
    trained_ids = set()
    for task in range(TASKS):
        cls = [2 * task, 2 * task + 1]
        idx = np.where(np.isin(ytr.numpy(), cls))[0]
        if method == 'joint':                              # upper bound: all classes so far, iid
            idx = np.where(ytr.numpy() < 2 * task + 2)[0]
        for st in range(steps_per_task):
            b = rng.choice(idx, bs, replace=False); trained_ids.update(b.tolist())
            X, y = Xtr[b], ytr[b]
            if method not in ('finetune', 'joint') and len(buf) > 0:
                rb = rng.choice(buf, min(rep, len(buf)), replace=False); X = torch.cat([X, Xtr[rb]]); y = torch.cat([y, ytr[rb]])
            m.zero_grad(); loss = torch.nn.functional.cross_entropy(m(X), y); loss.backward()
            g = torch.cat([p.grad.reshape(-1) for p in m.parameters()]).double(); t += 1
            mvec = beta1 * mvec + (1 - beta1) * g; vvec = beta2 * vvec + (1 - beta2) * g * g
            mh = mvec / (1 - beta1 ** t); vh = vvec / (1 - beta2 ** t)
            Bd = (lr / (vh.sqrt() + eps)).numpy(); d0 = -Bd * mh.numpy()
            d = d0
            if method in ('margin_qp', 'gem_task', 'margin_split') and len(buf) > 0:
                src = cbuf if method == 'margin_split' else buf
                pool = rng.choice(src, min(n_pool, len(src)), replace=False)
                if method in ('margin_qp', 'margin_split'):
                    C, mars = margin_grads(m, Xtr[pool], ytr[pool], k_riv)
                    s = np.where(mars > 0, kappa * np.maximum(mars - mfloor, 0), np.inf)    # headroom slack; wrong items unconstrained
                    keep = np.isfinite(s); C, s = C[:, keep], s[keep]
                else:   # GEM: one constraint, negative mean memory loss gradient
                    m.zero_grad(); torch.nn.functional.cross_entropy(m(Xtr[pool]), ytr[pool]).backward()
                    C = -torch.cat([p.grad.reshape(-1) for p in m.parameters()]).double().numpy()[:, None]; s = np.zeros(1)
                if C.shape[1] > 0:
                    d = qp_project(Bd, C, d0, s)
                    assert mh.numpy() @ d <= 1e-12, 'surrogate descent violated'
            set_flat(m, flat(m).detach().double().add(torch.from_numpy(d)).float())
        # end of task: add memory items (reservoir-free: n_mem per new class)
        for c in cls:
            ci = np.where(ytr.numpy() == c)[0]; pick = rng.choice(ci, n_mem, replace=False).tolist()
            if method in ('margin_split', 'er_half'):
                h = n_mem // 2; buf += pick[:h]
                if method == 'margin_split': cbuf += pick[h:]
            else: buf += pick
    with torch.no_grad():
        pred = m(Xte).argmax(1)
        bacc = (m(Xtr[buf]).argmax(1) == ytr[buf]).float().mean().item()
    acc_per_class = [(pred[yte == c] == c).float().mean().item() for c in range(K)]
    return float(np.mean(acc_per_class)), bacc

if __name__ == '__main__':
    # harness self-checks
    Xtr, ytr, Xte, yte = make_data(0)
    assert Xtr.shape[0] == K * PER_CLASS_TRAIN and Xte.shape[0] == K * PER_CLASS_TEST
    assert not torch.isclose(Xtr[:, None, :3], Xte[None, :50, :3]).all(-1).any(), 'train/test overlap'
    methods = sys.argv[1].split(',') if len(sys.argv) > 1 else ['finetune', 'er', 'gem_task', 'margin_qp', 'joint']
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    out = {}
    for meth in methods:
        t0 = time.time(); rr = [run(meth, s) for s in seeds]; accs = [r[0] for r in rr]
        out[meth] = {'mean': round(float(np.mean(accs)), 4), 'per_seed': [round(a, 4) for a in accs],
                     'replay_buffer_acc': round(float(np.mean([r[1] for r in rr])), 3), 'sec': round(time.time() - t0)}
        print(meth, out[meth], flush=True)
    json.dump(out, open('toy_results.json', 'w'), indent=1)
