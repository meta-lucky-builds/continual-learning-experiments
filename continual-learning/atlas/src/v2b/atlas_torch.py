# ===== Learner, stable random chart, online stream, feature extraction (PyTorch) — v2 =====
import copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, torchvision
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = torch.tensor([0.5071, 0.4865, 0.4409], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762], device=DEV).view(1, 3, 1, 1)

def load_cifar100(root='/content/data'):
    tr = torchvision.datasets.CIFAR100(root, train=True, download=True)
    te = torchvision.datasets.CIFAR100(root, train=False, download=True)
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(DEV)
    Xtr, ytr, Xte, yte = to_t(tr.data), np.array(tr.targets), to_t(te.data), np.array(te.targets)
    assert Xtr.shape == (50000, 3, 32, 32) and Xte.shape == (10000, 3, 32, 32)
    assert (np.bincount(ytr) == 500).all() and (np.bincount(yte) == 100).all()
    return Xtr, ytr, Xte, yte

def to_input(xu8):
    return (xu8.float() / 255.0 - MEAN) / STD

def augment(x, g):
    """Random horizontal flip + 4-pixel reflect-pad random crop on a FLOAT batch, driven by generator g."""
    n = x.shape[0]
    flip = torch.rand(n, generator=g, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = Fn.pad(x, (4, 4, 4, 4), mode='reflect')
    off = torch.randint(0, 9, (n, 2), generator=g, device=x.device).tolist()
    return torch.stack([xp[i, :, a:a + 32, b:b + 32] for i, (a, b) in enumerate(off)])

def make_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU())

def embed(enc, x):
    return enc(x).mean((2, 3))

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = make_encoder(); self.head = nn.Linear(128, 100)
    def forward(self, x):
        return self.head(embed(self.enc, x))

class RandomChart:
    """Stable reference r: fixed random 5x5 conv (256 maps) + ReLU + 2x2 average pool -> 1024-d. Never trained."""
    def __init__(self, seed, width=256):
        g = torch.Generator().manual_seed(seed)
        self.W = (torch.randn(width, 3, 5, 5, generator=g) / np.sqrt(75.0)).to(DEV)
        self.b = (0.1 * torch.randn(width, generator=g)).to(DEV)
    def __call__(self, x):
        return Fn.adaptive_avg_pool2d(Fn.relu(Fn.conv2d(x, self.W, self.b, padding=2)), 2).flatten(1)

@torch.no_grad()
def extract(fn, Xu8, bs=2500):
    return torch.cat([fn(to_input(Xu8[i:i + bs])).float().cpu() for i in range(0, len(Xu8), bs)]).numpy()

class Diverged(RuntimeError):
    pass

def split_data(seed, cfg, ytr):
    """Class order, held-out reserve (bridge, control) and stream. Depends only on the seed and sizes."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(100)[: cfg['n_classes']]
    tasks = order.reshape(cfg['n_tasks'], -1)
    nb, nc = cfg['bridge_per_class'], cfg['control_per_class']
    bridge, control, stream_by_class = [], [], {}
    for c in order:
        ids = rng.permutation(np.where(ytr == c)[0])
        bridge.append(ids[:nb]); control.append(ids[nb:nb + nc])
        s = ids[nb + nc:]
        if cfg.get('stream_per_class'): s = s[: cfg['stream_per_class']]
        stream_by_class[int(c)] = s
    return rng, order, tasks, np.concatenate(bridge).astype(np.int64), np.concatenate(control).astype(np.int64), stream_by_class

def run_one(seed, cfg, data, log=print, full=True):
    """One online class-incremental run.
    full=True: returns the dict D for run_analysis.  full=False (calibration): returns summary numbers only."""
    Xtr, ytr, Xte, yte = data
    t0 = time.time()
    rng, order, tasks, bridge, control, stream_by_class = split_data(seed, cfg, ytr)
    stream_ids = np.concatenate([stream_by_class[int(c)] for c in order])
    assert len(np.intersect1d(stream_ids, np.concatenate([bridge, control]))) == 0
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    model = Net().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    ytr_t = torch.from_numpy(ytr).to(DEV)
    M, B, iters, aug = cfg['M'], cfg['batch'], cfg['iters'], cfg['aug']
    buf = np.zeros(M, dtype=np.int64); nbuf = 0; nseen = 0
    buffers, snaps = [], []
    for t, cs in enumerate(tasks):
        ids = rng.permutation(np.concatenate([stream_by_class[int(c)] for c in cs]))
        for i in range(0, len(ids), B):
            bi = ids[i:i + B]
            for _ in range(iters):                      # several steps on the same incoming batch, fresh replay each step
                allidx = np.concatenate([bi, buf[rng.integers(0, nbuf, size=B)]]) if nbuf > 0 else bi
                it = torch.from_numpy(allidx).to(DEV)
                x = to_input(Xtr[it])
                if aug: x = augment(x, g)
                loss = Fn.cross_entropy(model(x), ytr_t[it])
                if not torch.isfinite(loss):
                    raise Diverged(f'non-finite loss at task {t}')
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            for j in bi:                                # reservoir sampling (Algorithm R), after the steps
                nseen += 1
                if nbuf < M: buf[nbuf] = j; nbuf += 1
                else:
                    k = rng.integers(0, nseen)
                    if k < M: buf[k] = j
        buffers.append(buf[:nbuf].copy())
        if full or t == len(tasks) - 1:
            snaps.append(copy.deepcopy(model.enc).eval())
    t_train = time.time() - t0
    model.eval()
    with torch.no_grad():
        def head_acc(X, yy):
            keep = np.isin(yy, order)
            lg = torch.cat([model(to_input(X[i:i + 2500])) for i in range(0, len(X), 2500)]).cpu().numpy()[keep]
            return float((order[lg[:, order].argmax(1)] == yy[keep]).mean() * 100)
        acc_test = head_acc(Xte, yte)
        ct_t = torch.from_numpy(control).to(DEV)
        acc_ctl = head_acc(Xtr[ct_t], ytr[control])
    if not full:   # calibration: learned-feature ceiling measured on CONTROL items only (never the test set)
        from atlas_lib import pooled_cov, lda_predict
        f = lambda X: extract(lambda x: embed(snaps[-1], x), X)
        Fs, Fc = f(Xtr[torch.from_numpy(stream_ids).to(DEV)]).astype(np.float64), f(Xtr[ct_t]).astype(np.float64)
        ys, yc = ytr[stream_ids], ytr[control]
        Mm = np.stack([Fs[ys == c].mean(0) for c in order])
        o_ctl = float((order[lda_predict(Fc, Mm, pooled_cov(Fs, ys))] == yc).mean() * 100)
        log(f'    head(control) {acc_ctl:.1f} | oracle-LDA(control) {o_ctl:.1f} | train {t_train:.0f}s')
        return dict(head_ctl=acc_ctl, oracle_ctl=o_ctl, seconds=t_train)
    F = [extract(lambda x, e=e: embed(e, x), Xtr) for e in snaps]
    chart = RandomChart(seed + 777)
    R, Rtest = extract(chart, Xtr, bs=500), extract(chart, Xte, bs=500)
    Ftest = extract(lambda x: embed(snaps[-1], x), Xte)
    for a in F + [R, Rtest, Ftest]:
        assert np.isfinite(a).all(), 'non-finite features'
    log(f'  seed {seed}: train {t_train:.0f}s, total {time.time()-t0:.0f}s, ER head acc {acc_test:.1f}, buffer {nbuf}, '
        f'per-class buffer min {np.bincount(ytr[buf[:nbuf]], minlength=100)[order].min()}')
    return dict(tasks=[np.array(c) for c in tasks], y_train=ytr, stream_ids=stream_ids, bridge_ids=bridge,
                control_ids=control, F=F, R=R, buffers=buffers, Ftest=Ftest, Rtest=Rtest, y_test=yte, head_acc=acc_test,
                sigma_from=cfg['sigma_from'])

def rc_control_acc(seed, cfg, data):
    """Random-chart LDA on CONTROL items (the bar the learner should clear). Same split as run_one."""
    from atlas_lib import pooled_cov, lda_predict
    Xtr, ytr, _, _ = data
    _, order, _, _, control, sbc = split_data(seed, cfg, ytr)
    stream_ids = np.concatenate([sbc[int(c)] for c in order])
    chart = RandomChart(seed + 777)
    Rs = extract(chart, Xtr[torch.from_numpy(stream_ids).to(DEV)], bs=500).astype(np.float64)
    Rc = extract(chart, Xtr[torch.from_numpy(control).to(DEV)], bs=500).astype(np.float64)
    ys, yc = ytr[stream_ids], ytr[control]
    Mm = np.stack([Rs[ys == c].mean(0) for c in order])
    return float((order[lda_predict(Rc, Mm, pooled_cov(Rs, ys))] == yc).mean() * 100)
