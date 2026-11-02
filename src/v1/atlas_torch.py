# ===== Learner, stable random chart, online stream, feature extraction (PyTorch) =====
import os, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, torchvision
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = torch.tensor([0.5071, 0.4865, 0.4409], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762], device=DEV).view(1, 3, 1, 1)

def load_cifar100(root='/content/data'):
    tr = torchvision.datasets.CIFAR100(root, train=True, download=True)
    te = torchvision.datasets.CIFAR100(root, train=False, download=True)
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(DEV)   # uint8 N,3,32,32 on GPU
    Xtr, ytr, Xte, yte = to_t(tr.data), np.array(tr.targets), to_t(te.data), np.array(te.targets)
    assert Xtr.shape == (50000, 3, 32, 32) and Xte.shape == (10000, 3, 32, 32)
    assert (np.bincount(ytr) == 500).all() and (np.bincount(yte) == 100).all()
    return Xtr, ytr, Xte, yte

def to_input(xu8):
    return (xu8.float() / 255.0 - MEAN) / STD

def make_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU())

def embed(enc, x):
    return enc(x).mean((2, 3))           # global average pool (plain mean: deterministic backward)

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

def run_one(seed, cfg, data, log=print):
    """One online class-incremental run. Returns the dict D that run_analysis expects."""
    Xtr, ytr, Xte, yte = data
    t0 = time.time()
    rng = np.random.default_rng(seed)
    order = rng.permutation(100)[: cfg['n_classes']]
    tasks = order.reshape(cfg['n_tasks'], -1)
    nb = cfg['reserve_per_class'] // 2
    bridge, control, stream_by_class = [], [], {}
    for c in order:
        ids = rng.permutation(np.where(ytr == c)[0])
        bridge.append(ids[:nb]); control.append(ids[nb:2 * nb])
        s = ids[2 * nb:]
        if cfg.get('stream_per_class'): s = s[: cfg['stream_per_class']]
        stream_by_class[int(c)] = s
    bridge, control = np.concatenate(bridge), np.concatenate(control)
    stream_ids = np.concatenate([stream_by_class[int(c)] for c in order])
    assert len(np.intersect1d(stream_ids, np.concatenate([bridge, control]))) == 0
    torch.manual_seed(seed)
    model = Net().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'])
    ytr_t = torch.from_numpy(ytr).to(DEV)
    M, B = cfg['M'], cfg['batch']
    buf = np.zeros(M, dtype=np.int64); nbuf = 0; nseen = 0
    buffers, snaps = [], []
    for t, cs in enumerate(tasks):
        ids = rng.permutation(np.concatenate([stream_by_class[int(c)] for c in cs]))
        for i in range(0, len(ids), B):
            bi = ids[i:i + B]
            allidx = np.concatenate([bi, buf[rng.integers(0, nbuf, size=B)]]) if nbuf > 0 else bi  # replay drawn before insertion
            it = torch.from_numpy(allidx).to(DEV)
            loss = Fn.cross_entropy(model(to_input(Xtr[it])), ytr_t[it])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            for j in bi:                                # reservoir sampling (Algorithm R)
                nseen += 1
                if nbuf < M: buf[nbuf] = j; nbuf += 1
                else:
                    k = rng.integers(0, nseen)
                    if k < M: buf[k] = j
            if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss at task {t}')
        buffers.append(buf[:nbuf].copy())
        snaps.append(copy.deepcopy(model.enc).eval())
    t_train = time.time() - t0
    model.eval()
    with torch.no_grad():
        keep = np.isin(yte, order)
        logits = torch.cat([model(to_input(Xte[i:i + 2500])) for i in range(0, len(Xte), 2500)]).cpu().numpy()[keep]
        head_acc = float((order[logits[:, order].argmax(1)] == yte[keep]).mean() * 100)
    F = [extract(lambda x, e=e: embed(e, x), Xtr) for e in snaps]
    chart = RandomChart(seed + 777)
    R, Rtest = extract(chart, Xtr, bs=500), extract(chart, Xte, bs=500)   # 500 keeps the 256x32x32 maps small
    Ftest = extract(lambda x: embed(snaps[-1], x), Xte)
    for a in F + [R, Rtest, Ftest]:
        assert np.isfinite(a).all(), 'non-finite features'
    log(f'  seed {seed}: train {t_train:.0f}s, total {time.time()-t0:.0f}s, ER head acc {head_acc:.1f}, '
        f'buffer {nbuf}, per-class buffer min {np.bincount(ytr[buf[:nbuf]], minlength=100)[order].min()}')
    return dict(tasks=[np.array(c) for c in tasks], y_train=ytr, stream_ids=stream_ids, bridge_ids=bridge,
                control_ids=control, F=F, R=R, buffers=buffers, Ftest=Ftest, Rtest=Rtest, y_test=yte, head_acc=head_acc)
