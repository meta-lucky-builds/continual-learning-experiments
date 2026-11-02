"""Smallest head-to-head test of two JEPA world-model designs.

World: 2D point mass with damping, a one-way "river" (strong +y drift in |x|<0.25,
stronger than the max action), observations = nonlinear 16-d view of POSITION only
(velocity hidden) + noise, 8 pure-noise distractor channels, random occlusion runs.

Model A (Lucky's table): Euclidean GRU recurrent state (posterior + prior GRU),
  Gaussian-regularized (SIGReg) prediction projection, diagonal Gaussian belief,
  quasimetric planning head, TQ submodule (q, qdot heads + kinematic consistency).
Model B (earlier combination): phase-space state (q, p) with structured
  port-Hamiltonian-style dynamics (control-affine, explicit PSD damping, symplectic
  Euler), a learned-gain filter (precision-weighted fusion of prediction and
  measurement), low-rank+diagonal Gaussian belief, same SIGReg projection and
  same quasimetric planning head.
Both get the same 5% position grounding labels and the same data.

Tests: (1) open-loop rollout accuracy (probe), (2) velocity decodability,
(3) belief NLL at 1 and 10 steps ahead and calibration, (4) MPC planning success
across the river with quasimetric cost vs plain latent L1 cost.

NOTE: this file was rebuilt on 2026-11-02 from the session record of the original
2026-10-23 run, because the original copy was missing from the project files.
It is the final version that produced result_seed0.json. Seeds 1 and 2 ran with
one earlier variant of the ridge probe in eval_model (float32, ridge 1e-3 instead
of float64, ridge 1e-2); see EXPLANATION.md.
Usage: python3 wmtest.py <seed> [train_steps=3000]
"""
import math, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
DT, DAMP, RIVER_W, RIVER_F = 0.1, 0.5, 0.25, 2.0
OBS_INF, OBS_DIS = 16, 8
OBS_D = OBS_INF + OBS_DIS + 1  # + occlusion flag


class World:
    def __init__(self, seed):
        r = np.random.RandomState(1000 + seed)
        self.W = r.randn(OBS_INF, 2) * 1.5
        self.b = r.randn(OBS_INF) * 0.5

    def step(self, x, v, u):
        u = np.clip(u, -1, 1)
        drift = np.zeros_like(x)
        drift[:, 1] = RIVER_F * (np.abs(x[:, 0]) < RIVER_W)
        v = v + DT * (u + drift - DAMP * v)
        x = x + DT * v
        hit = np.abs(x) > 1
        x = np.clip(x, -1, 1)
        v = np.where(hit, 0.0, v)
        return x, v

    def observe(self, x, occ, rng):
        inf = np.tanh(x @ self.W.T + self.b) + 0.05 * rng.randn(len(x), OBS_INF)
        inf = inf * (1 - occ[:, None])
        dis = rng.randn(len(x), OBS_DIS)
        return np.concatenate([inf, dis, occ[:, None]], 1).astype(np.float32)


def gen_data(world, n_ep, T, seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-0.9, 0.9, (n_ep, 2)); v = np.zeros((n_ep, 2))
    u = np.zeros((n_ep, 2)); occ_left = np.zeros(n_ep)
    O, A, P, V = [], [], [], []
    for t in range(T):
        start = (rng.rand(n_ep) < 0.05) & (occ_left == 0) & (t > 2)
        occ_left = np.where(start, 6, occ_left)
        occ = (occ_left > 0).astype(np.float64)
        occ_left = np.maximum(occ_left - 1, 0)
        O.append(world.observe(x, occ, rng)); P.append(x.copy()); V.append(v.copy())
        u = np.clip(0.8 * u + 0.6 * rng.randn(n_ep, 2), -1, 1)
        A.append(u.astype(np.float32))
        x, v = world.step(x, v, u)
    f = lambda L: torch.tensor(np.stack(L, 1), dtype=torch.float32)
    return f(O), f(A), f(P), f(V)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


def sigreg(z, n_dir=32):
    """Epps-Pulley-style sliced test toward N(0, I) (LeJEPA's SIGReg, simplified)."""
    z = z.reshape(-1, z.shape[-1])
    d = torch.randn(z.shape[-1], n_dir, device=z.device); d = d / d.norm(dim=0, keepdim=True)
    p = z @ d
    t = torch.linspace(-3, 3, 17)
    emp_c = torch.cos(p[..., None] * t).mean(0); emp_s = torch.sin(p[..., None] * t).mean(0)
    tgt = torch.exp(-t ** 2 / 2)
    w = torch.exp(-t ** 2 / 2)
    return (((emp_c - tgt) ** 2 + emp_s ** 2) * w).mean()


class QuasiMetric(nn.Module):
    """MRN-style quasimetric: max-relu asymmetric part + Euclidean symmetric part."""
    def __init__(self, d):
        super().__init__(); self.f = mlp(d, 128, 32); self.g = mlp(d, 128, 16)
    def forward(self, x, y):
        a, b = self.f(x), self.f(y)
        return F.relu(b - a).max(-1).values * 4 + (self.g(x) - self.g(y)).norm(dim=-1)


class ModelA(nn.Module):
    name = "A_table"
    def __init__(self, H=64, E=16):
        super().__init__()
        self.H = H
        self.enc = mlp(OBS_D, 64, E)
        self.post = nn.GRUCell(E + 2, H); self.prior = nn.GRUCell(2, H)
        self.proj = mlp(H, 64, E)
        self.bel_mu = mlp(H, 64, OBS_D - 1); self.bel_lv = mlp(H, 64, OBS_D - 1)
        self.q = nn.Linear(H, 2); self.qd = nn.Linear(H, 2)
    def init(self, B): return torch.zeros(B, self.H)
    def prior_step(self, s, u): return self.prior(u, s)
    def post_step(self, s_prior, s_prev, u, o):
        return self.post(torch.cat([self.enc(o), u], -1), s_prev)
    def belief(self, s):
        return torch.distributions.Independent(
            torch.distributions.Normal(self.bel_mu(s), (0.5 * self.bel_lv(s).clamp(-8, 4)).exp()), 1)
    def pos(self, s): return self.q(s)
    def extra_loss(self, S_post, S_prior, vis):
        q, qd = self.q(S_post), self.qd(S_post)
        kin = ((q[:, 1:] - q[:, :-1].detach() - DT * qd[:, :-1]) ** 2).sum(-1).mean()
        cons = ((S_prior[:, 1:] - S_post[:, 1:].detach()) ** 2).sum(-1).mean()
        return kin + cons


class ModelB(nn.Module):
    name = "B_combination"
    def __init__(self, Dq=4, E=16, R=4):
        super().__init__()
        self.Dq = Dq
        self.enc = mlp(OBS_D, 64, E)
        self.meas = mlp(E, 64, Dq)
        self.force = mlp(Dq, 64, Dq); self.damp = mlp(Dq, 32, Dq); self.G = mlp(Dq, 64, Dq * 2)
        self.gain = mlp(2 * Dq + E, 64, 2 * Dq)
        self.proj = mlp(2 * Dq, 64, E)
        self.bel_mu = mlp(2 * Dq, 64, OBS_D - 1); self.bel_lr = mlp(2 * Dq, 64, (OBS_D - 1) * R)
        self.bel_ld = mlp(2 * Dq, 64, OBS_D - 1); self.R = R
    def init(self, B): return torch.zeros(B, 2 * self.Dq)
    def prior_step(self, s, u):
        q, p = s[:, :self.Dq], s[:, self.Dq:]
        G = self.G(q).view(-1, self.Dq, 2)
        p = p + DT * (self.force(q) - F.softplus(self.damp(q)) * p + (G @ u[..., None])[..., 0])
        q = q + DT * p  # symplectic Euler
        return torch.cat([q, p], -1)
    def post_step(self, s_prior, s_prev, u, o):
        e = self.enc(o); vis = 1 - o[:, -1:]
        q, p = s_prior[:, :self.Dq], s_prior[:, self.Dq:]
        innov = self.meas(e) - q
        k = torch.sigmoid(self.gain(torch.cat([s_prior, e], -1))) * vis
        return torch.cat([q + k[:, :self.Dq] * innov, p + 5 * k[:, self.Dq:] * innov], -1)
    def belief(self, s):
        B = s.shape[0]
        return torch.distributions.LowRankMultivariateNormal(
            self.bel_mu(s), 0.3 * self.bel_lr(s).view(B, OBS_D - 1, self.R),
            F.softplus(self.bel_ld(s)) + 1e-3)
    def pos(self, s): return s[:, :2]
    def extra_loss(self, S_post, S_prior, vis):
        e_meas = self.meas(self.enc_cache)
        q_prior = S_prior[..., :self.Dq]
        return (((e_meas - q_prior) ** 2).sum(-1) * vis).mean()


def run_seq(m, O, A, s0=None):
    B, T, _ = O.shape
    s = m.init(B) if s0 is None else s0
    u_prev = torch.zeros(B, 2)
    Sp, Spr = [], []
    for t in range(T):
        sp = m.prior_step(s, u_prev)
        s = m.post_step(sp, s, u_prev, O[:, t])
        Spr.append(sp); Sp.append(s); u_prev = A[:, t]
    return torch.stack(Sp, 1), torch.stack(Spr, 1)


def train(m, data, steps, seed, log):
    torch.manual_seed(seed)
    O, A, P, V = data
    N, T = O.shape[:2]
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    glab = torch.rand(N, T) < 0.05  # 5% grounding labels, same for both models
    for it in range(steps):
        idx = torch.randint(0, N, (32,)); t0 = np.random.randint(0, T - 32)
        o, a, p = O[idx, t0:t0 + 32], A[idx, t0:t0 + 32], P[idx, t0:t0 + 32]
        g = glab[idx, t0:t0 + 32] & (o[..., -1] == 0)
        vis = 1 - o[..., -1]
        S, Sp = run_seq(m, o, a)
        e = m.enc(o)
        if isinstance(m, ModelB): m.enc_cache = e
        # JEPA prediction from the prior (imagined) state to the stop-grad embedding
        jepa = (((m.proj(Sp) - e.detach()) ** 2).sum(-1) * vis).sum() / vis.sum()
        reg = sigreg(e[vis > 0])
        bel = m.belief(Sp[:, 1:].reshape(-1, Sp.shape[-1]))
        nll = -(bel.log_prob(o[:, 1:, :-1].reshape(-1, OBS_D - 1)) * vis[:, 1:].reshape(-1)).sum() / vis[:, 1:].sum()
        gq = m.pos(S.reshape(-1, S.shape[-1]))[g.reshape(-1)]
        ground = ((gq - p[g]) ** 2).sum(-1).mean() if g.any() else torch.tensor(0.)
        # multi-step open-loop term: imagine 8 steps from t=8 and predict embeddings
        s = S[:, 8]; ol = 0
        for k in range(1, 9):
            s = m.prior_step(s, a[:, 8 + k - 1])
            vk = vis[:, 8 + k]
            ol = ol + (((m.proj(s) - e[:, 8 + k].detach()) ** 2).sum(-1) * vk).mean()
        loss = jepa + 0.5 * ol / 8 + 5 * reg + 0.05 * nll + 10 * ground + m.extra_loss(S, Sp, vis)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
        if it % 500 == 0:
            log(f"  {m.name} it{it} loss {loss.item():.3f} jepa {jepa.item():.3f} nll {nll.item():.2f} ground {ground.item():.4f}")


def train_qm(m, data, steps, log):
    O, A, P, V = data
    with torch.no_grad():
        S, _ = run_seq(m, O[:, :64], A[:, :64])
    S = S.detach(); N, T, D = S.shape
    qm = QuasiMetric(D); opt = torch.optim.Adam(qm.parameters(), 1e-3)
    lam = 1.0
    for it in range(steps):
        i = torch.randint(0, N, (256,)); t = torch.randint(0, T - 1, (256,))
        x, y = S[i, t], S[i, t + 1]
        j = torch.randint(0, N, (256,)); t2 = torch.randint(0, T, (256,))
        r = S[j, t2]
        d_rand = qm(x, r)
        d_step = qm(x, y)
        # QRL: maximise distances (saturating) subject to one-step distances <= 1
        viol = (F.relu(d_step - 1) ** 2).mean()
        loss = -(100 * torch.tanh(d_rand / 100)).mean() + lam * viol
        opt.zero_grad(); loss.backward(); opt.step()
        lam = max(0.0, lam + 10 * (viol.item() - 0.005))
    log(f"  qm {m.name}: step-dist {d_step.mean().item():.2f}, rand-dist {d_rand.mean().item():.1f}, lam {lam:.1f}")
    return qm


@torch.no_grad()
def eval_model(m, world, test, log):
    O, A, P, V = test
    S, _ = run_seq(m, O, A)
    B, T, D = S.shape
    # linear probes (ridge) on filtered states, fit on first half, test on second
    X = torch.cat([S.reshape(-1, D), torch.ones(B * T, 1)], 1)
    half = (B // 2) * T
    def probe(Y):
        Xa, Ya = X[:half].double(), Y.reshape(-1, Y.shape[-1])[:half].double()
        W = torch.linalg.solve(Xa.T @ Xa + 1e-2 * torch.eye(D + 1, dtype=torch.float64), Xa.T @ Ya)
        return W.float()
    Wp, Wv = probe(P), probe(V)
    Yv = V.reshape(-1, 2)[half:]; pv = X[half:] @ Wv
    vel_r2 = 1 - ((pv - Yv) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum()
    Yp = P.reshape(-1, 2)[half:]; pp = X[half:] @ Wp
    pos_rmse_filt = ((pp - Yp) ** 2).sum(-1).mean().sqrt()
    # open-loop rollouts: from t0=16, imagine k steps with true actions
    hb = B // 2; t0 = 16
    s = S[hb:, t0]; errs = {}; nll = {}; z2 = {}
    for k in range(1, 21):
        s = m.prior_step(s, A[hb:, t0 + k - 1])
        if k in (1, 5, 10, 20):
            ps = torch.cat([s, torch.ones(len(s), 1)], 1) @ Wp
            errs[k] = ((ps - P[hb:, t0 + k]) ** 2).sum(-1).mean().sqrt().item()
            o = O[hb:, t0 + k]; vis = o[:, -1] == 0
            bel = m.belief(s)
            nll[k] = (-bel.log_prob(o[:, :-1])[vis].mean() / (OBS_D - 1)).item()
            # calibration on informative dims: mean squared standardized error (1.0 = calibrated)
            if isinstance(bel.base_dist if hasattr(bel, 'base_dist') else bel, torch.distributions.Normal):
                var = bel.base_dist.scale ** 2
            else:
                var = bel.variance
            zz = ((o[:, :OBS_INF] - bel.mean[:, :OBS_INF]) ** 2 / var[:, :OBS_INF])[vis]
            z2[k] = zz.mean().item()
    return dict(vel_r2=vel_r2.item(), pos_rmse_filtered=pos_rmse_filt.item(),
                rollout_pos_rmse=errs, belief_nll_per_dim=nll, calib_z2=z2), Wp


@torch.no_grad()
def plan_eval(m, qm, world, Wp, cost, n_tasks, seed, log):
    rng = np.random.RandomState(500 + seed)
    succ, final_d = [], []
    for task in range(n_tasks):
        side = 1 if task % 2 == 0 else -1
        x0 = np.array([[-0.7 * side, rng.uniform(-0.8, 0.2)]]); v = np.zeros((1, 2))
        goal = np.array([[0.7 * side, rng.uniform(-0.8, 0.2)]])
        go = torch.tensor(world.observe(goal, np.zeros(1), rng))
        sg = m.init(1)
        for _ in range(3):
            sg = m.post_step(m.prior_step(sg, torch.zeros(1, 2)), sg, torch.zeros(1, 2), go)
        s = m.init(1); u_prev = torch.zeros(1, 2); x = x0
        reached = False; mean = torch.zeros(12, 2)
        for step in range(60):
            o = torch.tensor(world.observe(x, np.zeros(1), rng))
            s = m.post_step(m.prior_step(s, u_prev), s, u_prev, o)
            if step % 4 == 0:
                mu, sd = torch.cat([mean[4:], torch.zeros(4, 2)]), torch.ones(12, 2) * 0.8
                for _ in range(4):
                    U = (mu + sd * torch.randn(256, 12, 2)).clamp(-1, 1)
                    z = s.expand(256, -1); c = 0
                    for h in range(12):
                        z = m.prior_step(z, U[:, h])
                    if cost == "quasimetric":
                        c = qm(z, sg.expand(256, -1))
                    else:
                        c = (z - sg).abs().sum(-1)
                    el = U[c.argsort()[:25]]
                    mu, sd = el.mean(0), el.std(0) + 0.05
                mean = mu; plan = mu
            u = plan[step % 4].numpy()[None]
            x, v = world.step(x, v, u); u_prev = torch.tensor(u, dtype=torch.float32)
            if np.linalg.norm(x - goal) < 0.15: reached = True; break
        succ.append(reached); final_d.append(float(np.linalg.norm(x - goal)))
    return float(np.mean(succ)), float(np.mean(final_d))


def greedy_baseline(world, n_tasks, seed):
    """Reference: push straight at the goal using TRUE position (no model)."""
    rng = np.random.RandomState(500 + seed); succ = []
    for task in range(n_tasks):
        side = 1 if task % 2 == 0 else -1
        x = np.array([[-0.7 * side, rng.uniform(-0.8, 0.2)]]); v = np.zeros((1, 2))
        goal = np.array([[0.7 * side, rng.uniform(-0.8, 0.2)]])
        world.observe(goal, np.zeros(1), rng); ok = False
        for step in range(60):
            world.observe(x, np.zeros(1), rng)
            d = goal - x; u = np.clip(3 * d - 0.5 * v, -1, 1)
            x, v = world.step(x, v, u)
            if np.linalg.norm(x - goal) < 0.15: ok = True; break
        succ.append(ok)
    return float(np.mean(succ))


def main(seed, steps=3000):
    out = {}
    log = lambda s: (print(s, flush=True))
    world = World(seed)
    data = gen_data(world, 3000, 64, seed)
    test = gen_data(world, 400, 48, seed + 77)
    out["greedy_true_state_success"] = greedy_baseline(world, 30, seed)
    for M in (ModelA, ModelB):
        torch.manual_seed(seed); np.random.seed(seed)
        m = M(); t = time.time()
        nparam = sum(p.numel() for p in m.parameters())
        train(m, data, steps, seed, log)
        qm = train_qm(m, data, 3000, log)
        res, Wp = eval_model(m, world, test, log)
        res["params"] = nparam
        for cost in ("quasimetric", "latent_L1"):
            sr, fd = plan_eval(m, qm, world, Wp, cost, 30, seed, log)
            res[f"plan_{cost}_success"] = sr; res[f"plan_{cost}_final_dist"] = fd
        res["minutes"] = (time.time() - t) / 60
        log(f"{M.name} seed{seed}: {json.dumps(res)}")
        out[M.name] = res
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    r = main(seed, steps)
    json.dump(r, open(f"result_seed{seed}.json", "w"), indent=1)
