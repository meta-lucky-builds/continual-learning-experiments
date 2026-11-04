"""Idea B test: is the haystack "interference wall" a delayed-write-credit problem?

Model, task, loss, optimiser and evaluation are copied VERBATIM from the public harness of
arXiv:2609.16183 (github.com/JIBSIL/dualgoose, MIT License, (c) 2026 Julian Boesch and Andrew Wee):
  dualgoose/mixers.py   DeltaNetRefLayer, StaticMerge, BidirectionalRWKVMixer (delta-rule path)
  dualgoose/model.py    timestep_features, TinyDualGooseBlock, TinyDualGooseDenoiser
  dualgoose/toy_data.py make_long_retrieval_batch
  dualgoose/diffusion.py suppress_mask_logit, masked_nll_loss, apply_subs_sample
  scripts/long_retrieval_probe.py run_trial (training loop + eval + rebinding control)
`parity_test.py` checks numerically that this copy matches the original package.

The ONLY additions (all off unless aux_mode != "none"):
  * each DeltaNetRefLayer can read its recurrent state S_t with fixed-per-run learned probe vectors
    p_i (one per table key; keys are always tokens 5..5+pairs-1) at every step: r_{t,i} = S_t p_i.
  * an auxiliary linear head maps r_{t,i} -> value-token logits; loss = CE vs the true value of key i.
  * aux_mode="local": the aux loss is applied at EVERY step t after pair i is written, so a write that
    damages binding i is penalised one step later (immediate credit to the write path).
  * aux_mode="end": the same probes/heads/targets, applied ONLY at the last position (credit has to
    travel the whole gap, as for the task loss). Controls for "extra probe machinery / extra targets".
The aux head is never used at evaluation; eval accuracy is the model's own masked-token prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------------- verbatim copies


def timestep_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, math.log(1000.0), half, device=t.device, dtype=t.dtype)
    )
    angles = t[:, None] * freqs[None, :]
    features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if features.shape[-1] < dim:
        features = torch.cat([features, t[:, None]], dim=-1)
    return features


class DeltaNetRefLayer(nn.Module):
    def __init__(self, d_model, d_key=None, *, q_act="tanh", gated=False, short_conv=0):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key or d_model
        self.scan_scale = self.d_key**-0.5
        self.q_act = q_act
        self.gated = gated
        extra = self.d_key if gated else 0
        self.proj = nn.Linear(d_model, 3 * self.d_key + 1 + extra)  # q, k, v, beta (, decay)
        self.out = nn.Linear(self.d_key, d_model)
        self.short_conv = short_conv
        if short_conv:
            self.qkv_conv = nn.Conv1d(
                3 * self.d_key, 3 * self.d_key, kernel_size=short_conv, groups=3 * self.d_key
            )
        # --- addition: optional state probes (set externally per forward; None = original code path)
        self.probes = None       # (P, d_key) or None
        self.probe_reads = None  # filled with (B, L, P, d_key)

    def forward(self, x):
        sizes = [self.d_key, self.d_key, self.d_key, 1] + ([self.d_key] if self.gated else [])
        parts = torch.split(self.proj(x), sizes, dim=-1)
        q, k, v, beta = parts[0], parts[1], parts[2], parts[3]
        if self.short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.qkv_conv(F.pad(qkv, (self.short_conv - 1, 0)))
            qkv = F.silu(qkv.transpose(1, 2))
            q, k, v = torch.split(qkv, self.d_key, dim=-1)
        w = torch.sigmoid(parts[4]) if self.gated else None
        q = (torch.tanh(q) if self.q_act == "tanh" else q) * self.scan_scale
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(beta)
        out_dtype = v.dtype
        q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
        w = w.float() if self.gated else None
        batch, length, d_k = k.shape
        state = torch.zeros(batch, d_k, d_k, device=x.device, dtype=torch.float32)
        outs = []
        reads = [] if self.probes is not None else None
        for ti in range(length):
            k_t, v_t, q_t = k[:, ti], v[:, ti], q[:, ti]
            b_t = beta[:, ti]
            if self.gated:
                state = state * w[:, ti].unsqueeze(1)
            s_k = torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)
            delta = b_t * (v_t - s_k)
            state = state + delta.unsqueeze(-1) * k_t.unsqueeze(1)
            outs.append(torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1))
            if reads is not None:  # addition: r_{t,i} = S_t p_i, read AFTER the write at step t
                reads.append(torch.matmul(state, self.probes.t()).transpose(1, 2))  # (B, P, d_k)
        scanned = torch.stack(outs, dim=1)
        self.probe_reads = torch.stack(reads, dim=1) if reads is not None else None
        return self.out(scanned.to(out_dtype))


class StaticMerge(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            eye = torch.eye(d_model)
            self.proj.weight[:, :d_model].copy_(0.5 * eye)
            self.proj.weight[:, d_model:].copy_(0.5 * eye)

    def forward(self, u_fwd, u_bwd, t=None):
        return self.proj(torch.cat([u_fwd, u_bwd], dim=-1))


class BidirectionalRWKVMixer(nn.Module):
    """Delta-rule path of the original mixer (the only path used here: model_type=gated_deltanet)."""

    def __init__(self, d_model, *, direction="bidirectional", short_conv=0):
        super().__init__()
        self.direction = direction
        self.fwd = DeltaNetRefLayer(d_model, short_conv=short_conv, gated=True)
        self.bwd = DeltaNetRefLayer(d_model, short_conv=short_conv, gated=True)
        self.merge = StaticMerge(d_model)

    def forward(self, x, t=None):
        if self.direction == "forward":
            return self.fwd(x)
        if self.direction == "backward":
            return self.bwd(x.flip(1)).flip(1)
        u_fwd = self.fwd(x)
        u_bwd = self.bwd(x.flip(1)).flip(1)
        return self.merge(u_fwd, u_bwd, t)


class TinyDualGooseBlock(nn.Module):
    def __init__(self, d_model, *, direction="bidirectional", short_conv=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = BidirectionalRWKVMixer(d_model, direction=direction, short_conv=short_conv)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, t):
        x = x + self.mixer(self.norm1(x), t)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyDualGooseDenoiser(nn.Module):
    def __init__(self, vocab_size, max_length, *, d_model=64, n_layers=2,
                 direction="bidirectional", short_conv=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [TinyDualGooseBlock(d_model, direction=direction, short_conv=short_conv)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens, t):
        if t.ndim == 0:
            t = t.expand(tokens.shape[0])
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        x = x + self.time_proj(timestep_features(t.to(x.dtype), x.shape[-1]))[:, None, :]
        for block in self.blocks:
            x = block(x, t)
        return self.lm_head(self.norm(x))


@dataclass(frozen=True)
class ToyBatch:
    clean: torch.Tensor
    corrupted: torch.Tensor
    target_mask: torch.Tensor
    t: torch.Tensor


def _randint(low, high, shape, *, device, generator):
    return torch.randint(low, high, shape, device=device, generator=generator)


def make_long_retrieval_batch(batch_size, seq_len, vocab_size, mask_token_id, *, pairs=4,
                              n_queries=1, gap=None, collision=False, device=None, generator=None):
    tail = 2 * n_queries
    max_gap = seq_len - 2 * pairs - tail
    gap = max_gap if gap is None else gap
    assert 0 <= gap <= max_gap
    key_lo = 5
    val_lo = 5 + pairs
    clean = _randint(val_lo, vocab_size - 1, (batch_size, seq_len), device=device, generator=generator)
    table_start = seq_len - tail - gap - 2 * pairs
    values = torch.empty(batch_size, pairs, dtype=torch.long, device=device)
    for pair_idx in range(pairs):
        value = _randint(val_lo, vocab_size - 1, (batch_size,), device=device, generator=generator)
        pos = table_start + 2 * pair_idx
        clean[:, pos] = key_lo + pair_idx
        clean[:, pos + 1] = value
        values[:, pair_idx] = value
    assert not collision  # collision variant not used in this study
    noise = torch.rand(batch_size, pairs, device=device, generator=generator)
    sel = torch.argsort(noise, dim=1)[:, :n_queries]
    corrupted_positions = []
    for q in range(n_queries):
        qpos = seq_len - tail + 2 * q
        clean[:, qpos] = key_lo + sel[:, q]
        clean[:, qpos + 1] = values.gather(1, sel[:, q : q + 1]).squeeze(1)
        corrupted_positions.append(qpos + 1)
    corrupted = clean.clone()
    target_mask = torch.zeros_like(clean, dtype=torch.bool)
    for apos in corrupted_positions:
        corrupted[:, apos] = mask_token_id
        target_mask[:, apos] = True
    t = torch.full((batch_size,), 0.5, device=device)
    return ToyBatch(clean, corrupted, target_mask, t), table_start, values


def suppress_mask_logit(logits, mask_token_id):
    logits = logits.clone()
    logits[..., mask_token_id] = torch.finfo(logits.dtype).min
    return logits


def masked_nll_loss(logits, targets, corrupted, mask_token_id, *, t=None, eps=1e-4):
    masked = corrupted.eq(mask_token_id)
    clean_logits = suppress_mask_logit(logits, mask_token_id)
    nll = F.cross_entropy(clean_logits.view(-1, clean_logits.shape[-1]), targets.reshape(-1),
                          reduction="none").view_as(targets)
    if t is not None:
        weights = 1.0 / t.to(torch.float32)[:, None].clamp_min(eps)
        nll = nll * weights
    return nll[masked].mean()


def apply_subs_sample(logits, corrupted, mask_token_id):
    predicted = suppress_mask_logit(logits, mask_token_id).argmax(dim=-1)
    return torch.where(corrupted.eq(mask_token_id), predicted, corrupted)


# ----------------------------------------------------------------------------- addition: aux probes


class StateProbes(nn.Module):
    """One probe vector per table key per probed layer, plus a linear value decoder per layer."""

    def __init__(self, layers, pairs, d_key, vocab):
        super().__init__()
        self.p = nn.Parameter(torch.randn(layers, pairs, d_key))
        self.heads = nn.ModuleList([nn.Linear(d_key, vocab) for _ in range(layers)])

    def probe_vectors(self, li):
        return F.normalize(self.p[li], dim=-1)  # unit probes, like the cell's L2-normalised keys


def recurrent_layers(model):
    return [blk.mixer.fwd for blk in model.blocks]  # forward direction (causal arm)


def aux_loss_and_acc(model, probes, table_start, values, mode):
    """CE of decoding value_i from S_t p_i. mode: 'local' (all t after pair i is written) or 'end'."""
    losses, accs = [], []
    for li, layer in enumerate(recurrent_layers(model)):
        reads = layer.probe_reads  # (B, L, P, d)
        logits = probes.heads[li](reads)  # (B, L, P, V)
        B, L, P, V = logits.shape
        tpos = torch.arange(L, device=logits.device)[:, None]
        written = table_start + 2 * torch.arange(P, device=logits.device)[None, :] + 1  # value pos
        valid = tpos >= written  # (L, P): read at t includes the write at t
        if mode == "end":
            valid = valid & (tpos == L - 1)
        tgt = values[:, None, :].expand(B, L, P)
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").view(B, L, P)
        m = valid[None].expand(B, L, P)
        losses.append(ce[m].mean())
        accs.append(float(logits[:, L - 1].argmax(-1).eq(values).float().mean()))
    return torch.stack(losses).mean(), accs


# ----------------------------------------------------------------------------- run_trial (verbatim + aux)


def run_trial(seed, *, aux_mode="none", aux_weight=1.0, curriculum=False, n_queries=4,
              seq_len=64, pairs=4, d_model=32, layers=2, vocab=64, steps=3000, lr=1e-3, batch=256,
              eval_batch=256, eval_iters=8, direction="forward", short_conv=4, log_every=250,
              ckpt_path=None, ckpt_every=250, stop_after=None):
    device = torch.device("cpu")
    mask_id = vocab - 1
    torch.manual_seed(seed)
    model = TinyDualGooseDenoiser(vocab, seq_len, d_model=d_model, n_layers=layers,
                                  direction=direction, short_conv=short_conv).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    probes = None
    if aux_mode != "none":
        assert direction == "forward"
        probes = StateProbes(layers, pairs, d_model, vocab).to(device)  # created AFTER base model
        aopt = torch.optim.AdamW(probes.parameters(), lr=lr)
        asched = torch.optim.lr_scheduler.CosineAnnealingLR(aopt, T_max=steps, eta_min=lr / 10)
    gen = torch.Generator(device=device).manual_seed(seed)
    gcpu = torch.Generator().manual_seed(seed + 555)
    max_gap = seq_len - 2 * pairs - 2 * n_queries
    t0 = time.time()
    last = float("nan")
    curve = []
    start = 0
    # addition: exact-resume checkpointing (container restarts); restores every RNG and optimiser state
    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if probes is not None:
            probes.load_state_dict(ck["probes"]); aopt.load_state_dict(ck["aopt"]); asched.load_state_dict(ck["asched"])
        gen.set_state(ck["gen"]); gcpu.set_state(ck["gcpu"]); torch.set_rng_state(ck["torch_rng"])
        start, last, curve = ck["step"], ck["last"], ck["curve"]
    for step in range(start, steps):
        if ckpt_path is not None and step > start and step % ckpt_every == 0:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "gen": gen.get_state(), "gcpu": gcpu.get_state(), "torch_rng": torch.get_rng_state(),
                  "step": step, "last": last, "curve": curve}
            if probes is not None:
                ck.update(probes=probes.state_dict(), aopt=aopt.state_dict(), asched=asched.state_dict())
            torch.save(ck, str(ckpt_path) + ".tmp")
            Path(str(ckpt_path) + ".tmp").replace(ckpt_path)
        if stop_after is not None and step >= stop_after:
            return None  # testing hook: simulate a crash
        g = None
        if curriculum:
            g = int(torch.randint(0, max_gap + 1, (1,), generator=gcpu))  # uniform shape (paper)
        b, table_start, values = make_long_retrieval_batch(
            batch, seq_len, vocab, mask_id, pairs=pairs, n_queries=n_queries, gap=g,
            device=device, generator=gen)
        if probes is not None:
            for li, layer in enumerate(recurrent_layers(model)):
                layer.probes = probes.probe_vectors(li)
        logits = model(b.corrupted, b.t)
        loss = masked_nll_loss(logits, b.clean, b.corrupted, mask_id, t=b.t)
        total = loss
        aux_val = None
        if probes is not None:
            aux, _ = aux_loss_and_acc(model, probes, table_start, values, aux_mode)
            total = loss + aux_weight * aux
            aux_val = float(aux.detach())
            aopt.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if probes is not None:
            torch.nn.utils.clip_grad_norm_(probes.parameters(), 1.0)
            aopt.step()
            asched.step()
        last = float(loss.detach())
        if step % log_every == 0 or step == steps - 1:
            curve.append({"step": step, "task_loss": round(last, 4),
                          "aux_loss": None if aux_val is None else round(aux_val, 4)})
    # ---- evaluation (verbatim protocol: max gap, own masked prediction, rebinding control)
    model.eval()
    if probes is not None:
        for li, layer in enumerate(recurrent_layers(model)):
            layer.probes = probes.probe_vectors(li).detach()
    egen = torch.Generator(device=device).manual_seed(seed + 10_000)
    correct = total_n = rebind_correct = 0
    probe_acc = [0.0] * layers
    vpos = torch.tensor([2 * p + 1 for p in range(pairs)], device=device)
    with torch.no_grad():
        for _ in range(eval_iters):
            b, table_start, values = make_long_retrieval_batch(
                eval_batch, seq_len, vocab, mask_id, pairs=pairs, n_queries=n_queries,
                device=device, generator=egen)
            logits = model(b.corrupted, b.t)
            if probes is not None:
                _, accs = aux_loss_and_acc(model, probes, table_start, values, "end")
                probe_acc = [a + c / eval_iters for a, c in zip(probe_acc, accs)]
            pred = apply_subs_sample(logits, b.corrupted, mask_id)
            m = b.target_mask
            correct += int(pred[m].eq(b.clean[m]).sum())
            total_n += int(m.sum())
            rc = b.corrupted.clone()
            rc[:, vpos] = b.corrupted[:, vpos.roll(-1)]
            rlogits = model(rc, b.t)
            rpred = apply_subs_sample(rlogits, rc, mask_id)
            rebind_correct += int(rpred[m].eq(b.clean[m]).sum())
    return {
        "seed": seed, "aux_mode": aux_mode, "aux_weight": aux_weight, "curriculum": curriculum,
        "n_queries": n_queries, "seq_len": seq_len, "pairs": pairs, "d_model": d_model,
        "layers": layers, "direction": direction, "steps": steps, "lr": lr, "batch": batch,
        "params_model": nparams, "final_task_loss": last,
        "eval_accuracy": correct / total_n, "rebind_accuracy": rebind_correct / total_n,
        "eval_samples": total_n, "probe_end_accuracy": probe_acc if probes is not None else None,
        "value_marginal_floor": round(1.0 / (vocab - 1 - (5 + pairs)), 5),
        "curve": curve, "seconds": round(time.time() - t0, 1),
    }


ARMS = {
    "A0_naive":        dict(n_queries=1, curriculum=False, aux_mode="none"),
    "A1_dense":        dict(n_queries=4, curriculum=False, aux_mode="none"),
    "A2_dense_curric": dict(n_queries=4, curriculum=True,  aux_mode="none"),
    "A3_dense_local":  dict(n_queries=4, curriculum=False, aux_mode="local"),
    "A4_dense_end":    dict(n_queries=4, curriculum=False, aux_mode="end"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out) / f"{args.arm}_seed{args.seed}.json"
    if out.exists():
        print("SKIP", out)
        return
    ck = Path(args.out).parent / "ckpt" / f"{args.arm}_seed{args.seed}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    r = run_trial(args.seed, steps=args.steps, ckpt_path=ck, **ARMS[args.arm])
    r["arm"] = args.arm
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1) + "\n")
    ck.unlink(missing_ok=True)
    print(f"DONE {args.arm} seed{args.seed} acc={r['eval_accuracy']:.4f} "
          f"rebind={r['rebind_accuracy']:.4f} probe={r['probe_end_accuracy']} s={r['seconds']}")


if __name__ == "__main__":
    main()
