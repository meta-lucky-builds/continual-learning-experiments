"""Check that hay.py reproduces the original dualgoose harness exactly (aux off), and that the aux
code path does not change the forward computation.

usage: python parity_test.py /path/to/dualgoose_repo
"""
import sys

import torch

sys.path.insert(0, "/mnt/project-files/idea-b-write-credit")
import hay  # noqa: E402

repo = sys.argv[1]
sys.path.insert(0, repo)
sys.path.insert(0, repo + "/scripts")
import long_retrieval_probe as orig  # noqa: E402

torch.set_num_threads(4)
dev = torch.device("cpu")

# 1) identical init + forward + loss for armed_gdn_fwd (2 layers, d=32, L=64, vocab=64)
for seed in (0, 3):
    torch.manual_seed(seed)
    m_o = orig.build("armed_gdn_fwd", 64, 64, 32, 2, 2, 16)
    torch.manual_seed(seed)
    m_h = hay.TinyDualGooseDenoiser(64, 64, d_model=32, n_layers=2, direction="forward", short_conv=4)
    so, sh = m_o.state_dict(), m_h.state_dict()
    assert list(so) == list(sh), (list(so), list(sh))
    for k in so:
        assert torch.equal(so[k], sh[k]), k
    g1 = torch.Generator().manual_seed(seed)
    g2 = torch.Generator().manual_seed(seed)
    bo = orig.make_long_retrieval_batch(16, 64, 64, 63, pairs=4, n_queries=4, gap=None, device=dev, generator=g1)
    bh, ts, vals = hay.make_long_retrieval_batch(16, 64, 64, 63, pairs=4, n_queries=4, gap=None, device=dev, generator=g2)
    assert torch.equal(bo.clean, bh.clean) and torch.equal(bo.corrupted, bh.corrupted)
    assert ts == 0 and torch.equal(bh.clean[:, 1:8:2], vals)
    lo, lh = m_o(bo.corrupted, bo.t), m_h(bh.corrupted, bh.t)
    assert torch.allclose(lo, lh, atol=1e-6), (lo - lh).abs().max()
    from dualgoose.diffusion import masked_nll_loss as mo_loss
    a, b = mo_loss(lo, bo.clean, bo.corrupted, 63, t=bo.t), hay.masked_nll_loss(lh, bh.clean, bh.corrupted, 63, t=bh.t)
    assert torch.allclose(a, b, atol=1e-6)
    # probes on: forward output must be unchanged, reads must equal S_t p computed independently
    P = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
    for layer in hay.recurrent_layers(m_h):
        layer.probes = P
    lp = m_h(bh.corrupted, bh.t)
    assert torch.equal(lp, lh)
    # curriculum gap batches identical too
    g1 = torch.Generator().manual_seed(7); g2 = torch.Generator().manual_seed(7)
    bo = orig.make_long_retrieval_batch(8, 64, 64, 63, pairs=4, n_queries=4, gap=13, device=dev, generator=g1)
    bh, ts, vals = hay.make_long_retrieval_batch(8, 64, 64, 63, pairs=4, n_queries=4, gap=13, device=dev, generator=g2)
    assert torch.equal(bo.clean, bh.clean) and ts == 64 - 8 - 13 - 8
    assert torch.equal(bh.clean[:, ts + 1:ts + 8:2], vals)
print("init/forward/loss/data parity: OK")

# 2) probe reads equal an independent recomputation of S_t p (single layer, small)
torch.manual_seed(1)
layer = hay.DeltaNetRefLayer(32, gated=True, short_conv=4)
x = torch.randn(3, 10, 32)
P = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
layer.probes = P
layer(x)
reads = layer.probe_reads
# recompute state independently
with torch.no_grad():
    parts = torch.split(layer.proj(x), [32, 32, 32, 1, 32], dim=-1)
    qkv = torch.cat(parts[:3], -1).transpose(1, 2)
    qkv = torch.nn.functional.silu(layer.qkv_conv(torch.nn.functional.pad(qkv, (3, 0))).transpose(1, 2))
    q, k, v = torch.split(qkv, 32, -1)
    k = torch.nn.functional.normalize(k, dim=-1); beta = torch.sigmoid(parts[3]); w = torch.sigmoid(parts[4])
    S = torch.zeros(3, 32, 32)
    for t in range(10):
        S = S * w[:, t, None, :]
        S = S + (beta[:, t] * (v[:, t] - torch.einsum('bij,bj->bi', S, k[:, t])))[:, :, None] * k[:, t, None, :]
        ref = torch.einsum('bij,pj->bpi', S, P)
        assert torch.allclose(reads[:, t], ref, atol=1e-5)
print("probe read = S_t p: OK")

# 3) whole short training run identical to the original run_trial (aux off), incl. eval + rebind
ro = orig.run_trial("armed_gdn_fwd", 64, 0, pairs=4, d_model=32, layers=2, heads=2, vocab=64, steps=20,
                    lr=1e-3, batch=32, eval_batch=64, eval_iters=2, device=dev, mamba2_d_state=16,
                    fp32=True, n_queries=4, curriculum=True, rebind_eval=True)
rh = hay.run_trial(0, steps=20, batch=32, eval_batch=64, eval_iters=2, n_queries=4, curriculum=True)
print("orig:", ro["final_loss"], ro["eval_accuracy"], ro["rebind_accuracy"])
print("hay :", rh["final_task_loss"], rh["eval_accuracy"], rh["rebind_accuracy"])
assert abs(ro["final_loss"] - rh["final_task_loss"]) < 1e-5
assert ro["eval_accuracy"] == rh["eval_accuracy"] and ro["rebind_accuracy"] == rh["rebind_accuracy"]
print("training-run parity: OK")
