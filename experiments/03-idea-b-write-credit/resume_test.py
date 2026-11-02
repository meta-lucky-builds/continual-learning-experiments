"""Resumed run must equal an uninterrupted run exactly (both a plain and an aux arm, with curriculum)."""
import sys, tempfile, os, torch
sys.path.insert(0, "/mnt/project-files/idea-b-write-credit"); import hay
torch.set_num_threads(4)
for kw in (dict(n_queries=4, curriculum=True, aux_mode="none"), dict(n_queries=4, curriculum=False, aux_mode="local")):
    full = hay.run_trial(0, steps=12, batch=16, eval_batch=32, eval_iters=1, log_every=1, **kw)
    ck = tempfile.mktemp(suffix=".pt")
    assert hay.run_trial(0, steps=12, batch=16, eval_batch=32, eval_iters=1, log_every=1, ckpt_path=ck, ckpt_every=5, stop_after=7, **kw) is None
    res = hay.run_trial(0, steps=12, batch=16, eval_batch=32, eval_iters=1, log_every=1, ckpt_path=ck, ckpt_every=5, **kw)
    for k in ("final_task_loss", "eval_accuracy", "rebind_accuracy", "probe_end_accuracy", "curve"):
        assert full[k] == res[k], (k, full[k], res[k])
    os.unlink(ck)
print("resume parity: OK")
