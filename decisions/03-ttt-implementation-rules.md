# Decision 03: Implementation rules for TTT in the H-JEPA code

**Date:** 2026-10-30
**Basis:** every equation in both papers matched line by line to the H-JEPA code at commit `97054ec`. Details:
- [`analysis/hjepa-math-and-code.md`](../analysis/hjepa-math-and-code.md)
- [`analysis/jepa-ttt-math-and-code.md`](../analysis/jepa-ttt-math-and-code.md)

JEPA-TTT released no code. Its world model is LeWM, which is H-JEPA's level 1, so we take its update rule and keep H-JEPA's planner.

## Rules, each found in the code

| Rule | Why (what the code does) |
|---|---|
| **Call `model.zero_grad()` before every TTT step** | The planner's `cost.backward()` accumulates `.grad` on the world-model parameters across all 90 iterations and never clears them; only the action optimizer is zeroed (`gd.py`). A naive TTT step would apply that leftover gradient. |
| **Compute the TTT loss with the model in eval mode** | The projectors use BatchNorm and dropout is 0.1. In train mode, `encode()` also pushes into the latent-action queue that the upper-level planners seed from (see [test 01](../tests/01-colab-eval-cpu-dry-run.md)). |
| **TTT windows are 3 timesteps** | The predictor's positional embedding has only `history_size = 3` positions. |
| **Loss = one-step teacher-forced prediction loss, no SIGReg** | This is JEPA-TTT's loss. SIGReg prevents collapse while training an encoder, and our encoders are frozen. |
| **Hook point: `WorldModelPolicy.get_action`** | The single place where every replan goes through. |
| **Keep H-JEPA's gradient planner and goal images** | JEPA-TTT plans with CEM and a learned reward head. We adopt its update rule, not its planner, so we stay comparable to H-JEPA. |
| **Start from a no-op** | Zero-initialised low-rank Δ, so step 0 is exactly the pretrained model. |

## Observations from the JEPA-TTT paper that shaped this

- **It updates very little:** one gradient step per 64 new windows, about 240 steps over 500 episodes. The adaptation is light, so it fits on a T4.
- **An unremarked result:** a per-episode variant with batch size 8 scores better than the main method (0.729 against 0.678). Our guess is that it makes about 8× more updates, which suggests update count matters more than the paper says. We'll treat updates per episode as a tuned hyperparameter.

## Stability without pretraining

- TTT needs no offline training. It *is* training, done online.
- Stabilizers: zero-init Δ, the proximal term, gradient clipping.
- Tune learning rate and μ on a *different* shift from the one we report, so we aren't tuning on the test set.
- **Acceptance check:** with every new part at its initial value, the code must reproduce the baseline (96% FourRoom) before any adaptation. If it doesn't, there's a bug.
