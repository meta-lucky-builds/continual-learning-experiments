# Decision 01: How JEPA-TTT goes into three-level H-JEPA

**Date:** 2026-10-20, revised 2026-10-31
**Basis:** full reads of both papers (H-JEPA arXiv 2610.06805, JEPA-TTT arXiv 2610.00722) and the H-JEPA code. Details: [`analysis/hjepa-ttt-design.md`](../analysis/hjepa-ttt-design.md) and [`analysis/hjepa-ttt-loop-math.md`](../analysis/hjepa-ttt-loop-math.md).

This is a design decision, not a test. It records what we decided and why, so later tests can be read against it.

## The question

JEPA-TTT adapts a flat world model (LeWM) online when the dynamics change. H-JEPA stacks three world models, each with its own encoder, latent space and predictor. How do we add test-time training without breaking the hierarchy?

## What we decided

1. **Freeze every encoder at every level.**
2. **Freeze the action encoders too.** This revises the 2026-10-20 plan, which adapted the level-1 action encoder.
3. **Adapt only the predictors**, using a small low-rank update Δ with a proximal pull back to the pretrained weights: loss = prediction loss + (μ/2)‖Δ‖².
4. **Weight levels by their data**: level 1 adapts most, upper levels gently or only when their own prediction error rises.
5. **Keep a safe fallback planner.** H-JEPA's flat level-1 planner, scored with the upper-level goal cost, uses only the fastest-adapting level.

## Why

**Encoders have three jobs in H-JEPA.** They produce the training targets, they embed the goal image, and they define the "lift" that compares a lower level's rollout with the subgoal from the level above. Moving one silently changes all three. Frozen encoders also mean the anti-collapse terms (SIGReg, IDM) aren't needed at test time.

**Action encoders define macro-actions.** Each level's action embeddings are pooled into the level above, and the upper planners seed candidates from a stored queue of past macro-actions (the latent-action queue). Adapting an action encoder would quietly change the meaning of every stored macro-action and of everything above it. Flat JEPA-TTT can adapt its action encoder because it has no level above.

**Upper levels see far less test-time data.** With stride 2 per level, level ℓ sees about N₁ / 2^(ℓ−1) windows. A level-3 window spans about 61 env steps. Adapting upper levels at the same rate as level 1 would overfit, so we use dense windows at every env-step offset and a lower learning rate.

**Which levels need to adapt depends on the shift:**
- A rotated or remapped action changes how a raw action turns into motion, which is level 1's job, so adapting level 1 alone may be enough.
- A change in speed or friction changes which states one upper-level step can reach, so the upper predictors need to adapt too.
- This is a testable prediction, built into the shift benchmark's per-level model error ([test 05](../tests/05-kaggle-before-benchmark.md), prediction 2).

**The main risk is the levels drifting apart.** If level 3 proposes subgoals that the adapted level 1 can no longer reach, the hierarchy fails. The error gate and the fallback planner guard against that.

## Where it should and shouldn't help

- **Should help:** AntMaze and FourRoom. Their episodes are long (350 and 210 steps), three levels help there, and there's enough test-time data.
- **Won't help much:** Push-T three-level (already broken by data starvation in the paper) and Cube (a 30-step budget means level 3 is barely used).
