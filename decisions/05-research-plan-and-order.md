# Decision 05: The research claim and the order of work

**Dates:** 2026-10-20, updated 2026-10-31

## The claim worth making

A hierarchical latent planner that keeps solving long-horizon tasks when the world changes, where non-adaptive planners collapse, and that buys more success with more test-time compute (adaptation steps and planner loops).

We'd show it as **curves, not single numbers**:
- success against adaptation episodes;
- success against planner compute.

To our knowledge no paper shows this for latent world-model planning. That is our inference from what we've read.

## Why the parts fit together

| Part | Fixes | Touches |
|---|---|---|
| H-JEPA hierarchy | long horizons | frozen encoders, predictors |
| JEPA-TTT | **model error**: predicted dynamics wrong after a shift | predictors only (low-rank) |
| Looped planner | **optimization error**: solver stops short of a better plan | planner only |

They fix different failures and touch different parts, so their effects can be attributed.

## The diagnostic that decides what to build first

On failed tasks, compare the model's cost of the dataset's reference action sequence, J(u_ref), with the cost of the plan it found, J(u_found).
- If J(u_ref) < J(u_found), the model knew a better plan and the solver missed it, so **the looped planner** comes first.
- Otherwise the model itself is wrong, so **TTT** comes first.

The shift benchmark's 3× iterations headroom condition is a cheaper proxy for the same question.

## Order

1. ✓ Reproduce the baseline ([test 03](../tests/03-fourroom-l3-seed42-reproduction.md))
2. ✱ **"Before" shift benchmark on FourRoom** ([test 05](../tests/05-kaggle-before-benchmark.md)). If H-JEPA doesn't drop under shift, TTT has nothing to show and we rethink.
3. JEPA-TTT, predictors only, level 1 first, then upper levels. Rerun the same benchmark: the "after".
4. Looped planner at equal planner compute.
5. Looped predictors only if real model error remains.
6. AntMaze shifts (friction, mass, weak or disabled leg).
7. Real robot: SO-101 ([decision 06](06-so101-real-robot.md)).

Each step must beat the previous one at the same planning compute, and each has a kill criterion, so a failure gets dropped instead of patched.

## Fast weights

**Decision:** JEPA-TTT first. Fast weights are a later complement, not a replacement.

- H-JEPA plans with one step of context, so a sequence-level fast-weight memory has nothing to accumulate.
- Inside the planner's imagined rollout there's no ground truth to learn from.
- **Where fast weights would help:** within-episode adaptation at level 1, from a short history of real transitions, with no gradient steps. That gives two timescales: fast weights within an episode, and TTT across episodes.

## Honest odds (our judgement)

- Steps 2–3 have a good chance of a solid result.
- Step 4 is the high-risk, high-reward part.
- Calling it a breakthrough would need large gaps under shift, or the real robot.
