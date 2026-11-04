# Could looping work in H-JEPA? A first-principles feasibility note

Status: opinion and feasibility only. This is not a framework design. Claims from papers are cited; everything marked **(inference)** is my reasoning and has not been tested.

## Short answer

- **Yes, we could design a looping framework for H-JEPA, and part of it would probably work.** It would not come from naively looping every predictor. It would come from putting loops where the October 2026 literature says loops pay off.
- **Where the evidence says looping pays off:**
  - Decisions that need multistep planning. Looped Actor matches a 16x larger policy and roughly doubles Boxoban success.
  - Continuous refinement, but only with deep supervision on every loop and gated or modulated updates (Looped-DiT).
  - Adaptive depth, where hard items get more loops and easy items exit early (TaH2, Looped Actor).
- **Where it does not pay off:**
  - Uniform fixed-depth looping at matched compute. TaH2 found post-trained looped models underperform non-looped ones, and many tokens get no better or worse with more loops.
  - Running far beyond the trained depth. There is a finite extrapolation window, and outside it the model produces confident false outputs.
- **For H-JEPA specifically, I rank the candidates as:**
  1. Looping the planner, the strongest candidate.
  2. Adaptive looping of the upper-level predictors.
  3. Uniform looping of every predictor, the weakest.

## 1. What H-JEPA already iterates

H-JEPA already has two iterative axes, and looping would add a third.

| Axis | What iterates | Today |
|---|---|---|
| Time | Autoregressive rollout of `F(ℓ)` over the horizon | Fixed per level (strides 5/10/20) |
| Optimisation | Planner refines the action sequence | AdamW, up to 90 steps, S candidates |
| Depth (new) | The same block applied K times per prediction | Not present; 4 to 6 distinct layers |

So the real question is which axis is compute-starved. From the H-JEPA paper, the failures I saw are about **cost-landscape geometry and horizon**:
- the upper-level subgoal is unreachable or misleading;
- gradient planning gets stuck.

Predictor capacity per step is not the main failure. The predictors are small (width about 200), and adding layers was not reported as the fix.

**(inference)** Looping depth inside `F(ℓ)` attacks capacity, which is the less likely bottleneck. Looping the optimisation axis attacks the likely bottleneck.

## 2. The three candidate placements

### A. Loop the predictor `F(ℓ)` (depth looping)

**Mechanism.** Use a prelude, then a shared block applied K times, then a coda. Inject `z_t` and the action embedding on every loop, as the Part I topology study and LoopWM recommend.

**Why it might help (inference):**
- A level-3 transition covers 20 env steps. That is a "hard token": a long, contact-rich jump. Level-1 steps of 5 env steps are mostly easy.
- This matches the TaH2 and Looped Actor finding that gains concentrate on hard items. Adaptive halting would therefore spend loops at upper levels and on contact events, and almost none on free motion.

**Why it might not help:**
- **Planner cost multiplies.** The planner backpropagates through the rollout about 90 times per replan. A K-loop predictor multiplies forward and backward cost by about K unless loops are truncated or adaptive. At matched FLOPs, that budget could instead buy more planner iterations or candidates. TaH2's matched-compute result warns that this trade is often lost.
- **Contractive loops flatten action sensitivity.** Stable looped models (Parcae and LoopWM retention with eigenvalues in (0, 1), and FPRM) converge to a fixed point that depends weakly on the initial state. In H-JEPA the planner needs `∂ẑ/∂a` to be informative. **(inference)** If the action is injected only at the start, contraction washes it out. Re-injecting it every loop is mandatory, not optional.
- **The extrapolation window.** Too many loops produce confident false outputs. In a world model that means plausible-looking but physically impossible next states. The planner will happily exploit those, which is worse than an honest blurry prediction.
- **Knowledge-like recall gets worse with more loops** (What Makes Recurrence Effective). Much of what a JEPA predictor does is recall-like: how this object moves under this push. **(inference)** That suggests a modest K, not deep recurrence.

**Verdict.** Worth trying only as adaptive looping at level 2 and level 3, with per-loop supervision. Expect at best modest gains in prediction error, and no guaranteed gain in planning success.

### B. Loop the planner (optimisation looping)

**Mechanism (inference, built from the literature):**
- *Looped Transformers as Optimizers* (2609.37379) shows that a looped transformer's state behaves like a fast weight, with each loop acting as a gradient step on a local objective.
- H-JEPA's planner is literally an optimisation loop: AdamW over actions against a latent goal cost.
- A **learned looped planner** would take (current latent, goal latent, current plan) and output a refined plan each loop. It would halt when consecutive plans stop changing, in the style of Looped Actor's KL halting.

**Why this is the strongest candidate:**
- It targets the actual bottleneck: local minima and slow convergence of gradient planning.
- Looped Actor reports that future plan steps become more decodable over loops, and that its biggest gains are exactly on multistep planning.
- **Fixed-point reuse** (Looped Models Done Right, Part II; DEQ-Flow; StreamDEQ): in MPC you replan every step from an almost identical state. Warm-starting the looped planner from the previous solution should cut loops sharply. That is the same reason >95% of Looped Actor decisions halt early.
- It is fully compatible with the hierarchy. You could run one looped planner per level, or one shared planner conditioned on the level, while the frozen world model stays the objective. The world model remains the ground truth, so a bad planner shows up as a high cost that you can verify, not as a silent hallucination.
- There is a fallback. If the looped planner's cost is worse than a few AdamW steps would reach, finish with AdamW. Most amortised-optimisation work uses this hybrid.

**Risks:**
- **Distribution shift.** A planner trained on offline data may not generalise to new goals. That is exactly where JEPA-TTT adaptation lands, and it is a nice coupling.
- **Training needs targets.** Options are distillation from the existing gradient planner or RL. Distillation caps you at the teacher unless the planner learns to escape the teacher's local minima. **(inference)** Truncated-BPTT training against the world-model cost directly, with no teacher, is the more interesting version, but it is riskier.

### C. Hierarchy as recurrence (HRM/TRM-style)

- H-JEPA's levels already resemble a slow/fast recurrence, with a high level setting subgoals for a low level. HRM and TRM make that explicit with nested loops.
- Part I (2610.23036) found that separate H/L split states gave **no consistent gain** over a single looped state.
- **(inference)** Rebuilding H-JEPA's hierarchy as one looped network is not supported by current evidence. H-JEPA's explicit levels with different strides give interpretable subgoals and selective abstraction, which a merged loop would lose.

## 3. Literature lessons that transfer, and what each implies here

| Lesson (paper) | Implication for H-JEPA (inference) |
|---|---|
| Truncation, not full BPTT, shapes useful fixed points (2610.06833) | Train looped parts with truncated backprop over the last b loops; that also keeps memory bounded inside the planner's own backprop |
| Implicit/Neumann endpoint gradients collapse from init (2610.06833) | Do not start with DEQ-style implicit gradients; get a stable fixed point first |
| Uniform looping loses at matched compute; adaptive depth with lookahead supervision wins (2609.35748) | Every looped component needs a halting head trained on actual gain, and every comparison must be at matched planner FLOPs |
| Finite extrapolation window; it widens when training depth exceeds training hardness (2610.14456) | Train with more loops than the hardest transitions need, and cap test-time K; never extrapolate depth freely |
| Deep supervision on every loop plus gated updates prevents saturation in continuous models (2609.40305) | Apply the SIGReg/prediction loss to every loop's output, and gate the residual update |
| History-state injection and timestep conditioning beat initial-state injection (2609.36636) | Feed the loop index and re-inject `z_t` and the action every loop |
| Prelude–loop–coda sandwich (2610.23036, LoopWM) | Keep `E(ℓ)` as the prelude and a small coda before the predicted latent |
| Slow oscillatory modes bottleneck convergence (2610.18174) | Use momentum-style halting in the planner loop rather than damping |
| Residual α/β scaling keeps iterates bounded (2609.37432) | Bound the iterates cheaply, without hard contraction that kills action sensitivity |

## 4. How much to trust the world-model evidence

- **Looped World Models (2606.18208)** is the only direct world-model looping paper I found, and its evidence is weak for us:
  - Its results are on **text** environments (ScienceWorld and ALFWorld state prediction), compared against general LLMs.
  - It does not compare against a non-looped model of matched size.
  - The continuous visual results are only described ("loss is consistently reducible"), not shown.
  - It does not test planning success.
  - Its useful pieces are architectural: Parcae-style contractive retention, Poisson loop sampling with truncated backprop, an exit gate with entropy regularisation, and nested inner and outer loops. None of these has been validated on a JEPA planner.
- The strongest transferable evidence comes from **Looped Actor** (planning RL) and **Looped-DiT** (continuous latent refinement). Neither is a JEPA world model.
- **Bottom line (inference):** there is no published evidence yet that looping improves latent-planning success. A positive result for H-JEPA would be new.

## 5. Where looping should not be expected to help

- **Push-T and Cube.** H-JEPA's gains there are already small or negative. Looped Actor also only ties on OGBench Cube, Scene and Puzzle. Looping adds no information about contact.
- **Level 1 at short horizons.** These transitions are easy, so adaptive halting would exit at K=1 anyway.
- **Anything where the bottleneck is the encoder or representation** (selective abstraction, slow features). Looping the predictor cannot fix a latent space that does not contain the right information.

## 6. Cheapest experiments that would answer the question

1. **Looped level-2 predictor, adaptive halting.**
   - Compare against the baseline predictor at matched planner FLOPs on AntMaze.
   - Measure prediction error by horizon, planning success, and the K distribution across transitions.
   - Kill criterion: no success gain at matched FLOPs.
2. **Looped amortised planner.**
   - Distil it from the AdamW planner, then fine-tune with truncated BPTT against the world-model cost, warm-starting across replans.
   - Plot success against planning TFLOPs; the gradient planner is the baseline curve.
   - Kill criterion: the looped planner never beats AdamW at equal compute, even with a short AdamW finish.
3. **(Optional) Interaction with JEPA-TTT.** A looped predictor has fewer distinct weights, so test whether online adaptation is more stable or less stable. **(inference)** It could go either way, because a shared weight changes every loop at once.

Experiment 2 is where I would put the first effort.

## 7. Papers used

- Looped Models Done Right, Part II: https://www.alphaxiv.org/abs/2610.06833
- Part I: Topology: https://www.alphaxiv.org/abs/2610.23036
- TaH2 / Adaptive Looped Transformers: https://www.alphaxiv.org/abs/2609.35748
- What Makes Recurrence Effective: https://www.alphaxiv.org/abs/2609.36636
- Looped Actor: https://www.alphaxiv.org/abs/2609.37432
- Looped Transformers as Optimizers: https://www.alphaxiv.org/abs/2609.37379
- Looped Reasoning Finishes Earlier (QRR): https://www.alphaxiv.org/abs/2610.18174
- Learning to Extrapolate in Depth: https://www.alphaxiv.org/abs/2610.14456
- Looped-DiT: https://www.alphaxiv.org/abs/2609.40305
- Looped World Models: https://www.alphaxiv.org/abs/2606.18208
