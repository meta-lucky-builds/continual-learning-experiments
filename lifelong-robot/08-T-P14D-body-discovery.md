# 08 T-P14D: can a robot discover its own body's structure through varied practice, and carry that ability to new motor tasks?

**Date run:** 2026-10-19 (sanity runs, pilots and amendment pilots, all on that day). **Folder:** /mnt/project-files/research/tests-v6/p14d/ **Built by:** an independent test builder (the "Test-B builder"); the candidate designs came from two separate P14 architects (p14-v6, p14-v7). **Seeds:** sanity 1-4; pilot 101-108 (8 seeds), untuned default configurations. Tuning seeds (9001-9003) and main seeds (1..48) were **never run**. **Verdict:** PILOT ONLY. No arm of any kind (competitor, existing-methods stack, or our designs) showed the full P14 pattern in the pilot. Our p14v6 design was far below the generic value-carrying competitors (G at mid overlap 0.09 vs 0.47-0.58). The decisive main run never happened, so there is no pre-registered verdict.

---

## 1. Why we ran it

**The problem.** P14 in notes/problems.md is "transferable, task-agnostic motor memory". Lucky's meaning, clarified on 2026-10-18 and quoted in PROTOCOL.md §1: a robot that has practised hand-eye coordination and reflexes in one sport learns another sport that needs them faster. The gain is

- proportional to how much ability the new task shares with what was practised;
- near zero when nothing is shared (a pro gamer gains about nothing at chess or meditation);
- never negative;
- larger after varied practice than after narrow practice;
- kept for life (not overwritten);
- mainly faster learning; a better first try is only a bonus.

What transfers is motor **ability**, not a parameter starting point (memory note problems-self-motor.md, clarification 2026-10-18 18:32-18:34 UTC). In this test the robot is also **not told its body's structure**: P14 is treated as discovering that structure (timing/delay, the noise law, the slip signature, its own dynamics) from task-tied practice and carrying it to new tasks.

**Why this test at that moment.** The previous round's P14 design ("L", clusters-v5) had been found fatal by an audit (AUDIT-L), because its toy *handed* the body's "grammar" (the forms of delays, noise and slips) to L's learner only; a learner given the same grammar with no practice learned in 2-8 episodes and beat practised L in 20 of 30 cells (p14-v6/BRIEF.md). T-P14D was built so that:

- the body's structure is hidden from every learner;
- a privileged reference arm that *is* told the structure (`gram`) defines what "discovery" would mean;
- strong existing methods (Reptile, RMA, adapters, multi-task model, and later PBML-style priors and an existing-methods stack `heb`) are tuned and run with the same data and knowledge;
- the candidate can clearly lose.

The rule of the round (PROTOCOL.md §1): "the builder did not design any mechanism, and the test must be able to make the candidate lose." It was meant to settle whether any P14 design (ours or anyone's) shows the P14 pattern, and whether ours beats the best competitor by a clear margin.

---

## 2. What we did

### 2.1 The world

Source: IMPLEMENTATION.md §1, INTERFACE.md §1. [checked: read]

Two simulated effectors with different hidden "grammars":

| | E1 (practised) | E2 (never practised: "the other game") |
|---|---|---|
| body | planar 2-joint arm, 4 states, 2 commands | 1-joint pointer, 2 states, 1 command |
| delay (ability A1, timing) | integer shift of the command | fractional shift |
| noise (A2, precision) | grows with the command along a body direction | grows with velocity |
| slips (A4) | impulse along a body-constant direction | one-sided push |
| load (A3, control under difficulty) | rank-1 force field, fresh per task | same |
| body dynamics (A5) | body constants (w-body, w-offgram, w-dynonly) or redrawn per task (w-fresh, w-lhome) | own constants |

Each test task has a "demand vector" saying which abilities it draws on; learners never see it. "Declared overlap" with practice is computed from it (cells o25, o50, o75, o100). Other cells: `E2_0` (other effector, nothing shared), `E1_0` (same effector, only the unpractised ability), `conf` (conflict: o50-type tasks with the command mapping reversed), `var` (small variations of early practice tasks), and after a separate-abilities life, `comb` (A1 and A2 together) and `ood` (delay and noise strength beyond the practised range).

**Learning is system identification; the controller is shared.** Each learner outputs a linear one-step model Θ of the state increment. The harness turns Θ into the *same* model-predictive controller (MPC, horizon 14) for every arm and scores closed-loop success on 32 rollouts with common random numbers. So arms differ only in how well and how fast they estimate the body from data. A world where abilities act through control (reflex timing) was not built (IMPLEMENTATION.md §1, choice 1).

**Data.** Every learner gets exactly the same pre-generated episodes from a fixed random behaviour policy (AR(1) commands). No arm chooses its own data (except in the later w-active world).

**Lives per job** (PROTOCOL.md §2): V = varied practice, 24 tasks x 20 episodes on E1 (snapshots after 6 and 12 tasks), then 24 load tasks (S2) and a retention test; N = narrow practice, one task repeated 24 x 20 episodes; S = separate-ability practice (A1-only and A2-only) then combination and out-of-range tests. V and N have exactly the same number of real episodes (480) and frames (3840) [checked: sanity/SANITY.md].

**Worlds:** `w-body` (headline: everything about the body is constant), `w-lhome` (noise direction and slip signature constant, dynamics fresh per task; the home world of the L-reduced arm), `w-fresh` (nothing shared but the forms; kill world), `w-offgram` (noise follows velocity, which breaks an L-style assumption), `w-dynonly` (only dynamics shared). A sixth world, `w-active`, was added by amendment 4 (§2.6). The pilots ran **only w-body and w-lhome** (plus w-active for some arms).

### 2.2 Metric and jargon

- **EtC (episodes to criterion):** the first checkpoint (0,1,2,...,64 real episodes) at which closed-loop success reaches 0.8 x the expert's success (MPC with the true plant), confirmed at the next checkpoint. Censored at 64 (counted as 65). **EtC+ = EtC + 1**, so a first-try success counts as 1. Analysis is on log EtC+.
- **G (gain vs fresh):** G = 1 - exp(LE_arm - LE_fresh), where LE is the mean of log EtC+ over a cell's tasks. G = 0.33 means the geometric-mean EtC+ is 67% of the fresh learner's, i.e. about a third fewer episodes. G is unbounded below: G = -1 means twice as many episodes as fresh.
- **fresh:** the history-free learner (robust Student-t regression on the episode window). Every G is relative to it.
- **LSG (learning-speed gain):** compares the time from first reaching 0.3 x criterion to reaching criterion, arm vs fresh. It is meant to separate "learning faster" from "starting better".
- **DF (discovery fraction):** at o50, DF = (L_fresh - L_arm)/(L_fresh - L_gram) on the log scale: how far practice moves an arm from fresh towards the arm that is told the forms.
- **Value carriers:** arms that carry learned values (a starting point, a pooled model, an inferred latent) from practice to a new task. They give a better first try.
- **Race:** a windowed prequential comparison against the fresh learner that lets an arm fall back to fresh when its prior hurts. Offered as a tuned option to every history-using arm.

### 2.3 Arms

Source: PROTOCOL.md §2, §10b, §10c; IMPLEMENTATION.md §2-§4, §6, §8-§10. [checked: read]

**Baseline:** `fresh`.

**Privileged references (not competitors):**
- `gram` (grammar-given): told the forms of delay, noise, slip and kinematics, estimates every value per task from that task's own data, no practice, no body values.
- `oracle`: knows every value; reaches the criterion at the first try. A headroom reference.
- `oraclef` (oracle-factored): carries exactly the practised ability components, including the body-dynamics values. Used to check that each world has the structure it claims.
- Sanity-only: `naive` (very tight carry-over prior), `gram_cmd` (noise form forced to "command-dependent"), `gram_e1` (E1 forms applied to E2).

**Competitors** (all on the same window-regression model class, same data, same 200,000-float memory budget, same 16-configuration tuning budget):
- `reptile`: first-order Reptile. Reptile is a meta-learning method that, after each task, moves a shared initial parameter vector a little toward that task's solution, so new tasks start from a learned initialisation.
- `trunk`: a shared low-rank linear trunk with per-task heads (multi-task pretraining).
- `adapter`: a pooled base plus a library of low-rank per-task corrections; a new task blends stored adapters and fits a new correction.
- `rma`, `rma_ft`: an amortised identifier in the style of RMA (Rapid Motor Adaptation). RMA trains an encoder that maps a short history of states and commands to a latent "environment code", and a predictor conditioned on that code, so a new body or task is identified by one forward pass with no gradient. `rma_ft` also fine-tunes at test.
- `mbase`: one dynamics model pooled over all practice, with an empirical-Bayes per-coefficient prior.
- Amendment 1: `pbml` (a learned matrix-normal prior in the style of PBML, Huang and Michailidis, as described in BOTTLENECKS-LIT.md; PBML here means a meta-learned Gaussian prior over the per-task model with closed-form adaptation), `sigma_shrink` (per-module learned shrinkage, Chen et al.), `l2init` (L2 regularisation toward the initial weights, Kumar et al.), `cbp_lite` (continual backprop: a small MLP that reinitialises low-utility units, Dohare et al.).
- Amendment 3: `heb`, the "existing-methods stack": a hierarchical empirical-Bayes identifier per effector (pooled mean, low-rank plus diagonal covariance fitted by evidence across practised tasks, a per-task evidence scale, an optional pooled noise law), raced against fresh. It was declared the **main existing-methods bar**: a candidate counts as a breakthrough only if it also beats `heb` at <= 0.8 x EtC with p < 0.01 at mid and high overlap. Its ablations `heb_diag`, `heb_noeb`, `heb_nonoise`, `heb_norace`, `heb_meta` are diagnostic.

**Candidates:**
- `lred`: the "L-reduced" arm, the surviving part of the previous round's L design (learns only the noise-direction and slip-signature forms, then races against fresh).
- `p14v6g`, `p14v6p`, `p14v6nh`: the p14-v6 architect's design (§2.4).
- `p14v7`, `p14v7nm`, `p14v7g`, `p14v7w`, `p14v7nh`: the p14-v7 architect's design (amendment 2). Its design and result are explained in detail in file 09; this file reports only the rows.

### 2.4 The p14v6 design in plain words

Source: research/architecture/p14-v6/CORE.md (sharp idea, §6, §10); IMPLEMENTATION.md §6. [checked: read]

The bet: varied practice discovers the body's **invariants**: (1) coefficient *shapes* that stay valid under any change of task values (any lag, any gain, either sign), plus body constants shared by every practised task; and (2) the law of the body's own errors (how noise grows, what a slip looks like). A new task uses (1) only as **Stein shrinkage targets** for its own from-scratch estimate (it is pulled toward the shape by an amount set by its own data), and (2) only as per-sample weights. Because no task value is carried, the first try equals the fresh first try by construction. The architect argued "never negative" from the James-Stein/George shrinkage dominance theorem, which holds exactly only in the Gaussian linear case; for closed-loop episodes-to-criterion it was a sketch (CORE.md §10 lists "Never negative in EtC" as [sketch], open).

Arms: `p14v6g` = full (atoms may be closed over shifts: the "shift operator"); `p14v6p` = the same without the shift operator; `p14v6nh` = practice is a no-op (same per-task machinery, no practice). It carries a declared built-in language (shift operator, heteroscedastic noise law, scale mixture, constants, rank-1 coupling), so it is reported beside `gram`. The builder left out a change-point component the spec mentioned (flag inert), fixed the noise exponent at p = 1, and made other choices listed in IMPLEMENTATION.md §6.

### 2.5 Pre-registered decision rules

Source: PROTOCOL.md §4. Per world and arm, δ = 0.05:

| Property | Rule |
|---|---|
| P1 Proportional | mean per-seed Spearman ρ of G vs declared overlap over 6 levels ≥ 0.8, with lower bounds of ρ and of the OLS slope > 0 |
| P2 Near zero when nothing shared | E2_0: 90% t-interval of mean G inside [-0.05, 0.05] (TOST) |
| P3 Never negative | every cell: one-sided 95% lower bound of mean G ≥ -0.05 |
| P4 Varied beats narrow | G(V, o50) - G(N, o50) > 0, sign-flip p < 0.01 |
| P5 Not overwritten | G after S2 ≥ 0.8 x G after S1 (lower bound), and G after S1 > 0.05 |
| P6 Mainly faster learning | at o50, lower bound of mean(LSG - 0.5 G) ≥ 0, and G > 0.05 |
| P7 Grows with practice | G(o50) after 24 tasks > after 6 tasks, one-sided t, α = 0.05 |

An arm **shows P14 discovery** iff P1-P7 all hold AND DF ≥ 0.5 with lower bound > 0. A **breakthrough** additionally needs, against the best competitor (lowest mean log EtC+ over o50, o75, o100 in that world, chosen on the same data), a paired ratio ≤ 0.8 with sign-flip p < 0.01 at both mid and high overlap, and (amendment 3) the same against `heb`. A general claim needs the verdict in both w-body and w-lhome. A sensitivity sweep varies δ, the DF threshold, the breakthrough ratio, the criterion fraction (0.7/0.8/0.9), the overlap form and α.

**Power and seeds (PROTOCOL.md §7, §10f; POWER_amend.txt):** the decisive comparison is a per-seed paired log-ratio tested against log 0.8 at α = 0.01, powered for a true ratio of 0.65. From the pilot, the worst per-seed sd among headline candidates vs competitors was 0.44 (p14v7 vs adapter, w-body), giving n = (15.26 x 0.44)² = 45, so **N_MAIN = 48**. (An ablation pair had sd 0.53, giving 66; ablations are not decisive.)

**World validity checks (PROTOCOL.md §6):** "oracle validity" means checking, with the privileged reference arms only, that the world really has the structure it claims before any learner is judged on it. V1: oracle reaches criterion almost at once. V2: oraclef's G rises with overlap (Spearman ≥ 0.8, ≥ 0.9 in w-body) and |G(E2_0)| ≤ 0.05. V3: conflict tasks really conflict. V4: in w-offgram, the true noise form beats the wrong one by G ≥ 0.1. V5: E2 shares nothing. V6: w-fresh shares only forms.

### 2.6 Amendments and deviations (all before any main run)

- **Amendment 1** (approved by Lucky 12:17 UTC): four competitors added (pbml, sigma_shrink, l2init, cbp_lite); human-learning caveats added (Braun 2009 on matched variability; Weinstein and Botvinick 2017, which predicts the structure-learning advantage vanishes with about 5x more data per task), with a supplementary data-per-task sweep (5/20/80 episodes per practice task).
- **Amendment 2:** the p14v7 arms (file 09). Lucky's ruling at 12:50 UTC: p14v7's zero-mean learned metric counts as P14 "ability", because it never moves the first try.
- **Amendment 3:** `heb` and ablations; `heb` became the main existing-methods bar.
- **Amendment 4** (Lucky approved 12:50 UTC): world `w-active`. In all five original worlds the data come from a fixed behaviour policy, so no arm can choose how to excite the body. In w-active, while learning a test task, each arm chooses its own commands; a default "uncertainty-directed" rule was given to every arm (at each step pick, among 8 candidate commands, the one with the largest leverage under the task's own information matrix). The world counts only if choosing helps: the privileged oracle rule must lower fresh's mean log EtC+ by at least 0.15 versus the fixed policy.
- **§10d:** the builder checked numerically that across-task variation of the true Θ is low-dimensional (99%-variance rank 8 of 76 entries in w-body, about 12-13 in w-fresh/w-lhome).
- **§10e:** a note that `gram` is a **weak reference**: it knows forms but must estimate the body dynamics per task, so value carriers that pool dynamics get DF 1.5-2.0. "DF ≥ 0.5" therefore means "recovers at least half of what knowing the forms alone gives", not "close to knowing the body". A stronger reference `gramP` was proposed but **not built**; a sweep variant uses `oraclef` as the reference.
- **§11 Entry 0:** sanity validity outcomes: V1, V2, V5, V6 hold; V3 restated (oraclef conflicts only in worlds where it carries values); V4 only weakly met (gram minus gram_cmd = 0.12 at o100 and 0.02 at o50 in w-offgram, 4 seeds), so w-offgram is a weak kill world. The task generator threshold in code is expert ≥ 0.7 (the brief said 0.65); the code value is the registered one.
- Before any pilot, the builder moved slips into the first practice set S1 (an earlier draft lacked them, which would have starved the L-reduced arm) and logged it (IMPLEMENTATION.md §1).

---

## 3. What we got

All numbers are PILOT: 8 seeds (101-108), untuned default configurations, w-body and w-lhome only (plus w-active for 12 arms). The file headers say every number is computed from results/pilot.

### 3.1 World validity (references)

Source: research/tests-v6/p14d/pilot/PILOT_amend2-4.md §1 (same numbers in PILOT.md).

| reference | world | E2_0 | E1_0 | o25 | o50 | o75 | o100 | conf | var |
|---|---|---|---|---|---|---|---|---|---|
| gram | w-body | 0.19 | 0.26 | 0.27 | 0.37 | 0.26 | 0.53 | 0.38 | 0.46 |
| oracle | w-body | 0.74 | 0.78 | 0.81 | 0.86 | 0.94 | 0.95 | 0.85 | 0.90 |
| oraclef | w-body | 0.00 | 0.56 | 0.59 | 0.64 | 0.75 | 0.88 | -8.77 | 0.80 |
| gram | w-lhome | 0.16 | 0.24 | 0.29 | 0.42 | 0.51 | 0.54 | 0.41 | 0.50 |
| oracle | w-lhome | 0.74 | 0.78 | 0.82 | 0.89 | 0.94 | 0.96 | 0.87 | 0.91 |
| oraclef | w-lhome | 0.00 | 0.57 | 0.61 | 0.67 | 0.75 | 0.86 | 0.51 | 0.79 |

oraclef Spearman(G, overlap) = 0.87 (w-body; the V2 threshold for w-body is 0.9) and 0.90 (w-lhome). G(E2_0) of oraclef is exactly 0. In w-body, conflict tasks are strongly harmful to the value-carrying oraclef (G -8.77). Note that oraclef's G(E1_0) is 0.56-0.57: on the same effector, even a task drawing on no practised declared ability gains a lot, because the body dynamics (A5) are shared.

### 3.2 Competitors (the bar), mid-overlap gain and properties

Source: PILOT_amend2-4.md §2 (w-body and w-lhome). "X" = property fails. Only selected columns.

| arm | G(o50) w-body | G(o50) w-lhome | first try o50 (w-body) | properties passed, w-body | DF (gram), w-body |
|---|---|---|---|---|---|
| reptile | -1.18 | -0.43 | 0.52 | P2, P7 | -1.23 |
| trunk | -1.21 | -0.28 | 0.62 | P2 | -1.15 |
| adapter | 0.47 | 0.39 | 0.67 | P2, P3, P5 | 1.55 |
| rma | -0.10 | -0.01 | 0.67 | P2, P4, P5 | 0.91 |
| rma_ft | 0.47 | 0.31 | 0.67 | P2, P3 | 1.57 |
| mbase | 0.52 | 0.58 | 0.62 | P2, P3, P5 | 1.65 |
| pbml | 0.58 | 0.52 | 0.62 | P2, P5 | 2.00 |
| sigma_shrink | 0.49 | 0.57 | 0.62 | P2, P4, P5 | 1.51 |
| l2init | -7.22 | -5.08 | 0.24 | P2 | -4.42 |
| cbp_lite | -0.31 | -0.21 | 0.59 | P2, P7 | 0.27 |
| heb | 0.38 | 0.44 | 0.03 | P2, P5 | 1.21 |

(First try = success at episode 0 divided by expert success; fresh's is 0.03 in w-body.) P2 passes for every competitor almost automatically: on the never-practised effector E2 they have an empty history and fall back to fresh, so G(E2_0) is exactly 0.

No competitor passed P1 (proportionality) or P6 (mainly faster learning) in either world. The best Spearman ρ among competitors was heb's 0.53 (w-body; lower bound 0.34) and 0.52 (w-lhome). **No competitor showed P14 discovery.** The P3 column shows that adapter, rma_ft and mbase kept their worst cell lower bound at -0.03 to -0.04 in w-body, while other value carriers were badly hurt somewhere, most strongly in the conflict cell (rma -9.88, sanity arm naive -9.88 in w-body). Which default configurations had the race switched on is not printed in the pilot tables, so I cannot say how much of the protection came from the race.

### 3.3 Candidates

Source: PILOT_amend2-4.md §3. Ratios are paired geometric-mean EtC+ ratios (candidate over comparator); > 1 means the candidate needs more episodes.

| arm | world | G(o50) | P1 ρ (lower) | P2 E2_0 CI | P3 worst lower | P4 V-N (p) | P5 S1→S2 | P6 LSG | P7 6→24 tasks | DF (CI) | shows P14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lred | w-body | -0.00 | 0.45 (0.16) X | [0,0] | -0.05 X | 0.00 (1.0) X | 0.12→0.07 X | -0.00 X | 0.00→-0.00 X | -0.00 [-0.08,0.08] | no |
| lred | w-lhome | 0.07 | 0.56 (0.29) X | [0,0] | -0.05 X | 0.01 (0.25) X | 0.12→0.03 X | 0.20 | 0.06→0.07 X | 0.13 [0.01,0.24] | no |
| p14v6g | w-body | 0.09 | 0.31 (0.06) X | [0.00,0.14] X | -0.39 X | 0.08 (0.023) X | 0.26→0.25 | 0.11 X | 0.10→0.09 X | 0.21 [0.07,0.37] | no |
| p14v6g | w-lhome | 0.10 | 0.18 (-0.16) X | [0.01,0.07] X | -0.05 | 0.07 (0.008) | 0.11→0.14 | 0.15 | 0.10→0.10 X | 0.20 [0.09,0.30] | no |
| p14v6p | w-body | 0.09 | 0.32 (0.06) X | [0.00,0.14] X | -0.40 X | 0.08 (0.023) X | 0.26→0.25 | 0.11 X | 0.10→0.09 X | 0.21 [0.07,0.37] | no |
| p14v6nh | w-body | 0.01 | -0.64 (-0.83) X | [0.00,0.14] X | -1.02 X | 0.00 (1.0) X | -0.08→-0.08 X | -0.27 X | 0.01→0.01 X | 0.02 [-0.09,0.17] | no |
| p14v7 | w-body | 0.33 | 0.05 (-0.07) X | [0,0] | -0.02 | 0.03 (0.36) X | 0.32→0.40 | -0.30 X | 0.21→0.33 | 0.92 [0.59,1.30] | no |
| p14v7 | w-lhome | 0.43 | -0.02 (-0.25) X | [0,0] | -0.03 | 0.15 (0.008) | 0.31→0.33 | 0.10 X | 0.32→0.43 | 1.08 [0.69,1.52] | no |
| p14v7nm | w-body | 0.47 | 0.41 (0.28) X | [0,0] | 0.00 | 0.16 (0.023) X | 0.52→0.50 | 0.08 X | 0.36→0.47 | 1.46 [0.87,2.12] | no |

p14v6p's w-lhome row equals p14v6g's to two decimals except P3 (-0.06, fails) and P5 (0.12→0.14, marked as failing). p14v7g, p14v7w, p14v7nh rows are in file 09.

**Candidates vs the best competitor and vs heb** (PILOT_amend2-4.md §3, "vs best competitor" lines; mid = o50, high = o75 and o100 pooled):

| arm | world | best competitor | mid ratio | high ratio | vs heb mid / high | breakthrough |
|---|---|---|---|---|---|---|
| lred | w-body | heb | 1.75 | 2.01 | 1.75 / 2.01 | False |
| p14v6g | w-body | heb | 1.59 | 1.89 | 1.59 / 1.89 | False |
| p14v7 | w-body | heb | 1.14 | 1.92 | 1.14 / 1.92 | False |
| lred | w-lhome | sigma_shrink | 2.28 | 1.64 | 1.71 / 1.57 | False |
| p14v6g | w-lhome | sigma_shrink | 2.19 | 1.88 | 1.65 / 1.81 | False |
| p14v7 | w-lhome | sigma_shrink | 1.33 | 1.73 | 1.00 / 1.66 | False |

(The "best competitor" is chosen by mean log EtC+ over o50, o75 and o100 pooled, so it can differ from the arm with the highest G(o50): in w-body heb is chosen although pbml has the higher G(o50). [checked: analysis.py `best_competitor`, `breakthrough`].)

Pooled mid-plus-high ratio of p14v6g against the strong pooled-dynamics arms in w-body (PILOT_amend2-4.md): vs adapter 1.49 [1.12, 1.98], vs mbase 1.54 [1.28, 1.84], vs pbml 1.60 [1.32, 1.93], vs heb 1.78 [1.43, 2.23] (90% t intervals). In other words p14v6g needed about 1.5-1.8 times as many episodes as those arms at mid and high overlap.

### 3.4 First try

Source: PILOT_amend2-4.md §4. In w-body, the first-try score at o50 was 0.03 for fresh, lred, p14v6g/p/nh, heb and all p14v7 arms, and 0.52-0.67 for reptile, trunk, adapter, rma, rma_ft, mbase, pbml, sigma_shrink. So the value carriers' gain is largely a better starting point; the "ability-style" arms (first try = fresh) got much smaller G.

### 3.5 p14v6's own predictions (PROTOCOL.md §6, copied from p14-v6/CORE.md §7) against the pilot

| Prediction (conjecture by the architect) | Pilot |
|---|---|
| First try equals p14v6nh and fresh | Holds: 0.03 / 0.01 (o50 / E1_0) in w-body, identical [PILOT_amend2-4.md §4] |
| DF(o50) 0.75 [0.45, 1.10]; killed if lower bound ≤ 0 | 0.21 [0.07, 0.37] (w-body), 0.20 [0.09, 0.30] (w-lhome): below the predicted interval, but the lower bound is above 0, so this kill did not fire |
| G vs overlap 0 / 0.20 / 0.40 / 0.55 / 0.65; killed if G(0) ≠ 0 or ρ < 0.8 | G(o50) 0.09; ρ 0.31 / 0.18. **Fires** (ρ < 0.8). The E2_0 interval [0.00, 0.14] is not exactly 0 either (see §5) |
| Conflict lower bound < -0.05 is fatal to the core | The conflict cell alone is not printed for candidates; the worst cell over all cells had lower bound -0.39 (o75) in w-body, so "never negative" (P3) failed in w-body. Whether the conflict cell itself crossed -0.05 is not recorded in the pilot tables |
| Varied vs narrow passes at α = 0.01 with P ≈ 0.7 | Fails in w-body (p = 0.023), passes in w-lhome (p = 0.008) |
| Retention ratio 0.97; killed if lower bound < 0.8 | Passes in w-body (0.26 → 0.25) and w-lhome |
| Learning-speed share ≥ 0.8; killed if < 0.5 | P6 fails in w-body (LSG 0.11 vs G 0.09 does not pass the bound), passes in w-lhome (0.15) |
| p14v6p equals p14v6g on practised lags | Holds: rows near-identical |
| Breakthrough P ≈ 0.35 | No breakthrough; ratio vs best competitor 1.38-2.2 |

### 3.6 Data per practice task (amendment 1 sweep)

Source: research/tests-v6/p14d/pilot/EPS_SWEEP.txt (w-body, 8 seeds, defaults; practice episodes per task 5 / 20 / 80).

| arm | G@5 | G@20 | G@80 | DF@5 | DF@20 | DF@80 |
|---|---|---|---|---|---|---|
| gram | 0.37 | 0.37 | 0.37 | 1.00 | 1.00 | 1.00 |
| oraclef | 0.64 | 0.64 | 0.64 | 2.23 | 2.23 | 2.23 |
| mbase | 0.52 | 0.52 | 0.54 | 1.68 | 1.65 | 1.77 |
| adapter | 0.49 | 0.47 | 0.51 | 1.64 | 1.55 | 1.66 |
| pbml | 0.55 | 0.58 | 0.56 | 1.88 | 2.00 | 1.94 |
| sigma_shrink | 0.53 | 0.49 | 0.45 | 1.80 | 1.51 | 1.45 |
| lred | 0.00 | -0.00 | -0.00 | 0.00 | 0.00 | 0.00 |
| p14v6g | 0.06 | 0.09 | 0.13 | 0.14 | 0.21 | 0.29 |

The registered prediction (from Weinstein and Botvinick) was that the structure-discovering arms' DF would *fall* as data per task rose, with DF(80) ≤ 0.7 x DF(20) for p14v6g. It rose instead (0.14 → 0.21 → 0.29). **Prediction not supported.**

### 3.7 The w-active world and its failed validity check

Source: research/tests-v6/p14d/pilot/ACTIVE_VALIDITY.txt; PILOT_active.md §6; PROTOCOL.md §10f.

| arm, rule | mean log EtC+ over o50/o75/o100 | ratio vs w-body |
|---|---|---|
| fresh, w-body (fixed data) | 2.61 | 1.00 |
| fresh, w-active, fixed rule | 2.62 | 1.01 |
| fresh, w-active, declared default rule | 2.85 | 1.27 |
| fresh, w-active, privileged oracle rule v2 | 2.86 | 1.28 |
| mbase, w-body | 1.98 | 1.00 |
| mbase, w-active, default rule | 1.92 | 0.94 |
| mbase, w-active, oracle rule v2 | 2.14 | 1.17 |

For fresh, oracle minus fixed = +0.235 in log EtC+ (ratio 1.26, sd of per-seed difference 0.21); the registered threshold was ≤ -0.15. **Validity NOT MET.** Choosing commands by leverage did not help; it hurt. The first version of the oracle rule had a defect: its candidate commands were the same as the default rule's, so it picked exactly the same commands (found by comparing episodes); version 2 added privileged candidates along and across the true noise direction, and still did not help. Both registered predictions on the rules missed (default predicted ratio 0.8, found 1.27; oracle predicted 0.7, found 1.26).

By the registered rule, w-active rows carry no weight. For the record, the change in log EtC+ from fixed to active data (PILOT_active.md §6): adapter 0.81, rma_ft 0.87, mbase 0.94, sigma_shrink 0.94, oraclef 0.94, pbml 1.03, lred 1.08, gram 1.17, heb 1.18, p14v6g 1.21, p14v7 1.23, fresh 1.27. The prediction that the learning-speed arms would gain most from active choice was wrong: they lost the most. In w-active, G(o50) was adapter 0.62, mbase 0.57, heb 0.36, p14v7 0.37, p14v6g 0.08; no arm showed P14.

### 3.8 Compute

Source: estimate.md. Sanity 140 jobs (about 0.8 core-hours), first pilot 272 jobs (about 1.9 core-hours), amendment pilots about 5 core-hours more. The planned main run (6 worlds x 29 arms x 48 seeds = 8352 jobs) plus tuning was estimated at about 90 core-hours central, at most about 125. results/ contains only `pilot`, `sanity` and a 4-job tuning-machinery test for p14v6g (results/tune/p14v6g); there is no `results/main` and no tuning table [checked: directory listing].

---

## 4. What we inferred

**What the pilot says.**
1. **No arm showed the P14 pattern** in either pilot world: not the strong existing methods (adapter, PBML-style prior, sigma_shrink, mbase, RMA, Reptile), not the existing-methods stack heb, not our designs. Every arm failed P1 (gain proportional to shared ability). Every arm with a meaningful G also failed P6 in w-body.
2. **The value carriers win on the pooled metric**, mostly through a better first try (0.52-0.67 of expert at o50 vs 0.03 for fresh). That is exactly the kind of transfer Lucky said P14 is *not* about, and still they set the bar on episodes to criterion.
3. **p14v6 failed clearly.** Its gain was small (G(o50) 0.09-0.10, DF 0.20-0.21), far below its predicted DF of 0.75, and it needed about 1.4-2.2 times as many episodes as the best competitor. Its own registered kill (ρ < 0.8) fired. "Never negative" failed in w-body (worst lower bound -0.39). Its no-practice control `p14v6nh` was close to fresh (G(o50) 0.01), so whatever p14v6g gained did come from practice, but there was little of it.
4. **Our arms whose first try equals fresh (p14v7, lred, p14v6) never beat heb**, the registered existing-methods bar.
5. **Not a breakthrough by any rule**, and the sensitivity sweep did not flip any "shows P14" verdict (all "no" under every swept threshold, PILOT_amend2-4.md sweep tables).

**What it does NOT say.**
- It is not the pre-registered result. All arms ran **untuned defaults** on 8 pilot seeds and 2 of 6 worlds. Tuning could change rankings; for example l2init's -7.2 is very likely a default-setting artefact (PROTOCOL.md §10f says so: 20 truncated steps cannot correct the carried weights; its tuning space reaches 150 steps).
- It does not show that P14 is impossible, only that none of these learners, in this world, showed it.
- It does not measure motor ability acting through control (reflexes), because learning here is system identification under a shared controller.
- DF numbers above 1 do not mean an arm "knows the body better than the grammar", because `gram` is a weak reference (§10e).

---

## 5. Why we think it happened

- **Shared body dynamics dominate the transfer available in w-body.** [checked: PROTOCOL.md §10c amendment 4 facts; oraclef G(E1_0) 0.56] In w-body the body constants are identical across E1 tasks, so any arm that pools a dynamics model gets a big first-try advantage on every E1 task, whatever its declared overlap. That flattens G across overlap levels, which is why value carriers have ρ near 0 (adapter 0.02, mbase 0.06, pbml 0.10). [inferred]
- **Declared overlap is defined over abilities A1-A4 plus A5, but most of the attainable gain comes from A5, which is the same for every E1 cell.** So P1 asks for a gradient that the world mostly does not offer to value carriers. Even oraclef, which carries exactly the practised components, has Spearman 0.87 in w-body, just below the 0.9 validity threshold. [inferred from §3.1 numbers]
- **Why the first-try-equals-fresh arms gain less.** An arm that carries no values must earn its gain from the task's own first few episodes. With 20 episodes of data per practice task and 24 tasks, the shrinkage targets and noise laws that p14v6 extracted moved the per-task estimate only a little (G 0.09). [inferred] The data-per-task sweep shows its gain rising with more data per practice task (G 0.06 → 0.13), which suggests p14v6's discovery step was data-starved at 20 episodes per task rather than the opposite. [inferred from EPS_SWEEP.txt]
- **Why p14v6g is not exactly 0 on E2_0.** Its E2_0 interval is [0.00, 0.14], and the no-practice control p14v6nh has the same interval. Since nh has no practice at all, this gain is not transfer: p14v6's own per-task estimator (heteroscedastic weights, event fits) is simply a little better than fresh on some E2 tasks. [inferred: identical intervals for g and nh, PILOT_amend2-4.md §3; the architect predicted G_nh > 0 on precision/slip tasks, PROTOCOL.md §6] This makes P2 fail for a reason unrelated to P14.
- **Why heb's G is lower than mbase's.** heb is raced against fresh, and the race starts on fresh, so its first try is fresh's (0.03) and it gives up the first-try bonus; heb_norace has G 0.46 / 0.56. [checked: PROTOCOL.md §10f; first-try column]
- **Why P6 fails even for arms with fresh first tries** [inferred by the arch-v8 architect, research/architecture/arch-v8/NOTES.md line 46, from the LSG formula]: LSG counts only the time *after* an arm first reaches 0.3 x criterion. An arm whose first try equals fresh's but which reaches 0.3 x criterion sooner (heb, p14v7) gets G > 0 with LSG near 0 or negative: its gain is in the early episodes, which LSG books neither as first try nor as speed. If this reading is right, P6 as operationalised may under-credit exactly the "learns faster from a fresh start" behaviour Lucky describes. This was noticed after the pilot; no rule was changed.
- **Why w-active did not help.** Not tested. The builder's guess: large greedy commands push states outside the range the closed-loop test uses, and corner commands raise signal-dependent noise. [guess; PROTOCOL.md §10f calls it "inferred, not tested"]

---

## 6. Caveats and what could be wrong

- **Pilot only.** 8 seeds, untuned defaults, 2 of 6 worlds (w-fresh, w-offgram and w-dynonly have no pilot data at all). Power was planned for 48 seeds. Rankings among close arms (pbml vs mbase vs sigma_shrink vs heb) are not resolved by these data.
- **Default configurations were not equal in quality across arms.** Some arms' defaults were close to good (mbase), others clearly bad (l2init). The pilot cannot be read as a fair contest; the tuned main run was designed to be.
- **The "fresh" inside p14v7 used fresh's default configuration** in pilots (IMPLEMENTATION.md §9 point 8); the builder notes this makes pilot numbers slightly pessimistic for p14v7.
- **`gram` is a weak reference** (§10e). DF is inflated for pooled-dynamics arms. The proposed `gramP` was never built.
- **Validity was only partly met.** w-body's oraclef Spearman (0.87) sits just below its own V2 threshold for w-body (0.9) in the pilot; the sanity run (4 seeds) gave 0.93 (sanity/SANITY.md). V4 (w-offgram) was only weakly met. w-active failed validity.
- **The world is linear system identification with a shared MPC.** Reflex-like abilities acting through control, non-linear bodies, and real robot data are absent. The task-variation of Θ is low-dimensional by construction (§10d), which favours prior-based learners.
- **The metric P6 may be mis-specified** for fresh-start arms (§5). This cuts both ways: it may also have been too lenient for some value carriers.
- **Some interpretations in §5 are mine, not recorded results**: the A5-dominance explanation of flat ρ and the data-starvation reading for p14v6 are [inferred].
- **I could not check** the conflict-cell value alone for p14v6g (not printed), or the count of numerical fallbacks in p14v7 jobs (IMPLEMENTATION.md §9 point 10 says it was zero in the smoke job and "not checked for all pilot jobs").

---

## 7. What it changed in the project afterwards

- **The main run never happened.** Tuning and the 48-seed main run (about 90 core-hours, part of about 230 core-hours with T-CAUSE, about $10 on a rented 192-vCPU machine) were built, pre-registered and documented in README-RUN.md, then left waiting for a machine (memory note tests-v6-status.md; notes/all-tests-ledger-2026-10-28.md "Never tested"). So the project has no pre-registered T-P14D verdict.
- **The ledger's summary** (notes/all-tests-ledger-2026-10-28.md §8): "P14 unsolved by anyone. Our designs lost to generic adapters." That is a fair summary of the pilot, with the caveat that the adapter's gain is first-try driven.
- **p14v6 was dropped** as a P14 candidate; p14v7 was the next attempt and its result is file 09.
- **arch-v8** (2026-10-20, research/architecture/arch-v8/ARCH.md) adopted `heb`'s hierarchical empirical-Bayes statistics as its mechanism for body change (M3/M5), noting honestly that heb's covariance parts added little in the pilot (heb_diag -0.03, heb_noeb -0.04 vs heb) and that its gain is mostly the mean. For P14 it invented one mechanism, M10, a "form-evidence layer" (learn a prior over which *forms* recur, mixed with fresh by a fixed-share switch), with a registered kill test as a new T-P14D arm `formev`. That arm was never built or run.
- **The proposal `gramP`** (grammar plus the same practice history) remains unbuilt.
- **Lesson the project drew** (ledger, "one-line pattern"): the principle looked sound in design, then lost to a simpler or existing method once that method had the same data and knowledge. In this test, the existing methods themselves did not show P14 either.
