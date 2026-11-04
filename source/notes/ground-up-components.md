# Ground-up components: what we would design ourselves, and what the sources point to

2026-10-28. This document is not a design. It is a map: for each building block it says what the default does wrong, which direction the library points, what we would have to make ourselves, and how sure we are.

P-numbers refer to:
- the root library (/mnt/project-files/ewk.md);
- feed/ewk.md;
- feed/full/ewk.md.

The full reasoning is in feed/full/DEEP-PATTERNS.md.

**Confidence labels:**
- **Strong:** several papers agree, some at scale.
- **Moderate:** consistent, but small scale.
- **Guess:** my inference; no paper tests it.

---

## 1. Loss function

**The default fails because:**
- Cross-entropy on new data only drives old outputs toward zero, even when the old knowledge is correct (P129, P120).
- Under softmax, the importance signal dies just as a task is learned (P154).

**Direction from the sources:**
- The past must enter the loss as **data**, not only as a model-side term (P129). Distillation needs *old inputs* (P209: 0.57 vs 0.90).
- MSE or margin-type losses keep curvature information alive (P154). Normalising logit magnitude stops scale drift (P143, P120).
- Using stored items as *things to reject* (out-of-distribution negatives) beat using them as extra training labels at half the memory (P196).
- Reverse-KL / on-policy objectives forget less than forward KL (P129).

**What we would design:** a composite loss made of four terms:
- a new-task term;
- a term on real stored old items;
- a function-matching term evaluated on old inputs;
- a reject term (which context does this belong to?).

The output scale would be normalised so old and new classes compete fairly.

**Confidence:** moderate. The individual pieces are supported; the combination is a guess.

## 2. Classifier / readout

**The default fails because:** the head is where access is lost first. The same weights score 20.8 with the trained head and 64.8 with a re-fitted probe (P177; also P136, P143, P146, P178).

**Direction from the sources:**
- Rebuild the head instead of training it continually:
  - from class statistics (P143/P158);
  - from a few stored exemplars as class means (P177).
- Split prediction into two steps, as in P196:
  1. **which context** (a detection problem);
  2. **which class within it**.

  The routing step is where accuracy is lost, and the loss grows with the number of tasks.
- Intermediate layers hold readable old knowledge (P210).
- Stored statistics go stale as the backbone moves (P143, P174), so they must be refreshed from stored real items.

**What we would design:** a readout that is *recomputed*, not learned by gradient. It would be two-level (route, then classify), have optional readouts at more than one depth, and use statistics that are refreshed on a schedule.

**Confidence:** strong that this is where the gains are; the exact form is ours to design.

## 3. Optimiser

**The default fails because:**
- A single global learning rate trades plasticity against stability one for one (P158, P189).
- Adam's state is reset between tasks, which throws away a trace (P182).
- Annealing hurts under drift (P148).
- Large batches hurt online (P175).
- Plain SGD is itself unstable even with unlimited replay (P180: +7 points from preconditioning).

**Direction from the sources:**
- A per-layer step size: a slow backbone and a fast head (P158). The learning rate is the first brake (strong).
- Carry Adam's second moment across tasks; reset the first (P182; the split is a guess).
- Precondition each step so it changes outputs on stored items slowly (P180 LPR).
- Separate the size of a step from its direction (P195: an orthonormal update times one scale).
- Two timescales: a fast iterate plus a slow average. The gap between them is a free learning-rate controller (P148, P138).
- Flat-minimum seeking (SAM) is a cheap brake (P205).
- Per-parameter plasticity state, where uncertainty sets the step, needs a floor or release (P149, P163, P194).

**What we would design:** a continual optimiser in which every parameter keeps four things:
- how settled it is;
- how much it is used;
- how recently it was used;
- a release rate.

Step size would follow from that state. It would have a fast and a slow copy, and the step would be preconditioned against stored items.

**Confidence:** strong for the per-layer learning rate and two timescales; moderate for preconditioning; guess for the full per-parameter state.

## 4. Weights / parametrisation

**The default fails because:** one number per weight cannot be both fast and long-lasting (P189).

**Direction from the sources:**
- Hidden slow variables behind each weight: EMA copies (P138, P168) and cascade synapses (P176, P189). These hold an old policy after the visible weight has moved and can flow it back.
- Separate magnitude from direction (P195).
- Low-rank per-context adapters: rank works as a write dial (P123).

**What we would design:** a weight made of a fast part, one or more slow parts, and rules for consolidation (fast to slow) and reactivation (slow to fast). This is the engine of your pre-skill buffer.

**Confidence:** moderate. The theory is solid (P189); in deep nets it has only been shown at small scale, and P176 failed on Atari.

## 5. Architecture

**The default fails because:**
- Dense, shared layers are updated for every example.
- Bottleneck layers forget where a linear model provably cannot (P208).
- Learned routers drift (P156).

**Direction from the sources:**
- **Wide over deep**, especially the first layer (P208).
- **Sparse, gated activity**, where only units that fire get written (P167, P171, P188). It only works if applied network-wide.
- **Modules plus a stable router.**
  - Freeze the router once it works, or base it on a frozen query (P156, P203).
  - Spare dormant units serve as new capacity (P152, DER in P197).
- **Plasticity concentrated at the top layers**, since low layers stay stable anyway (P197: −65% compute).
- **Attention-based models** kept plasticity with replay where MLPs and RNNs did not (P186).
- **A strong pre-trained starting point** cuts forgetting 2–3× (P205, P124).

**What we would design:**
- a wide, sparse, gated backbone;
- context modules that can be added;
- a router that is stable but can be extended;
- plasticity budgeted by depth.

**Confidence:** moderate. Each piece is supported; nobody has shown the whole at scale.

## 6. Memory system

**The default fails because:**
- Clever replay selection barely beats random (P178).
- Generated replay drifts (P173, P135).
- Compute, not storage, is the limit (P148).

**Direction from the sources:**
- A fast episodic store plus a slow long-term store.
- **Fill the buffer by frequency and coverage, not by error** (P199, P200, P202, P146, P151, P159).
- Cap how often each item is reused (P133).
- Store class covariances, not only means (P174, P146).
- Any generated replay must be grounded:
  - P185: the generator should be the stored memory itself;
  - P211: track how many regeneration cycles each item has gone through.
- Keep the mixing ratio of old and new as a tuned dose (P148).

**What we would design:** a memory manager that decides what to keep and what to retire, and keeps stored items covering the directions old tasks use.

**Confidence:** strong for random-plus-dose as the baseline; moderate for coverage-based rules.

## 7. Algorithms / procedures (the training loop)

**What has to exist, with its source direction:**

1. **Write gate.** Learn into new capacity when the input falls outside what is already covered (residual novelty: P161, P165, P164).
2. **Context recognition.** "Have I seen this before?" Recognition is cheaper than relearning (P128). Typicality tests work for well-separated contexts (P155); near overlap nothing works.
3. **Consolidation phase** (offline, sleep-like). Replay real stored items mixed with recent ones, and write only changes that leave old outputs alone or improve them (P151 null space; P169/P173 offline recovery).
4. **Conflict split.** When old and new disagree on the same input, store separate versions keyed by context (P153: conflict, not overlap, causes forgetting).
5. **Compression.** Merge shared structure to free capacity (P164 refinement; P159: one item per still-undecided class).
6. **Release / forgetting schedule.** Locks loosen with disuse. The schedule depends on the stream type:
   - tightening is optimal for a recurring task pool (P204);
   - releasing is needed for an open-ended stream (P149/P150).
7. **Commit-only-determined.** Keep alternatives when the data leave the answer open (P165).

**Confidence:** guesses on how to assemble these. Each step has some source support.

## 8. Self-monitor

**The default fails because:** a model's own confidence cannot detect its own forgetting. Forgotten items look like never-seen ones (P160), and the model's confidence scores 0.618 against 0.966 for an external detector (P125).

**Direction from the sources:**
- A small set of real test items per skill, re-checked periodically.
- Score the monitor on three groups: In, Forgotten, Out (P160).
- Use performance gaps as signals (P148, P175), and drift estimates where the old encoder is kept (P202).
- Health proxies such as weight norm or feature rank are blind (P168).

**What we would design:** a monitor that is anchored to things the learner did not produce, has its own protection against forgetting, and triggers replay when a skill drops.

**Confidence:** strong that an anchor is required; the design is ours.

## 9. Evaluation harness (needed before any of the above)

Every claim should be measured with:
- **two readouts:** the trained head, and a re-fitted probe per block;
- **a relearn-from-scratch control** for savings, plus a wrong-item null (P128);
- **an LR-swept fine-tuning baseline** (P158) and **a frozen-feature baseline** (P197);
- **streams that are ordered or drifting**, not only random (P147), and **longer than 100 tasks** for locks;
- **matched compute, memory batch, capacity and hidden statistics** (P178, P138);
- **3–5 seeds** at minimum.

## What we would *adopt* rather than design

- A pre-trained backbone.
- Tuned random replay (ER) as the baseline to beat.
- Standard data pipelines and benchmarks: split CIFAR-100, ImageNet-R, CLOC-style streams.
- Existing CL code libraries for the baselines.

---

**Bottom line.** The sources give a clear direction for every component:
- what to stop doing (cross-entropy on new data only, one global learning rate, resetting optimiser state, trained heads, learned routers that drift, self-confidence as a monitor);
- what to try (data-side past terms, rebuilt readouts, two-timescale weights, a per-parameter plastic state with release, sparse gated wide modules, coverage-based memory, an anchored monitor).

What the sources **cannot** give is how these pieces behave together at scale. That is the part we would actually have to discover.
