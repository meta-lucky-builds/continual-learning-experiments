# Response to Lucky's ewt.md audit: verdicts and test results (2026-10-29)

**Sources:**
- The audit: continual_learner_ewt_audit/EVIDENCE_AUDIT_AND_SYNTHESIS.md, check_implications.py and implication_checks.json, all on Lucky's PC.
- Our tests: audit_plan.md, pre-declared at 13:45 UTC, before any run.
- Code: ts5_probe_run.py, ts5_probe.py (planted self-test probe_selftest.py, P1–P5 pass), sleep_exact.py, analyze_audit100.py, analyze_audit10.py.
- Outputs: results/audit/ and results/ts5_probe/.

**Gates, all passed:**
- **g1.** The regenerated learner equals the T-S5 sleep arm bit for bit through task 2, and shares the RC stream.
- **g2.** analysis.txt is reproduced exactly from the regenerated files.
- **g3.** The saved version statistics reproduce the online version log-probabilities exactly.
- **g4.** The matched probe reproduces RC's own online accuracy, to within 1e-7 on every seed.
- **Sleep control.** The exact copy is within 3.8e-6 in logits of the teacher, and bit-identical to ER through task 2.

I re-derived the audit's own checks (the 2·SE algebra, the 86% counterexample, the 3-class averaging example, the 1[x ∈ B] feature, and the 3,432 interpolation) by hand. They are correct.

## Verdict on each point

| # | Audit claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Retained knowledge ≠ usable predictions. The target is "can build a good predictor from what's retained". | **Agree**, and A1 strengthens it | The features improved all stream long; the readout statistics were the failure |
| 2 | The CIFAR-100 version curve doesn't isolate feature forgetting (coverage confound) | **Correct and decisive.** My claim is withdrawn | Code: versions get old classes from the buffer only. Matched probes: v18 − v8 = **+3.0 \***, v18 − RC = +6.7 \*, task-0 classes +0.9 (not significant). The online decline (−4.6) is entirely readout. |
| 3 | Uniform sampling ≠ unbiased statistics for adapted features. "Never touched training" is too strong. Accumulate-at-arrival is not enough under drift. | **Agree on all three** | On CIFAR-100, buffer-derived LDA ≈ 20 fresh items (24.9 vs 24.7), so there the gap is sample count. The MNIST bias (R3) stands for MNIST. |
| 4 | The mixture lesson is objective alignment; 69.1% is not a ceiling for averaging; 200 is not a threshold | **Agree** | – |
| 5 | Bias-correction wording: not "no formula", not a sign change, not a share of forgetting; 10.9% alone is not proof | **Agree**, wording fixed | – |
| 6 | 86% damage removed with no accuracy change doesn't show margin damage is irrelevant | **Agree**, claim withdrawn | The counterexample is exact |
| 7 | The M2c head isn't the spec's (10·cos, logits ≤ 10). "Constraints force scale" is unproved. | **Correct.** Every head we used, toy and CIFAR, is nn.Linear. | Conclusion narrowed to "this implementation failed". The audit understates one point: the toy M2c results also used a linear head. |
| 8 | Run the exact-transfer Sleep control | **Done** | Exact copy 52.2 vs ER 52.4 (−0.2, not significant); distilled −8.2 \* against the exact copy. The whole loss is approximate transfer. |
| 9 | Compare BANK_4spread against its own matched buffer (≈ 3,432) | **Done** at M = 3,439, from the exact byte count | 31.2 vs 30.0: **+1.3 ± 0.3 \***; against ER+bc at the same memory, +7.2 \*. Caveat: spread knows the horizon. |
| 10 | The star rule's SE condition is redundant; non-significant ≠ tie; "established" too strong | **Half right.** The rule as coded is 2·SE, which is not redundant (the audit's example fails it). But ewt.md *described* it as 1·SE, and that description was redundant. The rest is correct. | Fixed in ewt.md |

Also accepted, from the audit's §11:
- Toy "start-insert" is non-causal. diag.py inserts a task's memory items at task start, so ER-ACE 57.8 is a diagnostic only.
- Probe power numbers are conditional on their assumptions.
- Fresh-vs-stale Fisher conflates freshness with dose.

## What changes in our understanding

1. **The plastic network kept improving its features over 20 tasks.** That holds on the classes it saw first, too. In this setting, "catastrophic forgetting" of features did not happen. What failed was the readout's statistics.
2. **The real bottleneck is statistics stranded in old coordinates.** Old encoders have full history but older features. New encoders have better features but no history. The final learner's features with full-history statistics give 35.3. Our best online system gives 31.2, and ER+bc gives 22.7.
3. **Concrete next target:** carry class statistics forward into each new encoder (statistic transport), validated on held-out data, and compare against the spread bank at equal bytes.
   - Prior work exists: semantic drift compensation, and the feature-translation methods.
   - Our MNIST shared-affine transport failed.
   - So the open question is whether a class-aware, validated transport works on real images.
   This matches Lucky's goal: improve continually, keep old knowledge recoverable.
4. **Consolidation by distillation loses function only through imperfect transfer.** Any future consolidation must be measured against an exact-copy control.

## Not run (yet)

- **A causal version-retention rule.** That means choosing versions by marginal held-out benefit, without knowing the horizon.
- **Matched probes on CIFAR-10**, to check whether the 75.5 joint gap there is features or statistics.
- **Margin-crossing diagnostics for M2c.** These are low value while M2c is out.
