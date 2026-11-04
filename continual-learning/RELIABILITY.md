# How reliable are the tests themselves?

Copied from section 18 of `source/notes/ewt.md` (2026-10-30), with test numbers added. "Results reliable" and "tests reliable" are different questions, and this file answers the second.


Lucky asked for this separately from "are the results right". There are two layers:
- **Computational reliability:** given the design, did the code compute what it claims? This is strong.
- **Design reliability:** does the test actually measure the question, with fair comparisons? This is weaker, and it is where our errors have been.

## 1. What the evidence says about computational reliability (strong)

- **Exact reproduction gates.** Every recent experiment reloads saved runs and must reproduce earlier saved numbers exactly before reporting: T-S5 analysis (g2), matched probes (g1, g4), buffer reconstruction (gR), stream statistics (gS), the oracle arm (gO), and T-S5 headline numbers (gB). All passed. When result files vanished from disk twice, regenerated runs reproduced the old numbers exactly.
- **Planted self-tests with known answers** before each real run (C1–C13, K1–K10, sleep Z-tests, X1 P1–P5, X2 T1–T4, X3 S1–S3).
- **Independent reviewers recomputed numbers by hand** from the saved outputs for X2 and X3 and matched them.
- **Literature numbers** used in the feeds were re-fetched by an independent agent: 10/10 (geometry) and 11/11 (memory-and-drift) matched.
- **Track record:** BUGS.md (section 12 of ewt.md) lists 26 bugs, mistakes and process errors. Most were caught *before* they produced a reported number. The exceptions, where something already reported had to be withdrawn or superseded, are rows 2 (an in-sample claim), 6 (marathon T1 runs), 16 (an M2c status message), the T-S5 readout design (test 34) and rows 23–24 (X3 readings).

**My estimate:** a reported number in this file is very likely the number its code computes (I would put it above 95%). I have not found a case where a reported number was arithmetically wrong; the withdrawals above were wrong *readings* or wrong *designs*, not wrong arithmetic. This is a judgement, not a measured rate.

## 2. What the evidence says about design reliability (moderate; this is the real weakness)

**Track record.** Of the 7 experiments since the audit (A1, A2, A3, X1, X2, X3, and the T-S5 reading they corrected), design-level problems found *after* the run:

| Test | Design problem found after the run | Found by | Changed a conclusion? |
|---|---|---|---|
| T-S5 version curve (test 34) | Readouts not matched: late versions had about 20 items per class | Lucky's audit | **Yes:** "features decay" withdrawn |
| X2 (test 39) | Byte figures wrong (d = 512, not 256); MIX3 computed for the wrong arm; plan timestamps were my guesses | independent review | No (headline stands; byte ratio 6× became 4–8×) |
| X3 (test 40) | Covariance mismatch between arms; "OLD is the ceiling" false; bytes compared against the wrong encoder size | independent review | **Yes:** R3 reversed for the oldest tasks; cost claim reversed |
| X1 (test 38) | none found, but **X1 was not independently reviewed** | – | unknown |
| A1–A3 (tests 35–37) | none found, but **not independently reviewed as code** (they answered an external audit) | – | unknown |

So in the two experiments that got a full independent review, both had a design-level flaw, and one changed a conclusion. Self-tests did not catch any of these, because self-tests check the code against my design, not the design against the question.

**My estimate:** for a conclusion that has passed gates but *not* independent review, I would expect roughly a 1-in-3 chance that a design flaw changes part of the reading (not usually the headline number). This figure rests on very few cases (2 of 7 conclusions changed; 2 of 2 reviewed experiments had a flaw), so treat it as a rough judgement, not a rate. After independent review, the remaining risk is lower but not zero: reviews have so far found something every time.

## 3. Specific weaknesses of the test setup itself

1. **All recent tests reuse the same 5 training runs.** A1, X1, X2 and X3 replay the T-S5 ER runs. They are not independent replications. A flaw in those runs would propagate everywhere: the gates check consistency, not correctness of the training. The training code (cifar.py) has self-tests C1–C13 and an earlier review.
2. **No published number has been reproduced on this harness.** Lucky's test-reliability rule asks for that first. Our absolute accuracies are internally consistent but not calibrated against the literature. Relative comparisons are what we can trust.
3. **One learner, one scale.** Online ER, small GroupNorm CNN, CIFAR-10/100, 5 seeds, CPU. Every claim is conditional on that.
4. **The significance rule is weak and is applied many times.** |mean| > 2·SE plus 5/5 signs has sign-test p = 1/32 per comparison. Across dozens of comparisons, a few starred results are expected by chance. Small starred effects (for example X2 Q4, +0.86 with one seed at +0.04) should be treated as provisional.
5. **Pre-declaration is self-reported.** File modification times show order but not that a plan was unchanged; X2's plan times were my guesses, and X3's plan was saved only 19 s before the code. Test thresholds were amended twice in X2 before real runs, and those amendments were set by my expectation, not derived.
6. **Some tests are post-hoc or offline.** X1 tests post-hoc heads, not a trained cosine head. X2 and X3 replay transport offline; they have not been run live inside a learner.
7. **Hyperparameters.** lr was chosen on the seed-100 test set and not retuned for CIFAR-100; EWC λ came from a 3-value grid; ridge λ in X2/X3 was fixed in advance (conservative, but untuned).
8. **Measures differ between tests.** X3 uses a task-aware 5-way measure; X1, X2 and T-S5 use 100-class accuracy. They should not be compared directly.
9. **Infrastructure.** Result files vanished from the network mount twice (cause unknown). Gates were rebuilt on small JSON outputs that survived; no result was lost, but it is a fragility.

## 4. Bottom line

- **The numbers are reliable** for the code that produced them: gates reproduce exactly, self-tests pass, reviewers recompute the same values.
- **The tests are moderately reliable as tests of the questions they claim to answer.** Every independently reviewed experiment has had a design flaw; two of the last seven conclusions changed after outside checking. Treat any conclusion without an independent review as provisional.
- **Fix going forward:** independent review *before* reading results (of the design, not just the code); matched comparisons by construction (same covariance, same data, same readout); reproduce one published number on this harness; and replicate the key findings (X2, X3) on fresh training runs, not only the saved ones.

## Note on the significance rule's wording in older files

Some source write-ups (marathon/results.md, cifar/results_r2.md, results_sleep.md) describe the star rule as "the paired mean ± SE excludes 0 and all 5 seeds share the sign". The rule **as coded** (analyze_cifar.paired and later scripts) is stricter: |mean| > 2·SE **and** 5/5 signs. Lucky's audit found this mismatch; ewt.md was corrected. The 1·SE wording would be implied by 5/5 alone, so the coded rule is the one that applies to every star in this repository.
