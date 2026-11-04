# Top-level corpus (ewk.md + insights.md) compared with v2

Read in full on 2026-10-28: /mnt/project-files/ewk.md (9,772 lines) and /mnt/project-files/insights.md (2,696 lines). No folder named "cl" exists; these are the top-level files next to cross.md.

## What it is
- **ewk.md**: critical write-ups of 97 machine-learning continual-learning papers (P01–P97). Each one follows a fixed template: identity and coverage, aim, method, result tables, critical notes, connections, a one-breath takeaway, and often "Deepening" and audit addenda. Claims carry tags: [paper], [inferred], [critique], [observed], [known].
- **insights.md**: 537 numbered insights (I-001–I-537). Each is tagged and points back to papers and to the cross.md themes (T#) and debates (D#).
- **How it differs from v2**: this corpus is centred on ML papers and benchmarks. v2 leans on neuroscience: it has 92 sources and 60 background refs distilled into 71 principles with confidence labels, plus a map of your own ideas (pre-skill buffer, memory stamper, flywheel). The two overlap only a little: v2 S01 = P20, S06 = P90, B02 = P08 EWC, B03 = P04 GEM, B06 = P89 loss of plasticity.

## Main clusters
- **Self-teaching and grounding**: RLSD, SDFT, TTPO, BMG (D27).
- **Tiny updates in a shared low-dimensional space**: TinyLoRA, SVF, Share, SD-LoRA (D19). The LoRA down-projection acts as the interference knob (D23): CL-LoRA, SLAO, InfLoRA, FM-LoRA.
- **Merging as continual learning** (D24): BECAME, SLAO, P&M. Plain 1/t averaging gives most of the gain.
- **Frozen pre-trained features with statistics heads**: RanPAC, FeTT, SAFE, FoRo, plus analytic or recursive-least-squares theory (P96). P97 ranks closed-form, exemplar-free updates as the top future direction.
- **Hypernetworks**: context, description or prompt to LoRA (D2L, T2L, SHINE, DnD); HNET as an output regulariser; HyperNoise.
- **World models and imagination**: LucidWM (ensembles fail at doubt), Dreamer 4, RISE, Dream-RSI.
- **Optimiser geometry decides forgetting** (D26): NCL (P95). Same objective and same Fisher as KFAC, only the preconditioning differs, yet KFAC with λ=1 forgets catastrophically. Large λ in EWC works mainly by turning Adam into NCL-like preconditioning. Related: FOGO, refresh ≈ SAM, EFC.
- **Scenarios and theory**: P91 (Task-, Domain- and Class-IL; regularisation collapses in Class-IL; replay is near the top everywhere; ER beats A-GEM); P85 (CIL = within-task prediction × task-ID prediction, where task-ID ≈ OOD detection); P89 loss of plasticity and CBP.
- **Data, selection and active learning**: P92 uses an EVT outlier score in a VAE latent space to pick core sets and queries; it stays robust when 7.5% of data is corrupted, and dissimilar-first task order beats similar-first by up to ~10%. P94 Meta-CAL finds random selection the best "active" strategy. Uncertainty sampling repeatedly picks outliers (P92, P94).
- **Surveys**: P48–P54, P93 (few-shot CIL), P97 (systematic review of 189 papers; frames forgetting as loss of access).

## Findings that recur across papers
1. **Strong simple baselines win often** (D28): ER, random selection, plain averaging, separate networks in Task-IL.
2. **The evaluation setting decides the winner** (D8, T18). Mini-batch mixing of exemplars alone shifts results by >5% (P92). Benchmark tables copied from original papers mix different backbones (P93).
3. **Freezing a good base representation and classifying by prototypes or statistics** is a strong floor for pre-trained-model CIL and for few-shot CIL (T52).
4. **Forgetting is often loss of access rather than erasure** (P97, T45/D17). This is the main bridge to v2's "stored vs accessible" principle and to your "pre-skill buffer" (savings).
5. **The update path matters, not just the objective** (NCL, projection methods, step-size restraint).

## Where the two libraries could be joined
The common thread is v2's "stored vs accessible" and savings, which matches ewk's forgetting-as-access cluster. v2's "write control" principles match the optimiser-geometry and LoRA-subspace cluster here, and v2's replay principles match the ER and selection findings. A merged index would link S#/I-# to P#/I-### at these points.
