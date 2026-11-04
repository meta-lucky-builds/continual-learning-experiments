# 23. What is still open

1. **Is our learner the reason random features win?** [18](18-atlas-v1-cifar100.md) showed random > learned (35.1 vs 25.1) in our re-implementation, the reverse of ewt. v2b's calibration and gate answer this first.
2. **Does buffer training bias exist on CIFAR-100?** The matched-size test in v2b separates item count from provenance.
3. **How much does combining the random chart with the migrated learned readout give?** v2b's PoE and ε-mixture arms.
4. **Is X2's chained transport robust?** +5.6 in ewt, −2.1 in our runs. It needs a strong-learner replication.
5. **Where does the remaining headroom go?** On CIFAR-10, joint training's features reach 75.5 against ~59–61 for our best online system; the matched probes have not been run there. On CIFAR-100, the gap from oracle statistics (35.3 in ewt) to our best online system (32.1) remains.
6. **Why does removing first-order margin damage barely help?** Open since [03](03-m2c-diagnosis.md).
7. **A streaming rule for which versions or charts to keep.** "Spread" used stream length ([10](10-ts5-growth-cifar100.md)). ARCH-v2 suggests publishing before within-class R² of the previous chart drops too low; per-hop R² ≈ 0.95 vs 0.68 over 19 tasks supports small hops.
8. **Publication deadlock** (ARCH-v2's failure mode, where nothing can be published) has not been tested.
9. **Recurrence savings for dormant classes** under correct vs wrong traces has not been tested.
10. **Scale.** Nothing has been checked at ResNet multi-epoch scale or on a language model.
