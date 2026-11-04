# 16. Marathon W7: can interference be predicted before training?

**Date:** 2026-10-28. **Source files:** as test 05.

## Why we ran it
If we could predict which new class will absorb an old one, M2c could constrain only those rivals (cheaper and sharper).

## What we did
Predicted, for each old class, which of the two new classes would absorb its errors, using (a) centred feature-mean similarity and (b) raw-input similarity. Compared to chance (50%).

## What we got
- Feature similarity: 63–72% \* in most settings.
- Raw-input similarity: **as good or better**, 60–74% (74 vs 66 on MNIST).

## What we inferred
Interference is moderately predictable, and learned features add nothing over raw inputs. Similarity-chosen rivals are worth at most a modest gain.

## Why we think it occurred
Classes that look alike in the input end up close in the features, so the input already carries the similarity.

## How reliable this test is
- **Medium.** Simple, 5 seeds; a modest, unsurprising result.
