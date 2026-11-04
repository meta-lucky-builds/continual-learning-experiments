# 18. Marathon: centring the head

**Date:** 2026-10-28. **Source files:** as test 05.

## Why we ran it
If the bias is a common-mode offset (test 05), centring the head's logits might remove it during training, without a fitted correction.

## What we did
Trained with a centred head and decomposed the remaining error into common and item-specific parts.

## What we got
The bias moved from the common term into the item-specific term. Accuracy did not improve.

## What we inferred
Failed. Centring hides the offset in another form instead of removing it.

## Why we think it occurred
The training-prior pressure (16:1 per step) still acts; removing one degree of freedom just lets the same pressure shape the class-specific weights instead.

## How reliable this test is
- **Medium.** One configuration; a clean negative.
