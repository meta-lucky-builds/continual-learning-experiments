# Hostile read 11

- **Status:** run and recorded. No test here ran on a robot.
- **Source:** audit/round10_work/hostile/r11_a.md and r11_b.md

## Why we ran it

To find defects by attacking the spec between fixes.

## What we did

Two readers split the spec, on hash c430d540. Reader type: continued, change read.

## What we got

Critical/High/Medium/Low: 0/1/15/22.

## What we inferred

Findings went into the register and were fixed or classified.

## Why we think this happened

Fixes kept adding new interactions, so full reads stayed at 41-59 findings through round 10. Falls after that partly reflect change-only reading.

## Limits

Same kind of model as the author. Rounds 11 to 15 read changes only, so their counts are not comparable with full reads.
