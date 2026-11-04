#!/bin/bash
# waits for E2 runs, reruns cifar2 self-tests, then the round-2 queue (EWC lambda tuning first, then the rest)
cd /mnt/project-files/notes/cifar
until grep -q DONE results/run_e2.log 2>/dev/null; do sleep 15; done
RESV=20 LR=0.03 NT=4 python3 cifar2.py selftest > results/selftest_c2.log 2>&1
grep -q "ALL PASS" results/selftest_c2.log || { echo "selftest failed"; exit 1; }
export NT=1
cat results/jobs_r2a results/jobs_r2b | xargs -P 4 -I{} bash -c "{}"
echo DONE
