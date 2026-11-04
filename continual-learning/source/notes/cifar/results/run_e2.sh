#!/bin/bash
cd /mnt/project-files/notes/cifar
RESV=20 LR=0.03 NT=4 python3 cifar.py selftest > results/selftest_e2.log 2>&1
grep -q "ALL PASS" results/selftest_e2.log || { echo "selftest failed"; exit 1; }
export NT=1
xargs -P 4 -I{} bash -c "{}" < results/jobs_e2
echo DONE
