#!/bin/bash
cd /mnt/project-files/notes/cifar
until grep -q "SELFTEST" results/selftest_v2.log 2>/dev/null; do sleep 10; done
grep -q "ALL PASS" results/selftest_v2.log || { echo "selftest failed"; exit 1; }
export NT=1
xargs -P 4 -I{} bash -c "{}" < results/jobs_e1
echo DONE
