cd /mnt/project-files/notes/cifar
export CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03 MEM=2000
for s in 0 1 2 3; do python3 ts5_probe.py $s > results/ts5_probe/probe_$s.log 2>&1 & done
until [ -f results/audit/DONE ]; do sleep 20; done
python3 ts5_probe.py 4 > results/ts5_probe/probe_4.log 2>&1
wait
python3 analyze_ts5.py > results/ts5/analysis_rerun.txt 2>&1
diff <(grep -v Compute results/ts5/analysis.txt) <(grep -v Compute results/ts5/analysis_rerun.txt) > results/audit/g2_diff.txt && echo "g2 PASS: analysis.txt reproduced exactly from regenerated erp" > results/audit/g2.txt || echo "g2 FAIL: see g2_diff.txt" > results/audit/g2.txt
python3 analyze_audit100.py > results/audit/analysis100.txt 2>&1
env -u CL_K -u CL_NTASK -u CIFAR_DIR -u MEM RESV=20 LR=0.03 python3 analyze_audit10.py > results/audit/analysis10.txt 2>&1
echo ALLDONE > results/audit/ALLDONE
