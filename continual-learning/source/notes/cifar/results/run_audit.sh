# audit_plan.md A1-A3. Resumable: skips any job whose final output exists.
cd /mnt/project-files/notes/cifar
O=results/ts5
for s in 0 1 2 3 4; do
  [ -f results/ts5_probe/enc_$s.pt ] || echo "CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03 MEM=2000 python3 ts5_probe_run.py $s > $O/erp_$s.log 2> $O/erp_$s.err"
done > results/jobs_audit
for s in 0 1 2 3 4; do
  [ -f $O/er_m3439_$s.npz ] || echo "CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03 MEM=2000 python3 cifar2.py run er_m3439 $s $O mem=3439 > $O/er_m3439_$s.log 2> $O/er_m3439_$s.err"
  [ -f results/audit/slp_exact_$s.npz ] || echo "RESV=20 LR=0.03 python3 sleep_exact.py $s results/audit > results/audit/slp_exact_$s.log 2> results/audit/slp_exact_$s.err"
done >> results/jobs_audit
wc -l results/jobs_audit
xargs -P 4 -I{} bash -c "{}" < results/jobs_audit; echo DONE > results/audit/DONE
