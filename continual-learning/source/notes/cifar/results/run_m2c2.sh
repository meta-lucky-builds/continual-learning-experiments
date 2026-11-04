cd /mnt/project-files/notes/cifar
for s in 0 1 2 3 4; do [ -f results/r2/m2c2_$s.npz ] && continue; echo "RESV=20 LR=0.03 python3 cifar2.py run m2c2 $s results/r2 m2c=2 > results/r2/m2c2_$s.log 2> results/r2/m2c2_$s.err"; done > results/jobs_m2c2
xargs -P 4 -I{} bash -c "{}" < results/jobs_m2c2; echo DONE
