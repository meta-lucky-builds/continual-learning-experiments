"""Tally lock-in per arm and apply the pre-registered decision rules (PREREGISTRATION.md)."""
import glob
import json
from math import comb

R = "/mnt/project-files/idea-b-write-credit/results"
rows = [json.load(open(f)) for f in sorted(glob.glob(R + "/*.json"))]
arms = {}
for r in rows:
    arms.setdefault(r["arm"], []).append(r)


def fisher_one_sided(a, n1, b, n2):
    """P(X >= a) for the treatment count under the hypergeometric null (row 1 = treatment)."""
    k, N = a + b, n1 + n2
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / comb(N, k)


summ = {}
for arm in sorted(arms):
    rs = sorted(arms[arm], key=lambda r: r["seed"])
    locked = [r for r in rs if r["eval_accuracy"] > 0.8]
    genuine = [r for r in locked if r["rebind_accuracy"] < 0.1]
    summ[arm] = (len(genuine), len(rs))
    print(f"{arm:16s} n={len(rs):2d} lock-in={len(locked)}/{len(rs)} genuine(rebind<0.1)={len(genuine)}")
    print("   acc   :", [round(r["eval_accuracy"], 3) for r in rs])
    print("   rebind:", [round(r["rebind_accuracy"], 3) for r in rs])
    if rs[0].get("probe_end_accuracy"):
        print("   probe :", [[round(x, 3) for x in r["probe_end_accuracy"]] for r in rs])

print("\nValidity:")
if "A0_naive" in arms:
    print(" V1 A0 all < 0.05:", all(r["eval_accuracy"] < 0.05 for r in arms["A0_naive"]))
if "A2_dense_curric" in summ:
    print(" V2 A2 lock-in >= 3/10:", summ["A2_dense_curric"][0] >= 3, summ["A2_dense_curric"])
if "A1_dense" in summ:
    print(" V3 A1 lock-in <= 2/10:", summ["A1_dense"][0] <= 2, summ["A1_dense"])
for t, c in (("A3_dense_local", "A1_dense"), ("A3_dense_local", "A4_dense_end"), ("A2_dense_curric", "A1_dense"),
             ("A4_dense_end", "A1_dense")):
    if t in summ and c in summ:
        a, n1 = summ[t]; b, n2 = summ[c]
        print(f" Fisher one-sided {t} > {c}: {a}/{n1} vs {b}/{n2}  p = {fisher_one_sided(a, n1, b, n2):.2e}")
