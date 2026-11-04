"""Check the repository without a GPU, data or torch.

Runs from anywhere: python scripts/check.py
  1. every relative markdown link resolves
  2. the Atlas self-tests pass (src/v1 and src/v2b; sk_check needs scikit-learn)
  3. experiment 03's lock-in counts and p-values recompute from its per-seed results
Exits non-zero if anything fails.
"""
import importlib.util
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")


def broken_links():
    bad, total = [], 0
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(folder, name)
            text = open(path, encoding="utf-8").read()
            text = re.sub(r"```.*?```", "", text, flags=re.S)
            text = re.sub(r"`[^`\n]*`", "x", text)
            for target in LINK.findall(text):
                if re.match(r"[a-z]+:|#", target):
                    continue
                total += 1
                if not os.path.exists(os.path.join(folder, target.split("#")[0])):
                    bad.append(f"{os.path.relpath(path, ROOT)} -> {target}")
    return bad, total


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")


def main():
    failed = []
    bad, total = broken_links()
    print(f"links: {total - len(bad)} of {total} resolve")
    for line in bad:
        print("  broken:", line)
    failed += bad

    scripts = {"v1": ["test_lib.py", "test_analysis.py", "audit_tests.py"],
               "v2b": ["test_lib.py", "test_lib_v2.py", "test_analysis.py", "audit_tests.py"]}
    has_sklearn = importlib.util.find_spec("sklearn") is not None
    for version, names in scripts.items():
        src = os.path.join(ROOT, "continual-learning", "atlas", "src", version)
        for name in names + (["sk_check.py"] if has_sklearn else []):
            result = run([sys.executable, name], src)
            ok = result.returncode == 0
            print(f"atlas {version}/{name}: {'pass' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"{version}/{name}")
                print(result.stdout[-1000:], result.stderr[-1000:])
    if not has_sklearn:
        print("atlas sk_check.py: skipped (scikit-learn not installed)")

    exp = os.path.join(ROOT, "experiments", "03-idea-b-write-credit")
    result = run([sys.executable, "analyze.py"], exp)
    expected = ("10/10 vs 1/10  p = 5.95e-05", "10/10 vs 9/10  p = 5.00e-01", "9/10 vs 1/10  p = 5.47e-04")
    ok = result.returncode == 0 and all(line in result.stdout for line in expected)
    print(f"experiment 03 statistics recomputed: {'pass' if ok else 'FAIL'}")
    if not ok:
        failed.append("experiment 03")
        print(result.stdout[-1000:], result.stderr[-1000:])

    print("OK" if not failed else f"FAILED: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
