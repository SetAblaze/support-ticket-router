"""Clone the current commit into an empty directory and run release checks.

This is the reproducible check used by the notebook archive stage.  It does not
copy the working tree, so uncommitted edits are deliberately absent.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import support_router as sc


def run(command, cwd):
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=False).returncode


def main(argv=None):
    root = Path(sc.project_root())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(root))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)
    tmp = tempfile.mkdtemp(prefix="support-router-clean-")
    target = Path(tmp) / "repo"
    if run(["git", "clone", "--depth", "1", args.repo, str(target)], cwd=tmp) != 0:
        return 2
    checks = [["python", "scripts/prepare_data.py", "--rebuild"],
              ["python", "scripts/prepare_data.py", "--verify"],
              ["python", "scripts/release_manifest.py", "--verify"]]
    if not args.skip_tests:
        checks.append(["python", "-m", "pytest", "-q"])
    failed = []
    for command in checks:
        if run(command, cwd=target) != 0:
            failed.append(" ".join(command))
    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        print("CLEAN_CHECKOUT_FAILED:", "; ".join(failed))
        return 2
    print("CLEAN_CHECKOUT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
