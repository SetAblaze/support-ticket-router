"""Rebuild or verify the processed train/calibration/val/test split.

The split is produced by the same functions used for training:
``clean_frame`` -> ``lexical_groups`` -> ``make_splits``.  This script never
fits a classifier.  It refuses to overwrite the reference manifest when the
rebuilt hashes differ, because that would hide a reproducibility regression.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import support_router as sc
from support_router.train import DEFAULT_RAW, build_splits

SPLIT_NAMES = ("train", "calibration", "val", "test")
EXTRA_FILES = ("short_examples_retained.csv",)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _differences(reference: dict, candidate: dict) -> list[str]:
    problems = []
    if reference.get("raw_sha256") != candidate.get("raw_sha256"):
        problems.append("raw_sha256 differs")
    if reference.get("seed") != candidate.get("seed"):
        problems.append("SEED differs")
    if reference.get("roles") != candidate.get("roles"):
        problems.append("split roles differ")
    ref_splits = reference.get("splits", {})
    new_splits = candidate.get("splits", {})
    for name in SPLIT_NAMES:
        ref, new = ref_splits.get(name, {}), new_splits.get(name, {})
        if ref.get("rows") != new.get("rows"):
            problems.append(f"{name}: row count {ref.get('rows')} != {new.get('rows')}")
        if ref.get("sha256") != new.get("sha256"):
            problems.append(f"{name}: sha256 differs")
        if ref.get("class_counts") != new.get("class_counts"):
            problems.append(f"{name}: class distribution differs")
    ref_grouping = reference.get("grouping", {})
    new_grouping = candidate.get("grouping", {})
    for key in ("n_groups", "largest_group", "largest_fraction", "threshold"):
        if ref_grouping.get(key) != new_grouping.get(key):
            problems.append(f"grouping.{key}: {ref_grouping.get(key)} != {new_grouping.get(key)}")
    return problems


def _build(raw: Path, directory: Path):
    splits, manifest = build_splits(raw, directory)
    return splits, manifest


def _copy_processed(source: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    for name in (*SPLIT_NAMES, *EXTRA_FILES):
        src = source / f"{name}.csv"
        if src.exists():
            shutil.copyfile(src, target / f"{name}.csv")
    shutil.copyfile(source / "split_manifest.json", target / "split_manifest.json")


def main(argv=None):
    root = sc.project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=root / "data/processed")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true", help="compare rebuilt split with the reference manifest")
    mode.add_argument("--rebuild", action="store_true", help="write the split only if it matches the reference")
    parser.add_argument("--allow-new-manifest", action="store_true",
                        help="with --rebuild, create a manifest when none exists")
    args = parser.parse_args(argv)

    raw = Path(args.raw)
    out = Path(args.out)
    if not raw.exists():
        sys.exit(f"Raw data not found: {raw}")
    reference_path = out / "split_manifest.json"
    reference = _read_json(reference_path) if reference_path.exists() else None
    if reference is None and not (args.rebuild and args.allow_new_manifest):
        sys.exit(f"Reference manifest not found: {reference_path}; use --allow-new-manifest explicitly")

    with tempfile.TemporaryDirectory(prefix="support-router-data-") as tmp:
        generated_dir = Path(tmp)
        _, candidate = _build(raw, generated_dir)
        if reference is not None:
            problems = _differences(reference, candidate)
            if problems:
                print("SPLIT_MISMATCH")
                for problem in problems:
                    print(" -", problem)
                print("Reference manifest was not modified.")
                return 2
            print("SPLIT_MATCH")
            print("raw_sha256:", candidate["raw_sha256"])
            for name in SPLIT_NAMES:
                info = candidate["splits"][name]
                print(f"{name}: rows={info['rows']} sha256={info['sha256']}")
        if args.rebuild:
            _copy_processed(generated_dir, out)
            print(f"WROTE {out}")
        else:
            print("VERIFIED (no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
