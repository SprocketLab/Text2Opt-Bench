#!/usr/bin/env python3
"""
Download supplementary Text2Opt-Bench assets from the Hugging Face Hub.

The main benchmark (Template/, Unstructured/) is bundled with this repo.
This script fetches the three supplementary assets used by the auxiliary
experiments in the paper:

    https://huggingface.co/datasets/ZhiqiGao/Text2Opt-Bench

  Template_train/   — binding-specialist SFT training corpus (~150 MB)
  Template_large/   — large-tier binding stress test (~240 MB)
  ruler/samples/    — pre-generated RULER long-context tasks (~220 MB)

The bundled Template/ and Unstructured/ sets are also mirrored on the Hub;
pass --only template / --only unstructured if you ever need to re-fetch
them (e.g., after deleting the bundled copy).

Usage:
    python scripts/download_data.py                    # default: train + large + ruler
    python scripts/download_data.py --only train       # one asset
    python scripts/download_data.py --only template    # re-fetch the bundled eval set
    python scripts/download_data.py --all              # everything

Requires: pip install huggingface_hub
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("Install huggingface_hub first: pip install huggingface_hub")

REPO_ID = "ZhiqiGao/Text2Opt-Bench"
REPO_TYPE = "dataset"
REPO_ROOT = Path(__file__).resolve().parent.parent

ASSETS = {
    "template":     ("Template/**",       REPO_ROOT / "synthetic_dataset"),
    "unstructured": ("Unstructured/**",   REPO_ROOT / "synthetic_dataset"),
    "train":        ("Template_train/**", REPO_ROOT / "synthetic_dataset"),
    "large":        ("Template_large/**", REPO_ROOT / "synthetic_dataset"),
    "ruler":        ("ruler_samples/**",  REPO_ROOT / "ruler"),
}

DEFAULT_TARGETS = ["train", "large", "ruler"]


def fetch(name: str) -> None:
    pattern, dest = ASSETS[name]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] downloading {pattern} -> {dest}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=pattern,
        local_dir=str(dest),
    )
    if name == "ruler":
        downloaded = dest / "ruler_samples"
        target = dest / "samples"
        if downloaded.exists() and not target.exists():
            downloaded.rename(target)
    print(f"[{name}] done")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only",
        action="append",
        choices=sorted(ASSETS.keys()),
        help="Restrict to specific asset(s). Repeat for multiple.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download every asset, including the bundled Template/ and Unstructured/.",
    )
    args = parser.parse_args()

    if args.all:
        targets = list(ASSETS.keys())
    elif args.only:
        targets = args.only
    else:
        targets = DEFAULT_TARGETS

    for name in dict.fromkeys(targets):
        fetch(name)


if __name__ == "__main__":
    main()
