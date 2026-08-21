#!/usr/bin/env python3
"""Offline DGN2 membership audit before running either sampler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CLOSED_LOOP = HERE.parents[1]
if str(CLOSED_LOOP) not in sys.path:
    sys.path.insert(0, str(CLOSED_LOOP))

from dgn2_sampling_policy import audit_network_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("scene_postfilter", "target_cate"),
        default="target_cate",
    )
    args = parser.parse_args()

    result = audit_network_input(
        args.input_root.resolve() / "network_input.npz",
        expected_target_id=args.target_id,
        mode=args.mode,
    )
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
