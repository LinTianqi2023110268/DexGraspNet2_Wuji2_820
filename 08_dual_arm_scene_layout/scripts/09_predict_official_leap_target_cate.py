#!/usr/bin/env python3
"""
New DGN2 target-conditioned inference line.

It preserves the full 40k scene input and uses the official
GraspnessSample.sample(..., cate=True) branch.  The input segmentation must
contain exactly one non-zero category: the current perception-selected target.

The legacy scene-wide/post-filter script
09_predict_official_leap_target.py is intentionally NOT replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout"
OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network/official_core"
CHECKPOINT = OFFICIAL_ROOT / "experiments/dex_ours/ckpt/ckpt_50000.pth"

PROPOSAL_COUNT = 1024
GRASPNESS_SCALE = 5.0
RANDOM_SEED = 0
SEED_MATCH_TOLERANCE_M = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_input(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        required = {"pc", "seg", "edge", "extrinsics", "target_segmentation_id"}
        missing = sorted(required - set(z.files))
        if missing:
            raise RuntimeError(f"network_input missing fields: {missing}")
        pc = np.asarray(z["pc"], dtype=np.float32)
        seg = np.asarray(z["seg"], dtype=np.int64)
        edge = np.asarray(z["edge"], dtype=np.int64)
        extrinsics = np.asarray(z["extrinsics"], dtype=np.float32)
        target_id = int(np.asarray(z["target_segmentation_id"]).item())

    if pc.shape != (1, 40000, 3):
        raise ValueError(f"expected pc=(1,40000,3), got {pc.shape}")
    if seg.shape != (1, 40000):
        raise ValueError(f"expected seg=(1,40000), got {seg.shape}")
    if edge.shape != (1, 40000):
        raise ValueError(f"expected edge=(1,40000), got {edge.shape}")
    if extrinsics.shape != (1, 4, 4):
        raise ValueError(f"expected extrinsics=(1,4,4), got {extrinsics.shape}")
    if target_id == 0:
        raise RuntimeError("target_segmentation_id must be non-zero")

    target_count = int(np.count_nonzero(seg == target_id))
    background_count = int(np.count_nonzero(seg == 0))
    nonzero_ids = sorted(int(v) for v in np.unique(seg) if int(v) != 0)

    if target_count <= 0:
        raise RuntimeError(f"target id {target_id} has zero sampled points")
    if nonzero_ids != [target_id]:
        raise RuntimeError(
            "target_cate requires exactly one non-zero segmentation id: "
            f"expected [{target_id}], got {nonzero_ids}"
        )
    if background_count <= 0:
        raise RuntimeError(
            "target_cate requires full-scene context; no background points remain"
        )

    return {
        "pc": pc,
        "seg": seg,
        "edge": edge,
        "extrinsics": extrinsics,
        "target_id": target_id,
        "target_count": target_count,
        "background_count": background_count,
        "nonzero_ids": nonzero_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="target")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()

    if args.rounds < 1:
        raise ValueError("--rounds must be positive")

    root = args.input_root.resolve()
    input_path = root / "network_input.npz"
    output_path = root / "official_leap_1024_target_ranked.npz"
    report_path = output_path.with_suffix(".json")
    audit_path = root / "official_leap_target_cate_audit.json"

    for p in (input_path, CHECKPOINT):
        if not p.is_file():
            raise FileNotFoundError(p)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; use the graspnet2.0 environment")

    loaded = load_and_validate_input(input_path)
    pc_np = loaded["pc"]
    seg_np = loaded["seg"]
    edge_np = loaded["edge"]
    extrinsics = loaded["extrinsics"]
    target_id = int(loaded["target_id"])
    input_target_count = int(loaded["target_count"])
    nonzero_ids = list(loaded["nonzero_ids"])

    input_audit = {
        "schema_version": 1,
        "status": "TARGET_CATE_INPUT_VALID",
        "sampling_mode": "target_cate",
        "official_cate": True,
        "network_input": str(input_path),
        "full_scene_point_count": int(pc_np.shape[1]),
        "target_segmentation_id": target_id,
        "target_sampled_point_count": input_target_count,
        "target_sampled_fraction": float(input_target_count / pc_np.shape[1]),
        "background_sampled_point_count": int(loaded["background_count"]),
        "nonzero_segmentation_ids": nonzero_ids,
        "full_scene_context_preserved": True,
        "seed_domain": "seg == target_segmentation_id",
        "simulator_target_identity_used": False,
    }
    audit_path.write_text(
        json.dumps(input_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    os.chdir(OFFICIAL_ROOT)
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        from src.network.model import get_model
        from src.utils.config import ckpt_to_config
        from src.utils.dataset import get_sparse_tensor
        from src.utils.robot_model import RobotModel
        from src.utils.util import set_seed

        set_seed(args.seed)
        device = torch.device("cuda:0")
        config = ckpt_to_config(str(CHECKPOINT))
        model = get_model(config.model)
        model.config.voxel_size = config.data.voxel_size
        checkpoint = torch.load(str(CHECKPOINT), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device).eval()

        pc = torch.as_tensor(pc_np, dtype=torch.float32)
        seg = torch.as_tensor(seg_np, dtype=torch.long)
        edge = torch.as_tensor(edge_np, dtype=torch.long)

        sparse = get_sparse_tensor(pc, config.data.voxel_size)
        sparse["seg"] = seg
        sparse = {key: value.to(device) for key, value in sparse.items()}

        chunks = [[] for _ in range(8)]
        with torch.no_grad():
            for round_index in range(args.rounds):
                result = model.sample(
                    sparse,
                    PROPOSAL_COUNT,
                    graspness_scale=GRASPNESS_SCALE,
                    allow_fail=True,
                    cate=True,
                    edge=edge.to(device),
                    with_score_parts=True,
                    with_point=True,
                )
                if len(result) != 8:
                    raise RuntimeError(
                        f"unexpected sampler result length: {len(result)}"
                    )
                for value_index, (bucket, value) in enumerate(zip(chunks, result)):
                    array = value.detach().cpu().numpy()
                    if value_index == 7:
                        array = array.reshape(1, PROPOSAL_COUNT, 3)
                    bucket.append(array)
                print(
                    f"[SAMPLER][TARGET_CATE] round {round_index + 1}/{args.rounds}",
                    flush=True,
                )

        (
            rotation,
            translation,
            qpos,
            score,
            object_index,
            graspness,
            log_prob,
            seed,
        ) = [np.concatenate(bucket, axis=1) for bucket in chunks]

        proposal_count = PROPOSAL_COUNT * args.rounds
        seed = seed.reshape(1, proposal_count, 3)
        robot = RobotModel(
            "robot_models/urdf/leap_hand_simplified.urdf",
            "robot_models/meta/leap_hand/meta.yaml",
        )
    finally:
        os.chdir(original_cwd)

    # Strict audit: official cate=True seeds must map back to the target points.
    tree = cKDTree(pc_np[0].astype(np.float64))
    distance, point_index = tree.query(seed[0].astype(np.float64), k=1)
    max_seed_distance = float(np.max(distance))
    if max_seed_distance > SEED_MATCH_TOLERANCE_M:
        raise RuntimeError(
            f"seed/input mismatch: maximum={max_seed_distance:.9g} m"
        )

    seed_segmentation = seg_np[0, point_index]
    target_mask = seed_segmentation == target_id
    target_candidates = np.flatnonzero(target_mask)
    non_target_candidates = np.flatnonzero(~target_mask)

    if len(non_target_candidates):
        bad_ids = sorted(
            int(v) for v in np.unique(seed_segmentation[non_target_candidates])
        )
        raise RuntimeError(
            "target_cate contract violation: "
            f"{len(non_target_candidates)}/{proposal_count} seeds are non-target; "
            f"ids={bad_ids}"
        )
    if len(target_candidates) != proposal_count:
        raise RuntimeError(
            f"target candidate accounting mismatch: "
            f"{len(target_candidates)} != {proposal_count}"
        )

    r_wc = extrinsics[0, :3, :3]
    t_wc = extrinsics[0, :3, 3]
    rotation_world = np.einsum("ij,njk->nik", r_wc, rotation[0])
    translation_world = translation[0] @ r_wc.T + t_wc
    seed_world = seed[0] @ r_wc.T + t_wc

    # Preserve existing downstream ranking/output contract.
    target_order = target_candidates[
        np.argsort(-score[0, target_candidates], kind="stable")
    ]
    all_order = np.argsort(-score[0], kind="stable")
    best = int(target_order[0])

    np.savez_compressed(
        output_path,
        rotation_world=rotation_world.astype(np.float32),
        translation_world=translation_world.astype(np.float32),
        rotation_camera=rotation[0].astype(np.float32),
        translation_camera=translation[0].astype(np.float32),
        leap_qpos_rad=qpos[0].astype(np.float32),
        leap_joint_order=np.asarray(robot.joint_names),
        score=score[0].astype(np.float32),
        graspness=graspness[0].astype(np.float32),
        log_prob=log_prob[0].astype(np.float32),
        seed_point_world=seed_world.astype(np.float32),
        seed_point_camera=seed[0].astype(np.float32),
        seed_point_input_index=point_index.astype(np.int64),
        seed_segmentation_id=seed_segmentation.astype(np.int64),
        target_segmentation_id=np.asarray(target_id, dtype=np.int64),
        target_candidate_index=target_candidates.astype(np.int64),
        target_score_descending_candidate_index=target_order.astype(np.int64),
        all_score_descending_candidate_index=all_order.astype(np.int64),
        official_sampler_object_index=object_index[0].astype(np.int64),
        extrinsic_T_world_camera=extrinsics[0].astype(np.float32),
    )

    report = {
        "schema_version": 2,
        "status": "official_leap_target_candidates_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_query": args.target,
        "sampling_mode": "target_cate",
        "official_cate": True,
        "target_segmentation_id": target_id,
        "input_full_scene_point_count": int(pc_np.shape[1]),
        "input_target_point_count": input_target_count,
        "input_target_fraction": float(input_target_count / pc_np.shape[1]),
        "input_nonzero_segmentation_ids": nonzero_ids,
        "proposal_count": proposal_count,
        "sampler_random_seed": args.seed,
        "sampler_rounds": args.rounds,
        "target_proposal_count": int(len(target_candidates)),
        "non_target_seed_count": int(len(non_target_candidates)),
        "max_seed_to_input_distance_m": max_seed_distance,
        "seed_selection_rule": (
            "official cate=True; only non-zero segmentation category is the "
            "current perception target"
        ),
        "selection_rule": "target-conditioned proposals; descending official score",
        "official_score_formula": "log_prob + 5 * graspness",
        "selected_candidate_index": best,
        "selected_score": float(score[0, best]),
        "selected_graspness": float(graspness[0, best]),
        "selected_log_prob": float(log_prob[0, best]),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "device": torch.cuda.get_device_name(0),
        "network_input": str(input_path),
        "input_audit": str(audit_path),
        "output": str(output_path),
        "legacy_scene_postfilter_script_untouched": str(
            PROJECT_ROOT
            / "08_dual_arm_scene_layout/scripts/09_predict_official_leap_target.py"
        ),
        "simulator_target_identity_used": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[PASS][TARGET_CATE] full-scene input: {pc_np.shape[1]}")
    print(
        f"[PASS][TARGET_CATE] target input points: "
        f"{input_target_count}/{pc_np.shape[1]}"
    )
    print(
        f"[PASS][TARGET_CATE] target proposals: "
        f"{len(target_candidates)}/{proposal_count}"
    )
    print(f"[BEST TARGET] candidate={best}; score={score[0, best]:.6f}")
    print(f"[OK] {output_path}")
    print(f"[AUDIT] {audit_path}")


if __name__ == "__main__":
    main()
