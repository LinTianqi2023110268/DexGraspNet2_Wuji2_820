#!/usr/bin/env python3
"""Official DGN2 cate=True inference for ALL trusted requested-color objects."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT_ROOT=Path(__file__).resolve().parents[2]
OFFICIAL_ROOT=PROJECT_ROOT/"03_prediction_network/official_core"
CHECKPOINT=OFFICIAL_ROOT/"experiments/dex_ours/ckpt/ckpt_50000.pth"
PROPOSAL_COUNT=1024
GRASPNESS_SCALE=5.0
RANDOM_SEED=0
SEED_MATCH_TOLERANCE_M=1.0e-6

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--catalog",type=Path,required=True)
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--seed",type=int,default=0)
    a=p.parse_args()

    root=a.input_root.resolve()
    input_path=root/"network_input.npz"
    output_path=root/"official_leap_color_multi_ranked.npz"
    report_path=root/"official_leap_color_multi_ranked.json"
    if a.rounds<1: raise ValueError("--rounds must be positive")
    if not input_path.is_file(): raise FileNotFoundError(input_path)
    if not CHECKPOINT.is_file(): raise FileNotFoundError(CHECKPOINT)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is not visible")

    catalog=json.loads(a.catalog.resolve().read_text(encoding="utf-8"))
    catalog_labels=sorted(int(r["target_label"]) for r in catalog.get("objects",[]))
    if not catalog_labels: raise RuntimeError("empty trusted color catalog")

    with np.load(input_path,allow_pickle=False) as z:
        pc_np=np.asarray(z["pc"],dtype=np.float32)
        seg_np=np.asarray(z["seg"],dtype=np.int64)
        edge_np=np.asarray(z["edge"],dtype=np.int64)
        extrinsics=np.asarray(z["extrinsics"],dtype=np.float32)
        input_labels=sorted(int(v) for v in np.asarray(z["target_labels"]).tolist())
    if pc_np.shape!=(1,40000,3): raise RuntimeError(f"invalid pc shape {pc_np.shape}")
    if seg_np.shape!=(1,40000): raise RuntimeError(f"invalid seg shape {seg_np.shape}")
    if edge_np.shape!=(1,40000): raise RuntimeError(f"invalid edge shape {edge_np.shape}")
    if input_labels!=catalog_labels:
        raise RuntimeError(f"catalog/network labels mismatch: {catalog_labels} vs {input_labels}")
    nonzero=sorted(int(v) for v in np.unique(seg_np) if int(v)!=0)
    if nonzero!=catalog_labels:
        raise RuntimeError(f"network nonzero labels {nonzero} != catalog {catalog_labels}")

    old_cwd=Path.cwd()
    os.chdir(OFFICIAL_ROOT)
    sys.path.insert(0,str(OFFICIAL_ROOT))
    try:
        from src.network.model import get_model
        from src.utils.config import ckpt_to_config
        from src.utils.dataset import get_sparse_tensor
        from src.utils.robot_model import RobotModel
        from src.utils.util import set_seed

        set_seed(a.seed)
        device=torch.device("cuda:0")
        config=ckpt_to_config(str(CHECKPOINT))
        model=get_model(config.model)
        model.config.voxel_size=config.data.voxel_size
        checkpoint=torch.load(str(CHECKPOINT),map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
        model.to(device).eval()

        pc=torch.as_tensor(pc_np,dtype=torch.float32)
        seg=torch.as_tensor(seg_np,dtype=torch.long)
        edge=torch.as_tensor(edge_np,dtype=torch.long)
        sparse=get_sparse_tensor(pc,config.data.voxel_size)
        sparse["seg"]=seg
        sparse={k:v.to(device) for k,v in sparse.items()}

        chunks=[[] for _ in range(8)]
        with torch.no_grad():
            for round_index in range(a.rounds):
                result=model.sample(
                    sparse,PROPOSAL_COUNT,
                    graspness_scale=GRASPNESS_SCALE,
                    allow_fail=True,cate=True,
                    edge=edge.to(device),
                    with_score_parts=True,with_point=True,
                )
                if len(result)!=8:
                    raise RuntimeError(f"unexpected sampler result length {len(result)}")
                for vi,(bucket,value) in enumerate(zip(chunks,result)):
                    arr=value.detach().cpu().numpy()
                    if vi==7:
                        arr=arr.reshape(1,PROPOSAL_COUNT,3)
                    bucket.append(arr)
                print(
                    f"[SAMPLER][COLOR_MULTI_CATE] round {round_index+1}/{a.rounds}",
                    flush=True,
                )
        rotation,translation,qpos,score,object_index,graspness,log_prob,seed=[
            np.concatenate(bucket,axis=1) for bucket in chunks
        ]
        robot=RobotModel(
            "robot_models/urdf/leap_hand_simplified.urdf",
            "robot_models/meta/leap_hand/meta.yaml",
        )
    finally:
        os.chdir(old_cwd)

    proposal_count=PROPOSAL_COUNT*a.rounds
    seed=seed.reshape(1,proposal_count,3)
    tree=cKDTree(pc_np[0].astype(np.float64))
    distance,point_index=tree.query(seed[0].astype(np.float64),k=1)
    max_seed_distance=float(np.max(distance))
    if max_seed_distance>SEED_MATCH_TOLERANCE_M:
        raise RuntimeError(f"seed/input mismatch max={max_seed_distance:.9g} m")
    seed_label=seg_np[0,point_index].astype(np.int64)
    official_label=object_index[0].astype(np.int64)
    if np.any(seed_label==0):
        raise RuntimeError(f"cate=True returned {int(np.count_nonzero(seed_label==0))} background seeds")
    if not np.array_equal(seed_label,official_label):
        raise RuntimeError(
            f"official object_index mismatch for "
            f"{int(np.count_nonzero(seed_label!=official_label))} candidates"
        )
    unexpected=sorted(int(v) for v in np.unique(seed_label) if int(v) not in catalog_labels)
    if unexpected: raise RuntimeError(f"unexpected seed labels {unexpected}")

    R_wc=extrinsics[0,:3,:3]
    t_wc=extrinsics[0,:3,3]
    rotation_world=np.einsum("ij,njk->nik",R_wc,rotation[0])
    translation_world=translation[0]@R_wc.T+t_wc
    seed_world=seed[0]@R_wc.T+t_wc
    global_order=np.argsort(-score[0],kind="stable")
    counts={label:int(np.count_nonzero(seed_label==label)) for label in catalog_labels}

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
        seed_target_label=seed_label.astype(np.int64),
        official_sampler_object_index=official_label.astype(np.int64),
        global_score_descending_candidate_index=global_order.astype(np.int64),
        target_score_descending_candidate_index=global_order.astype(np.int64),
        target_candidate_index=np.arange(proposal_count,dtype=np.int64),
        target_labels=np.asarray(catalog_labels,dtype=np.int64),
        extrinsic_T_world_camera=extrinsics[0].astype(np.float32),
    )
    report={
        "schema_version":1,
        "status":"official_leap_color_multi_ready",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "sampling_mode":"color_multi_cate",
        "official_cate":True,
        "proposal_count":proposal_count,
        "sampler_rounds":a.rounds,
        "proposals_per_round":PROPOSAL_COUNT,
        "target_labels":catalog_labels,
        "proposal_count_by_target_label":{str(k):int(v) for k,v in counts.items()},
        "background_seed_count":int(np.count_nonzero(seed_label==0)),
        "max_seed_to_input_distance_m":max_seed_distance,
        "selection_rule":(
            "no perception-level best object; all trusted color objects sampled "
            "by official cate=True; downstream globally ordered by official DGN2 "
            "score and route feasibility"
        ),
        "official_score_formula":"log_prob + 5 * graspness",
        "checkpoint":str(CHECKPOINT),
        "checkpoint_sha256":sha256(CHECKPOINT),
        "network_input":str(input_path),
        "catalog":str(a.catalog.resolve()),
        "output":str(output_path),
        "simulator_identity_used":False,
    }
    report_path.write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print(
        f"[PASS][COLOR_MULTI_CATE] proposals={proposal_count} "
        f"labels={catalog_labels} counts={counts}",flush=True
    )

if __name__=="__main__":
    main()
