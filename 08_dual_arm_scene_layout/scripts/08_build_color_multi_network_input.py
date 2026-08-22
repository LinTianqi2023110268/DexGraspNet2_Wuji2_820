#!/usr/bin/env python3
"""Build one 40k DGN2 input for ALL trusted requested-color objects.

pc=(1,40000,3), seg=(1,40000), edge=(1,40000) stay unchanged.
Context points come from GraspContextZone.
seg=0 is environment; seg=1..N are capture-local SourceZone targets.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2 as cv
import numpy as np

POINT_COUNT=40_000
RANDOM_SEED=0
MINIMUM_SAMPLED_POINTS_PER_TARGET=128

def backproject(depth_m, K):
    depth=np.asarray(depth_m,dtype=np.float32)
    depth=np.where(np.isfinite(depth)&(depth>0.0),depth,np.nan)
    h,w=depth.shape
    rows,cols=np.indices((h,w),dtype=np.float32)
    with np.errstate(invalid="ignore"):
        x=(cols-float(K[0,2]))*depth/float(K[0,0])
        y=(rows-float(K[1,2]))*depth/float(K[1,1])
    return np.stack((x,y,depth),axis=-1)

def official_depth_edges(depth_m):
    finite=np.isfinite(depth_m)&(depth_m>0.0)
    image=np.zeros(depth_m.shape,dtype=np.uint8)
    if np.any(finite):
        maximum=float(np.max(depth_m[finite]))
        image[finite]=np.clip(depth_m[finite]/maximum*200.0,0,200).astype(np.uint8)
    edges=cv.Canny(image,10,20)
    mask=edges>0
    mask[:,0]=mask[:,-1]=True
    mask[0,:]=mask[-1,:]=True
    for _ in range(5):
        new=mask.copy()
        new[1:,:]|=mask[:-1,:]
        new[:-1,:]|=mask[1:,:]
        new[:,1:]|=mask[:,:-1]
        new[:,:-1]|=mask[:,1:]
        mask=new
    edges[mask]=255
    return edges

def repair_sampling(selected,candidates,flat_labels,target_labels,rng,minimum):
    out=np.asarray(selected,dtype=np.int64).copy()
    sampled=flat_labels[out]
    for label in target_labels:
        current=int(np.count_nonzero(sampled==label))
        if current>=minimum:
            continue
        pool=candidates[flat_labels[candidates]==label]
        if len(pool)==0:
            raise RuntimeError(f"target_label={label} has no context pixels")
        need=minimum-current
        repl=rng.choice(pool,size=need,replace=len(pool)<need)
        bg=np.flatnonzero(sampled==0)
        if len(bg)<need:
            raise RuntimeError(f"not enough background slots for target_label={label}")
        slots=rng.choice(bg,size=need,replace=False)
        out[slots]=repl
        sampled=flat_labels[out]
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--capture-root",type=Path,required=True)
    p.add_argument("--context-depth",type=Path,required=True)
    p.add_argument("--context-mask",type=Path,required=True)
    p.add_argument("--target-label-image",type=Path,required=True)
    p.add_argument("--catalog",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--minimum-sampled-points-per-target",type=int,default=128)
    a=p.parse_args()

    capture=a.capture_root.resolve()
    out=a.output_root.resolve()
    out.mkdir(parents=True,exist_ok=True)

    depth=np.load(a.context_depth.resolve()).astype(np.float32)
    context=np.load(a.context_mask.resolve()).astype(bool)
    labels_img=np.load(a.target_label_image.resolve()).astype(np.int64)
    K=np.load(capture/"intrinsics.npy").astype(np.float64)
    T_wc=np.load(capture/"T_world_camera.npy").astype(np.float64)
    catalog=json.loads(a.catalog.resolve().read_text(encoding="utf-8"))
    labels=sorted(int(r["target_label"]) for r in catalog.get("objects",[]))
    if not labels:
        raise RuntimeError("trusted color catalog is empty")
    if depth.shape!=context.shape or depth.shape!=labels_img.shape:
        raise ValueError("depth/context/label image shape mismatch")

    camera=backproject(depth,K)
    flat_camera=camera.reshape(-1,3)
    flat_context=context.reshape(-1)
    flat_labels=labels_img.reshape(-1)
    valid=np.isfinite(flat_camera).all(axis=1)&(flat_camera[:,2]>0.0)
    candidates=np.flatnonzero(valid&flat_context)
    if len(candidates)==0:
        raise RuntimeError("no valid GraspContextZone points")
    for label in labels:
        if int(np.count_nonzero(flat_labels[candidates]==label))<=0:
            raise RuntimeError(f"target_label={label} has zero context points")

    rng=np.random.default_rng(RANDOM_SEED)
    selected=rng.choice(candidates,size=POINT_COUNT,replace=len(candidates)<POINT_COUNT)
    selected=repair_sampling(
        selected,candidates,flat_labels,labels,rng,
        int(a.minimum_sampled_points_per_target),
    )
    seg=flat_labels[selected].astype(np.int64)
    unknown=sorted(int(v) for v in np.unique(seg) if int(v)!=0 and int(v) not in labels)
    if unknown:
        raise RuntimeError(f"unexpected target labels: {unknown}")
    counts={label:int(np.count_nonzero(seg==label)) for label in labels}
    for label,count in counts.items():
        if count<int(a.minimum_sampled_points_per_target):
            raise RuntimeError(f"target_label={label} undersampled: {count}")

    settled=json.loads((capture/"settled_scene_manifest.json").read_text(encoding="utf-8"))
    world_from_source=np.asarray(settled["world_from_source_zone"],dtype=np.float64)
    source_from_world=np.linalg.inv(world_from_source)
    edge=official_depth_edges(depth)

    network_path=out/"network_input.npz"
    np.savez_compressed(
        network_path,
        pc=flat_camera[selected][None].astype(np.float32),
        seg=seg[None],
        edge=edge.reshape(-1)[selected][None].astype(np.int64),
        extrinsics=T_wc[None].astype(np.float32),
        pixel_indices=selected[None].astype(np.int64),
        target_labels=np.asarray(labels,dtype=np.int64),
        source_from_world=source_from_world.astype(np.float64),
        world_from_source=world_from_source.astype(np.float64),
    )
    report={
        "schema_version":1,
        "status":"color_multi_network_input_ready",
        "point_count":POINT_COUNT,
        "context_candidate_count":int(len(candidates)),
        "target_labels":labels,
        "sampled_target_counts":{str(k):int(v) for k,v in counts.items()},
        "sampled_background_count":int(np.count_nonzero(seg==0)),
        "minimum_sampled_points_per_target":int(a.minimum_sampled_points_per_target),
        "pc_context":"GraspContextZone",
        "target_membership":"SourceZone trusted DINO+SAM objects only",
        "network_input":str(network_path),
    }
    (out/"network_input.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print(json.dumps({
        "status":"PASS","point_count":POINT_COUNT,"target_labels":labels,
        "sampled_target_counts":report["sampled_target_counts"],
        "network_input":str(network_path),
    },ensure_ascii=False))

if __name__=="__main__":
    main()
