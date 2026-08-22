#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def project(points_world,K,T_wc):
    T_cw=np.linalg.inv(T_wc)
    hom=np.concatenate([points_world,np.ones((len(points_world),1))],axis=1)
    cam=(T_cw@hom.T).T[:,:3]
    z=cam[:,2]
    valid=z>1e-6
    u=K[0,0]*cam[:,0]/np.maximum(z,1e-9)+K[0,2]
    v=K[1,1]*cam[:,1]/np.maximum(z,1e-9)+K[1,2]
    return u,v,valid

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--capture-root",type=Path,required=True)
    p.add_argument("--prediction",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--top-n",type=int,default=100)
    a=p.parse_args()
    capture=a.capture_root.resolve()
    overlay=Image.open(capture/"rgb.png").convert("RGBA")
    draw=ImageDraw.Draw(overlay)
    K=np.load(capture/"intrinsics.npy").astype(np.float64)
    T_wc=np.load(capture/"T_world_camera.npy").astype(np.float64)
    with np.load(a.prediction.resolve(),allow_pickle=False) as z:
        seed=np.asarray(z["seed_point_world"],dtype=np.float64)
        root=np.asarray(z["translation_world"],dtype=np.float64)
        labels=np.asarray(z["seed_target_label"],dtype=np.int64)
        order=np.asarray(z["global_score_descending_candidate_index"],dtype=np.int64)
    us,vs,oks=project(seed,K,T_wc)
    ur,vr,okr=project(root,K,T_wc)
    n=min(int(a.top_n),len(order))
    chosen=order[:n]
    for rank,idx in enumerate(chosen):
        i=int(idx)
        if not (oks[i] and okr[i]): continue
        sx,sy=float(us[i]),float(vs[i])
        rx,ry=float(ur[i]),float(vr[i])
        draw.line((sx,sy,rx,ry),fill=(255,255,255,90),width=1)
        draw.ellipse((sx-3,sy-3,sx+3,sy+3),fill=(0,255,80,230))
        fill=(255,255,0,240) if rank<20 else (0,220,255,180)
        rr=5 if rank<20 else 3
        draw.ellipse((rx-rr,ry-rr,rx+rr,ry+rr),fill=fill)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    overlay.convert("RGB").save(a.output)
    dist=np.linalg.norm(root-seed,axis=1)
    report={
        "schema_version":1,
        "legend":{
            "green":"DGN2 seed",
            "cyan":"predicted LEAP root",
            "yellow":"Top20 predicted LEAP root",
            "white_line":"seed -> root",
        },
        "root_to_seed_m":{
            "p50":float(np.quantile(dist,0.50)),
            "p90":float(np.quantile(dist,0.90)),
            "p99":float(np.quantile(dist,0.99)),
            "max":float(np.max(dist)),
        },
        "target_label_counts_top_n":{
            str(label):int(np.count_nonzero(labels[chosen]==label))
            for label in sorted(int(v) for v in np.unique(labels))
        },
        "output":str(a.output.resolve()),
    }
    a.output.with_suffix(".json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print(json.dumps({"status":"PASS",**report},ensure_ascii=False))

if __name__=="__main__":
    main()
