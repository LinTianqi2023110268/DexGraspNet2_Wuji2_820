from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


RIGHT_ARM_JOINTS: Tuple[str, ...] = (
    "arm_r_joint_1",
    "arm_r_joint_2",
    "arm_r_joint_3",
    "arm_r_joint_4",
    "arm_r_joint_5",
    "arm_r_joint_6",
    "arm_r_joint_7",
)


@dataclass
class VisualizationBundle:
    scene_points_base: np.ndarray              # [P,3]
    sphere_centers_base: np.ndarray            # [N,S,3]
    sphere_radii_m: np.ndarray                 # [S]
    sphere_link_names: np.ndarray              # [S]
    sphere_active_mask: np.ndarray             # [S] bool
    ee_positions_base: np.ndarray              # [N,3]
    time_s: np.ndarray                         # [N]
    q_rad: np.ndarray                          # [N,7]
    joint_names: np.ndarray                    # [7]
    frame_min_clearance_m: Optional[np.ndarray] = None   # [N]
    frame_worst_sphere_index: Optional[np.ndarray] = None # [N]
    scene_colors_rgb: Optional[np.ndarray] = None        # [P,3], uint8/float
    metadata_json: str = "{}"


def _as(a, dtype=None):
    x = np.asarray(a, dtype=dtype)
    if not np.all(np.isfinite(x)) and x.dtype.kind in "fc":
        raise ValueError("bundle contains non-finite numeric values")
    return x


def validate_bundle(b: VisualizationBundle) -> VisualizationBundle:
    p = _as(b.scene_points_base, np.float64)
    c = _as(b.sphere_centers_base, np.float64)
    r = _as(b.sphere_radii_m, np.float64).reshape(-1)
    links = np.asarray(b.sphere_link_names).astype(str).reshape(-1)
    active = np.asarray(b.sphere_active_mask, dtype=bool).reshape(-1)
    ee = _as(b.ee_positions_base, np.float64)
    t = _as(b.time_s, np.float64).reshape(-1)
    q = _as(b.q_rad, np.float64)
    jn = np.asarray(b.joint_names).astype(str).reshape(-1)

    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"scene_points_base must be [P,3], got {p.shape}")
    if c.ndim != 3 or c.shape[2] != 3:
        raise ValueError(f"sphere_centers_base must be [N,S,3], got {c.shape}")
    n, s, _ = c.shape
    if r.shape != (s,):
        raise ValueError(f"sphere_radii_m must be [{s}], got {r.shape}")
    if links.shape != (s,):
        raise ValueError(f"sphere_link_names must be [{s}], got {links.shape}")
    if active.shape != (s,):
        raise ValueError(f"sphere_active_mask must be [{s}], got {active.shape}")
    if ee.shape != (n, 3):
        raise ValueError(f"ee_positions_base must be [{n},3], got {ee.shape}")
    if t.shape != (n,):
        raise ValueError(f"time_s must be [{n}], got {t.shape}")
    if q.shape != (n, 7):
        raise ValueError(f"q_rad must be [{n},7], got {q.shape}")
    if tuple(jn.tolist()) != RIGHT_ARM_JOINTS:
        raise ValueError(f"joint_names must exactly be {RIGHT_ARM_JOINTS}, got {jn.tolist()}")
    if np.any(r <= 0):
        raise ValueError("all sphere radii must be positive")
    if n <= 1:
        raise ValueError("trajectory must contain more than one point")
    if np.any(np.diff(t) < -1e-12):
        raise ValueError("time_s must be monotonic")

    fmc = None
    if b.frame_min_clearance_m is not None:
        fmc = _as(b.frame_min_clearance_m, np.float64).reshape(-1)
        if fmc.shape != (n,):
            raise ValueError(f"frame_min_clearance_m must be [{n}], got {fmc.shape}")

    fws = None
    if b.frame_worst_sphere_index is not None:
        fws = np.asarray(b.frame_worst_sphere_index, dtype=np.int64).reshape(-1)
        if fws.shape != (n,):
            raise ValueError(f"frame_worst_sphere_index must be [{n}], got {fws.shape}")
        if np.any((fws < -1) | (fws >= s)):
            raise ValueError("frame_worst_sphere_index contains invalid sphere indices")

    colors = None
    if b.scene_colors_rgb is not None:
        colors = np.asarray(b.scene_colors_rgb)
        if colors.shape != (p.shape[0], 3):
            raise ValueError(
                f"scene_colors_rgb must be [{p.shape[0]},3], got {colors.shape}"
            )

    return VisualizationBundle(
        scene_points_base=p,
        sphere_centers_base=c,
        sphere_radii_m=r,
        sphere_link_names=links,
        sphere_active_mask=active,
        ee_positions_base=ee,
        time_s=t,
        q_rad=q,
        joint_names=jn,
        frame_min_clearance_m=fmc,
        frame_worst_sphere_index=fws,
        scene_colors_rgb=colors,
        metadata_json=str(b.metadata_json),
    )


def save_bundle(path: str | Path, bundle: VisualizationBundle) -> Path:
    b = validate_bundle(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_points_base": b.scene_points_base.astype(np.float32),
        "sphere_centers_base": b.sphere_centers_base.astype(np.float32),
        "sphere_radii_m": b.sphere_radii_m.astype(np.float32),
        "sphere_link_names": b.sphere_link_names.astype("U"),
        "sphere_active_mask": b.sphere_active_mask.astype(np.uint8),
        "ee_positions_base": b.ee_positions_base.astype(np.float32),
        "time_s": b.time_s.astype(np.float64),
        "q_rad": b.q_rad.astype(np.float32),
        "joint_names": b.joint_names.astype("U"),
        "metadata_json": np.asarray(b.metadata_json, dtype="U"),
    }
    if b.frame_min_clearance_m is not None:
        payload["frame_min_clearance_m"] = b.frame_min_clearance_m.astype(np.float32)
    if b.frame_worst_sphere_index is not None:
        payload["frame_worst_sphere_index"] = b.frame_worst_sphere_index.astype(np.int32)
    if b.scene_colors_rgb is not None:
        payload["scene_colors_rgb"] = b.scene_colors_rgb
    np.savez_compressed(path, **payload)
    return path


def load_bundle(path: str | Path) -> VisualizationBundle:
    z = np.load(Path(path), allow_pickle=False)
    get = lambda k, default=None: z[k] if k in z.files else default
    b = VisualizationBundle(
        scene_points_base=z["scene_points_base"],
        sphere_centers_base=z["sphere_centers_base"],
        sphere_radii_m=z["sphere_radii_m"],
        sphere_link_names=z["sphere_link_names"],
        sphere_active_mask=z["sphere_active_mask"].astype(bool),
        ee_positions_base=z["ee_positions_base"],
        time_s=z["time_s"],
        q_rad=z["q_rad"],
        joint_names=z["joint_names"],
        frame_min_clearance_m=get("frame_min_clearance_m"),
        frame_worst_sphere_index=get("frame_worst_sphere_index"),
        scene_colors_rgb=get("scene_colors_rgb"),
        metadata_json=str(get("metadata_json", np.asarray("{}", dtype="U")).item()),
    )
    return validate_bundle(b)
