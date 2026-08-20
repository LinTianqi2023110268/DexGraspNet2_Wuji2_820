#!/usr/bin/env python3
"""View a prepared scene folder with Trimesh.

This is a lightweight offline viewer for folders such as:

  02_training_dataset/data/scene_datasets/.../scenes/scene_0000

It reads ``scene_manifest.json`` and places the table plus all object meshes in
the manifest frame.  It does not start Isaac, simulate physics, or run any
grasping model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "trimesh/outputs/scene_folder_views"


PALETTE = np.asarray(
    [
        [230, 80, 70, 190],
        [70, 170, 240, 190],
        [80, 210, 110, 190],
        [245, 185, 65, 190],
        [180, 120, 235, 190],
        [80, 215, 210, 190],
        [235, 120, 185, 190],
        [160, 160, 160, 190],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-folder",
        type=Path,
        required=True,
        help="Folder containing scene_manifest.json.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional output .glb path. Defaults to trimesh/outputs/scene_folder_views/<scene>.glb.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Trimesh viewer after exporting. Requires a working GUI.",
    )
    parser.add_argument(
        "--calibrated-layout",
        action="store_true",
        help=(
            "Embed the local scene_manifest poses into the calibrated layout "
            "SourceZone and draw SourceZone/PlacementZone markers."
        ),
    )
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_color(mesh: trimesh.Trimesh, rgba: np.ndarray | list[int]) -> None:
    mesh.visual.face_colors = np.asarray(rgba, dtype=np.uint8)


def load_calibrated_layout() -> dict:
    path = PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
    return json.loads(path.read_text(encoding="utf-8"))


def rigid_source_zone_transform(layout: dict) -> np.ndarray:
    # SourceZone is authored as a scaled Cube to show its area.  The scene
    # manifest coordinates are already metric SourceZone-local tabletop poses,
    # so the runtime spawn contract uses only the rigid frame, not the visual
    # Cube scale.  Current calibrated SourceZone has no rotation; keep this
    # explicit here so the offline viewer matches Persistent Isaac.
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(
        layout["transforms"]["source_zone"]["position_world_m"], dtype=np.float64
    )
    return transform


def zone_box(center: np.ndarray, size: np.ndarray, rgba: list[int]) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=size)
    box.apply_translation(center)
    set_color(box, rgba)
    return box


def load_centered_object_mesh(record: dict) -> trimesh.Trimesh:
    asset = record["asset"]
    loaded = trimesh.load(project_path(asset["source_obj"]), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = loaded.copy()
    mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64)
        - np.asarray(asset.get("native_aabb_center", [0.0, 0.0, 0.0]), dtype=np.float64)
    ) * float(asset.get("scale", 1.0))
    return mesh


def build_scene(manifest: dict, *, calibrated_layout: bool = False) -> trimesh.Scene:
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.axis(origin_size=0.004, axis_length=0.10),
        geom_name="world_frame",
    )

    world_from_local = np.eye(4, dtype=np.float64)
    if calibrated_layout:
        layout = load_calibrated_layout()
        world_from_local = rigid_source_zone_transform(layout)
        table_size = np.asarray(layout["geometry"]["table_size_m"], dtype=np.float64)
        table_center = np.asarray(
            layout["transforms"]["table"]["position_world_m"], dtype=np.float64
        )
        source_center = np.asarray(
            layout["transforms"]["source_zone"]["position_world_m"], dtype=np.float64
        )
        source_size = np.asarray(
            layout["geometry"]["source_zone_size_m"], dtype=np.float64
        )
        placement_center = np.asarray(
            layout["transforms"]["placement_zone"]["position_world_m"], dtype=np.float64
        )
        placement_size = np.asarray(
            layout["geometry"]["placement_zone_size_m"], dtype=np.float64
        )
        scene.add_geometry(
            zone_box(source_center, source_size, [60, 120, 255, 85]),
            geom_name="SourceZone_blue_visual_no_collision",
        )
        scene.add_geometry(
            zone_box(placement_center, placement_size, [70, 230, 90, 85]),
            geom_name="PlacementZone_green_visual_no_collision",
        )
    else:
        table_info = manifest.get("table", {})
        table_size = np.asarray(table_info.get("size_m", [0.5, 0.3, 0.02]), dtype=np.float64)
        table_top = float(table_info.get("top_z_m", 0.0))
        table_center = np.asarray([0.0, 0.0, table_top - 0.5 * table_size[2]], dtype=np.float64)
    table = trimesh.creation.box(extents=table_size)
    table.apply_translation(table_center)
    set_color(table, [145, 145, 145, 65])
    scene.add_geometry(table, geom_name="table")

    for i, record in enumerate(manifest.get("objects", [])):
        mesh = load_centered_object_mesh(record)
        local_pose = np.asarray(record["T_world_centered_object"], dtype=np.float64)
        mesh.apply_transform(world_from_local @ local_pose)
        color = PALETTE[i % len(PALETTE)].copy()
        set_color(mesh, color)
        name = (
            f"object_{int(record.get('segmentation_id', i)):03d}_"
            f"{record.get('object_code', 'unknown')}"
        )
        scene.add_geometry(mesh, geom_name=name)
    return scene


def main() -> int:
    args = parse_args()
    scene_folder = args.scene_folder.expanduser().resolve()
    manifest_path = scene_folder / "scene_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing scene_manifest.json: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    display = build_scene(manifest, calibrated_layout=bool(args.calibrated_layout))

    if args.export is None:
        destination = DEFAULT_OUTPUT_DIR / f"{scene_folder.name}.glb"
    else:
        destination = args.export.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    display.export(destination)

    names = [
        str(record.get("object_code", "unknown"))
        for record in manifest.get("objects", [])
    ]
    print(f"scene_folder: {scene_folder}")
    print(f"objects     : {len(names)}")
    print(f"exported    : {destination}")
    print(f"layout mode : {'calibrated SourceZone/world' if args.calibrated_layout else 'manifest local tabletop'}")
    if names:
        print("object codes:")
        for name in names:
            print(f"  - {name}")
    if args.show:
        display.show(caption=f"Scene folder: {scene_folder.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
