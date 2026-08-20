# Portable Demo Scene

This directory contains one self-contained dynamic-object scene exported from a real closed-loop run.

- Source session: `20260820_124401`
- Source scene: `scene_0020`
- Object count: `6`
- Runtime scene manifest: `09_portable_demo_scene/scene/portable_scene_manifest.json`

The full training dataset under `02_training_dataset/data/` is not required for this demo scene's object assets.
Robot assets are not duplicated; after clone, run submodule setup for the existing robot/vendor assets.

## Fresh clone setup

```bash
git clone https://github.com/LinTianqi2023110268/DexGraspNet2_Wuji2_820.git
cd DexGraspNet2_Wuji2_820
git submodule update --init --recursive
```

Validate the portable scene without launching Isaac:

```bash
python 09_portable_demo_scene/scripts/validate_portable_scene.py
python 09_portable_demo_scene/scripts/print_scene_contents.py
```

Load the scene only, without grasp planning or robot execution:

```bash
bash 09_portable_demo_scene/scripts/run_demo_scene.sh
```

Use GUI instead of headless:

```bash
bash 09_portable_demo_scene/scripts/run_demo_scene.sh --gui
```

## Scene contents

| index | name | object_code | seg id | pool index | session color | portable USD | copied size |
|---:|---|---|---:|---:|---|---|---:|
| 0 | Cup | `sem-Cup-d8ea3aa39bcb162798910e50f05b8001` | 7 | 6 | blue | `09_portable_demo_scene/objects/object_00_Cup/simulation/object_006_editable.usd` | 338.3 KiB |
| 1 | Flashlight | `sem-Flashlight-5c7bf45b0f847489181be2d6e974dccd` | 10 | 9 | blue | `09_portable_demo_scene/objects/object_01_Flashlight/simulation/object_009_editable.usd` | 685.1 KiB |
| 2 | Bear | `sem-Bear-1629215db795111ba649bd1425725662` | 15 | 14 | red | `09_portable_demo_scene/objects/object_02_Bear/simulation/object_014_editable.usd` | 1217.4 KiB |
| 3 | Camera | `sem-Camera-1221201ed6aac041745b4b48c30a506e` | 22 | 21 | red | `09_portable_demo_scene/objects/object_03_Camera/simulation/object_021_editable.usd` | 467.7 KiB |
| 4 | Candle | `sem-Candle-1be58678b919b12bc5fe7f65b41f3b19` | 23 | 22 | blue | `09_portable_demo_scene/objects/object_04_Candle/simulation/object_022_editable.usd` | 254.6 KiB |
| 5 | Pencil | `sem-Pencil-370867e6cf6f4ef8b5f19880cadbe491` | 53 | 52 | red | `09_portable_demo_scene/objects/object_05_Pencil/simulation/object_052_editable.usd` | 128.0 KiB |

## Optional closed-loop commands using this scene

When prompted for `Scene folder >`, use:

```text
/home/lin/Projects/DexGraspNet2_Wuji2/09_portable_demo_scene/scene
```

Semantic grasp + Route A:

```bash
./run_closed_loop.sh --task semantic-grasp --motion-route legacy --sim-execute --no-planner-collision-check
```

Semantic grasp + Route B:

```bash
./run_closed_loop.sh --task semantic-grasp --motion-route curobo --sim-execute
```

Color-sort + Route A:

```bash
./run_closed_loop.sh --task color-sort --motion-route legacy --color-seed 42 --sim-execute --no-planner-collision-check
```

Color-sort + Route B:

```bash
./run_closed_loop.sh --task color-sort --motion-route curobo --color-seed 42 --sim-execute
```

## Contracts

- Object poses are unchanged from the source scene manifest.
- `simulation_usd` paths are explicit repo-relative portable paths.
- Source mesh, centered combined mesh, URDF, and COACD pieces are copied per object.
- No runtime object asset dependency on `02_training_dataset/data/` is allowed.
- No `/home/lin/` paths are allowed inside the portable bundle.
