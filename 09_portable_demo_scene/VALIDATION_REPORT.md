# Portable Demo Scene Validation Report

Created UTC: 2026-08-20T07:20:47.098188+00:00

Source session: `20260820_124401`
Source scene manifest: `/home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/data/scene_datasets/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1/scenes/scene_0020/scene_manifest.json`
Portable manifest: `09_portable_demo_scene/scene/portable_scene_manifest.json`

## Offline validation

Run:

```bash
python 09_portable_demo_scene/scripts/validate_portable_scene.py
```

## Isaac load-only validation

Run:

```bash
bash 09_portable_demo_scene/scripts/run_demo_scene.sh
```

Status will be updated after validation.
