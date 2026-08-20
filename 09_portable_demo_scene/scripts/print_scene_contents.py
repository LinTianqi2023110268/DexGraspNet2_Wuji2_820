#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
meta = json.load(open(ROOT/'09_portable_demo_scene/scene/source_scene_metadata.json', encoding='utf-8'))
print('PORTABLE DEMO SCENE CONTENTS')
print('============================')
print('source session:', meta['source_session'])
print('source scene  :', meta['source_scene_folder'])
print('objects       :', meta['object_count'])
for o in meta['objects']:
    print(f"{o['index']:02d} {o['name']:<12} seg={o['segmentation_id']:<3} pool={o['object_pool_index']:<3} color={o.get('sort_color_in_source_session')} code={o['object_code']}")
    print('    usd :', o['portable_simulation_usd'])
    print('    mesh:', o['portable_source_mesh'])
