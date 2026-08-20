#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from xml.etree import ElementTree as ET

LOCAL_FORBIDDEN = [b'/home/lin/']
DATA_FORBIDDEN = [b'02_training_dataset/data/']
TEXT_EXTS = {'.json','.urdf','.obj','.mtl','.txt','.config','.usda','.usd','.wrl'}
USD_EXTS = {'.usd','.usda','.usdc'}
IMG_EXTS = {'.png','.jpg','.jpeg','.bmp','.tga','.exr'}

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]

def is_binary(p: Path) -> bool:
    data=p.read_bytes()[:4096]
    return b'\0' in data

def scan_forbidden(paths):
    hits=[]
    for p in paths:
        if not p.is_file(): continue
        data=p.read_bytes()
        for token in LOCAL_FORBIDDEN + DATA_FORBIDDEN:
            if token in data:
                hits.append((str(p), token.decode('utf-8', 'ignore')))
    return hits

def parse_obj_deps(obj: Path):
    deps=[]
    try:
        for line in obj.read_text(encoding='utf-8', errors='ignore').splitlines():
            s=line.strip()
            if s.startswith('mtllib '):
                for item in s.split()[1:]: deps.append(obj.parent / item)
    except Exception:
        pass
    return deps

def parse_mtl_deps(mtl: Path):
    deps=[]
    keys=('map_Kd','map_Ka','map_Ks','map_Bump','bump','disp','decal','map_d','map_Ns')
    try:
        for line in mtl.read_text(encoding='utf-8', errors='ignore').splitlines():
            parts=line.strip().split()
            if parts and parts[0] in keys and len(parts)>1:
                deps.append(mtl.parent / parts[-1])
    except Exception:
        pass
    return deps

def parse_urdf_deps(urdf: Path):
    deps=[]
    try:
        tree=ET.parse(urdf)
        for mesh in tree.findall('.//mesh'):
            fn=mesh.attrib.get('filename')
            if fn:
                s=fn.replace('package://','')
                deps.append((urdf.parent / s).resolve() if not Path(s).is_absolute() else Path(s))
    except Exception:
        text=urdf.read_text(encoding='utf-8', errors='ignore')
        for m in re.finditer(r'filename=["\']([^"\']+)["\']', text):
            s=m.group(1).replace('package://','')
            deps.append((urdf.parent / s).resolve() if not Path(s).is_absolute() else Path(s))
    return deps

def parse_usd_text_deps(usd: Path):
    deps=[]
    if is_binary(usd):
        return deps
    text=usd.read_text(encoding='utf-8', errors='ignore')
    for pat in [r'@([^@]+)@', r'asset\s*=\s*["\']([^"\']+)["\']']:
        for m in re.finditer(pat, text):
            s=m.group(1)
            if any(s.lower().endswith(ext) for ext in list(USD_EXTS)+list(IMG_EXTS)+['.obj','.mtl']):
                deps.append((usd.parent / s).resolve() if not Path(s).is_absolute() else Path(s))
    return deps

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle-root', type=Path, default=repo_root_from_script()/'09_portable_demo_scene')
    args=ap.parse_args()
    bundle=args.bundle_root.resolve()
    repo=bundle.parent
    manifest=bundle/'scene/portable_scene_manifest.json'
    metadata_path=bundle/'scene/source_scene_metadata.json'
    checksums=bundle/'CHECKSUMS.sha256'
    errors=[]
    warnings=[]
    if not manifest.is_file(): errors.append(f'missing manifest: {manifest}')
    if not metadata_path.is_file(): errors.append(f'missing metadata: {metadata_path}')
    if not checksums.is_file(): errors.append(f'missing checksums: {checksums}')
    if errors:
        print('\n'.join(errors)); return 2
    scene=load_json(manifest)
    meta=load_json(metadata_path)
    objs=scene.get('objects', [])
    if len(objs)!=6: errors.append(f'expected 6 objects, found {len(objs)}')
    missing=[]
    for i,o in enumerate(objs):
        for key in ['simulation_usd']:
            p=repo/o[key]
            if not p.is_file(): missing.append(str(p))
        a=o.get('asset',{})
        for key in ['source_obj','centered_combined_obj','urdf']:
            if key in a:
                p=repo/a[key]
                if not p.is_file(): missing.append(str(p))
    # Dependency checks.
    missing_deps=[]
    for p in bundle.rglob('*'):
        if not p.is_file(): continue
        suffix=p.suffix.lower()
        deps=[]
        if suffix=='.obj': deps += parse_obj_deps(p)
        elif suffix=='.mtl': deps += parse_mtl_deps(p)
        elif suffix=='.urdf': deps += parse_urdf_deps(p)
        elif suffix in USD_EXTS: deps += parse_usd_text_deps(p)
        for dep in deps:
            if not dep.exists(): missing_deps.append(f'{p.relative_to(bundle)} -> {dep}')
    # Runtime assets must be portable.  Lineage metadata intentionally records
    # original /home/lin source paths and is therefore excluded from this scan.
    runtime_scan_paths = [
        bundle/'scene/scene_manifest.json',
        bundle/'scene/portable_scene_manifest.json',
    ]
    runtime_scan_paths += [p for p in (bundle/'objects').rglob('*') if p.is_file()]
    forbidden=scan_forbidden(runtime_scan_paths)
    if missing: errors.append('missing assets: '+json.dumps(missing, indent=2))
    if missing_deps: errors.append('missing recursive dependencies: '+json.dumps(missing_deps, indent=2))
    if forbidden: errors.append('forbidden local/dataset paths: '+json.dumps(forbidden[:50], indent=2))
    # Checksums.
    checksum_errors=[]
    expected={}
    for line in checksums.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        h, path = line.split(maxsplit=1)
        expected[path.strip()]=h.strip()
    for rel,h in expected.items():
        p=repo/rel
        if not p.is_file(): checksum_errors.append(f'missing checksum file {rel}')
        elif sha256(p)!=h: checksum_errors.append(f'sha mismatch {rel}')
    if checksum_errors: errors.append('checksum errors: '+json.dumps(checksum_errors[:30], indent=2))
    print('PORTABLE SCENE VALIDATION')
    print('=========================')
    print(f'Scene: {manifest.relative_to(repo)}')
    print(f'Objects: {len(objs)}/6 ' + ('PASS' if len(objs)==6 else 'FAIL'))
    print(f'Absolute local paths: {sum(1 for _,tok in forbidden if tok.startswith("/home"))}')
    print(f'External dataset dependencies: {sum(1 for _,tok in forbidden if tok.startswith("02_training_dataset/data"))}')
    print(f'Missing assets: {len(missing)}')
    print(f'Missing textures/deps: {len(missing_deps)}')
    print(f'USD unresolved references: {len([x for x in missing_deps if ".usd" in x.lower()])}')
    print(f'Checksum entries: {len(expected)}')
    print('Portable: ' + ('PASS' if not errors else 'FAIL'))
    if errors:
        print('\nERRORS:')
        for e in errors: print('-', e[:2000])
        return 1
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
