#!/usr/bin/env python3
"""Download Allen Human Brain Atlas meshes and pre-compute the voxel label grid.

This script MUST be run once before using mindVisualizer. It:
  1. Downloads the Allen Human Brain Atlas (500um) via brainglobe
  2. Copies OBJ mesh files to data/meshes/
  3. Copies structures.json (region name mappings) to data/
  4. Pre-computes a 100^3 voxel label grid for fast point-in-region queries
     and saves it to data/label_grid_cache.npz

Usage:
  pip install brainglobe-atlasapi
  python setup_brain_data.py
"""

import shutil
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MESH_DIR = DATA_DIR / "meshes"
ALIGNMENT_FILE = DATA_DIR / "brain_alignment.json"
GRID_CACHE = DATA_DIR / "label_grid_cache.npz"


def main():
    # ---- Step 1: Download atlas ----
    print("[setup] Downloading Allen Human Brain Atlas (500um)...")
    print("  (This is a one-time download, ~269 MB)")
    from brainglobe_atlasapi import BrainGlobeAtlas
    atlas = BrainGlobeAtlas("allen_human_500um")

    atlas_dir = Path.home() / ".brainglobe" / "allen_human_500um_v1.0"
    src_meshes = atlas_dir / "meshes"

    if not src_meshes.exists():
        print(f"[setup] ERROR: Atlas meshes not found at {src_meshes}")
        return

    # ---- Step 2: Copy meshes to both meshes/ and meshes_obj/ ----
    MESH_OBJ_DIR = DATA_DIR / "meshes_obj"
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    MESH_OBJ_DIR.mkdir(parents=True, exist_ok=True)
    obj_files = list(src_meshes.glob("*.obj"))
    print(f"[setup] Copying {len(obj_files)} OBJ meshes...")

    copied = 0
    for obj in obj_files:
        for target_dir in (MESH_DIR, MESH_OBJ_DIR):
            dst = target_dir / obj.name
            if not dst.exists() or dst.stat().st_size != obj.stat().st_size:
                shutil.copy2(obj, dst)
                copied += 1
    print(f"[setup] Copied meshes ({len(obj_files)} files to meshes/ and meshes_obj/)")

    # ---- Step 3: Copy structures.json ----
    structs_src = atlas_dir / "structures.json"
    structs_dst = DATA_DIR / "structures.json"
    if structs_src.exists():
        shutil.copy2(structs_src, structs_dst)
        print(f"[setup] Copied structures.json")
    else:
        print(f"[setup] WARNING: structures.json not found at {structs_src}")

    # ---- Step 4: Build and cache label grid ----
    print(f"\n[setup] Building voxel label grid (this takes ~30-60 seconds)...")
    import vtk
    from src.mesh_overlay import FlowMeshOverlay

    ren = vtk.vtkRenderer()
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(ren)

    overlay = FlowMeshOverlay(ren=ren, win=win,
                               mesh_dir=MESH_DIR,
                               alignment_file=ALIGNMENT_FILE)

    n_regions = len(overlay.get_all_region_keys())
    print(f"[setup] Loaded {n_regions} region meshes")

    overlay.build_label_grid()
    overlay.save_label_grid(GRID_CACHE)

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"  Setup complete!")
    print(f"  Meshes:       {MESH_DIR}/ ({len(obj_files)} files)")
    print(f"  Structures:   {structs_dst}")
    print(f"  Label grid:   {GRID_CACHE}")
    print(f"  Regions:      {n_regions}")
    print(f"{'='*50}")
    print(f"\n  You can now run:")
    print(f"    python -m src.main")
    print(f"    python examples/rdcim_propagation.py")


if __name__ == "__main__":
    main()
