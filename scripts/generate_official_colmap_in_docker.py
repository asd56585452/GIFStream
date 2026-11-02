#!/usr/bin/env python3
"""
Generate official COLMAP triangulations for N3DV scenes strictly inside Docker.
- Reads multi-view images and poses from /datasets (mounted read-only :ro).
- Generates official COLMAP 3D reconstructions with authors' exact SIFT & triangulation parameters.
- Writes to /media/tsaichenghan/2TBSSD/GIFStream/datasets/<scene>/colmap_{frame}.
- Records duration to colmap_{frame}/init_time.txt.
"""

import os
import sys
import time
import json
import shutil
import struct
import argparse
from pathlib import Path

# Add dataset_process
sys.path.append("/workspace/dataset_process")
from n3d_video_process import convertdynerftocolmapdb, getcolmapsinglen3d

SCENES = ["sear_steak", "coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak"]
DATASETS_SRC = Path("/datasets")
GIFSTREAM_DATASETS = Path("/media/tsaichenghan/2TBSSD/GIFStream/datasets")

def generate_scene_gop_colmap(scene: str, frame: int, force: bool = False) -> float:
    scene_src = DATASETS_SRC / scene
    scene_tgt = GIFSTREAM_DATASETS / scene
    scene_tgt.mkdir(parents=True, exist_ok=True)
    
    # 1. Ensure poses_bounds.npy exists in target
    src_poses = scene_src / "poses_bounds.npy"
    tgt_poses = scene_tgt / "poses_bounds.npy"
    if not tgt_poses.exists() or tgt_poses.is_symlink():
        if tgt_poses.is_symlink():
            tgt_poses.unlink()
        shutil.copy(src_poses, tgt_poses)
        
    colmap_dir = scene_tgt / f"colmap_{frame}"
    points_bin = colmap_dir / "sparse" / "0" / "points3D.bin"
    init_time_file = colmap_dir / "init_time.txt"
    
    # Check if already generated as real directory with points3D.bin and init_time.txt
    if not force and colmap_dir.exists() and not colmap_dir.is_symlink() and points_bin.exists() and init_time_file.exists():
        with open(points_bin, "rb") as f:
            n_pts = struct.unpack("<Q", f.read(8))[0]
        with open(init_time_file, "r") as f:
            dur = float(f.read().strip())
        print(f"[{scene} GOP frame {frame}] [SKIP] Already exists with {n_pts} points ({dur:.2f}s).")
        return dur
        
    if colmap_dir.is_symlink():
        print(f"[{scene} GOP frame {frame}] Removing old symlink to precomputed colmap.")
        colmap_dir.unlink()
    elif colmap_dir.exists():
        shutil.rmtree(colmap_dir)
        
    colmap_dir.mkdir(parents=True, exist_ok=True)
    input_dir = colmap_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy camera images mapped to contiguous cam00.png .. cam{N-1}.png matching poses_bounds.npy
    cams = sorted([d for d in os.listdir(scene_src) if d.startswith("cam") and (scene_src / d).is_dir()])
    for ind, cam in enumerate(cams):
        src_img = scene_src / cam / f"{frame}.png"
        dst_img = input_dir / f"cam{ind:02d}.png"
        if src_img.exists():
            shutil.copy(src_img, dst_img)
            
    print(f"[{scene} GOP frame {frame}] Starting official COLMAP triangulation ({len(cams)} cameras)...")
    t0 = time.perf_counter()
    convertdynerftocolmapdb(str(scene_tgt), offset=frame)
    getcolmapsinglen3d(str(scene_tgt), offset=frame)
    t1 = time.perf_counter()
    duration = t1 - t0
    
    n_pts = 0
    if points_bin.exists():
        with open(points_bin, "rb") as f:
            n_pts = struct.unpack("<Q", f.read(8))[0]
            
    # Save init_time.txt
    with open(init_time_file, "w") as f:
        f.write(f"{duration:.4f}\n")
        
    print(f"[{scene} GOP frame {frame}] Official COLMAP complete in {duration:.2f}s ({n_pts} points).")
    
    # Clean up intermediate databases and manuals to save space
    input_db = colmap_dir / "input.db"
    manual_dir = colmap_dir / "manual"
    distorted_dir = colmap_dir / "distorted"
    tmp_dir = colmap_dir / "tmp"
    if input_db.exists(): input_db.unlink()
    if manual_dir.exists(): shutil.rmtree(manual_dir, ignore_errors=True)
    if distorted_dir.exists(): shutil.rmtree(distorted_dir, ignore_errors=True)
    if tmp_dir.exists(): shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return duration

def main():
    parser = argparse.ArgumentParser(description="Generate official COLMAP in Docker")
    parser.add_argument("--scenes", nargs="+", default=SCENES, help="Target scene names")
    parser.add_argument("--gops", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="GOP IDs (0 to 4)")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    
    print("==========================================================")
    print(" Running Official GIFStream COLMAP Generation inside Docker")
    print(f" Scenes: {args.scenes}")
    print(f" GOP IDs: {args.gops}")
    print("==========================================================")
    
    all_timings = {}
    for scene in args.scenes:
        print(f"\n>>> Scene: {scene}")
        scene_timings = {}
        for gop_id in args.gops:
            start_frame = gop_id * 60
            dur = generate_scene_gop_colmap(scene, start_frame, force=args.force)
            scene_timings[gop_id] = dur
            
        all_timings[scene] = scene_timings
        timings_file = GIFSTREAM_DATASETS / scene / "colmap_timings.json"
        with open(timings_file, "w") as f:
            json.dump(scene_timings, f, indent=2)
            
    print("\n==========================================================")
    print(" All requested official COLMAP triangulations complete!")
    print("==========================================================")

if __name__ == "__main__":
    main()
