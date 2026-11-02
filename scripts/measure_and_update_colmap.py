#!/usr/bin/env python3
"""
Measure COLMAP triangulation duration using GIFStream's official flags for all 6 N3DV scenes (5 GOPs each),
update local timing records, and sync adjusted Init_Time and Total_Recon_Time to Google Sheets.
"""

import os
import sys
import json
import time
import shutil
import struct
from pathlib import Path

# Add dataset_process to sys.path
sys.path.append("/home/tsaichenghan/GIFStream/dataset_process")
from n3d_video_process import convertdynerftocolmapdb, getcolmapsinglen3d

SCENES = ["sear_steak", "coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak"]
DATASETS_SRC = "/media/tsaichenghan/2TBSSD/datasets"
OUTPUT_DIR = Path("/media/tsaichenghan/2TBSSD/GIFStream/output")
TEMP_COLMAP_ROOT = Path("/media/tsaichenghan/2TBSSD/GIFStream/temp_colmap")

def measure_gop_colmap(scene: str, frame: int) -> float:
    """Runs COLMAP for a specific GOP start frame and returns execution duration in seconds."""
    scene_dir = TEMP_COLMAP_ROOT / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy poses_bounds.npy
    src_poses = os.path.join(DATASETS_SRC, scene, "poses_bounds.npy")
    dst_poses = scene_dir / "poses_bounds.npy"
    if not dst_poses.exists():
        shutil.copy(src_poses, dst_poses)
        
    colmap_dir = scene_dir / f"colmap_{frame}"
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir, ignore_errors=True)
    input_dir = colmap_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy camera images mapped to contiguous cam00.png .. cam{N-1}.png matching n3d_video_process.py
    cams = sorted([d for d in os.listdir(os.path.join(DATASETS_SRC, scene)) if d.startswith("cam") and os.path.isdir(os.path.join(DATASETS_SRC, scene, d))])
    for ind, cam in enumerate(cams):
        src_img = os.path.join(DATASETS_SRC, scene, cam, f"{frame}.png")
        dst_img = input_dir / f"cam{ind:02d}.png"
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            
    # Measure COLMAP execution time
    t0 = time.perf_counter()
    convertdynerftocolmapdb(str(scene_dir), offset=frame)
    getcolmapsinglen3d(str(scene_dir), offset=frame)
    t1 = time.perf_counter()
    duration = t1 - t0
    
    # Read generated point count
    pts_file = colmap_dir / "sparse" / "0" / "points3D.bin"
    n_pts = 0
    if pts_file.exists():
        with open(pts_file, "rb") as f:
            n_pts = struct.unpack("<Q", f.read(8))[0]
            
    print(f"  [{scene} Frame {frame}] COLMAP complete in {duration:.2f}s (Points: {n_pts})")
    
    # Cleanup heavy temp files for this frame
    shutil.rmtree(colmap_dir, ignore_errors=True)
    return duration

def main():
    TEMP_COLMAP_ROOT.mkdir(parents=True, exist_ok=True)
    
    all_timings = {}
    print("==========================================================")
    print(" Measuring Official GIFStream COLMAP Triangulation Times")
    print("==========================================================")
    
    for scene in SCENES:
        print(f"\nProcessing Scene: {scene}")
        scene_out = OUTPUT_DIR / scene
        timings_file = scene_out / "colmap_timings.json"
        if timings_file.exists():
            print(f"  [SKIP] Scene {scene} already measured in {timings_file}. Reusing measured timings.")
            with open(timings_file, "r") as f:
                scene_timings = {int(k): v for k, v in json.load(f).items()}
        else:
            scene_timings = {}
            for gop_id in range(5):
                start_frame = gop_id * 60
                dur = measure_gop_colmap(scene, start_frame)
                scene_timings[gop_id] = dur
            
        all_timings[scene] = scene_timings
        total_scene_colmap = sum(scene_timings.values())
        print(f"--> Scene {scene} total COLMAP time for 5 GOPs: {total_scene_colmap:.2f}s (avg {total_scene_colmap/5:.2f}s/GOP)")
        
        # Save timings to output folder
        if scene_out.exists():
            with open(timings_file, "w") as f:
                json.dump(scene_timings, f, indent=2)
                
            # Update each GOP's stats JSON
            for gop_id, colmap_dur in scene_timings.items():
                r0_dir = scene_out / f"GOP_{gop_id}" / "r0"
                stats_file = r0_dir / "stats" / "compress_step29999.json"
                if stats_file.exists():
                    with open(stats_file, "r") as f:
                        data = json.load(f)
                    
                    train_recon = float(data.get("train_recon_time_s", data.get("ellipse_time", 0.0)))
                    data["init_time_s"] = colmap_dur
                    data["total_recon_time_s"] = colmap_dur + train_recon
                    
                    with open(stats_file, "w") as f:
                        json.dump(data, f, indent=2)
                        
    # Cleanup temp colmap directory
    shutil.rmtree(TEMP_COLMAP_ROOT, ignore_errors=True)
    print("\n==========================================================")
    print(" All COLMAP measurements completed and stats updated!")
    print(" Triggering Google Sheets Synchronization...")
    print("==========================================================")
    
    # Run sync script with update_existing=True to refresh Init_Time in sheets
    sys.path.append("/home/tsaichenghan/GIFStream/scripts")
    import sync_gifstream_to_sheets
    sync_gifstream_to_sheets.sync_to_sheets(update_existing=True)

if __name__ == "__main__":
    main()
