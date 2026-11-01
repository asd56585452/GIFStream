#!/usr/bin/env python3
"""
Sync GIFStream benchmark results to Google Sheets (Scheme A 4-Tier Relational Database).
Spreadsheet ID: 10_KMZqFWPzhv4VZELX8sgknl2G2rxwL-uNwZrrFvWsE
"""

import os
import sys
import json
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict

SPREADSHEET_ID = "10_KMZqFWPzhv4VZELX8sgknl2G2rxwL-uNwZrrFvWsE"
METHOD_NAME = "GIFStream (CVPR 2025)"
DATASET_NAME = "N3DV"

OUTPUT_DIR = Path("/media/tsaichenghan/2TBSSD/GIFStream/output")

def run_gsheets_cmd(cmd_list):
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.local/bin:{Path.home()}/.gemini/config/plugins/gsheets/skills/gsheets/scripts:" + env.get("PATH", "")
    res = subprocess.run(cmd_list, env=env, capture_output=True, text=True)
    return res

def check_auth():
    res = run_gsheets_cmd(["gsheets", "auth", "status"])
    combined = (res.stdout or "") + (res.stderr or "")
    return (res.returncode == 0 and "Logged in" in combined)

def get_existing_keys(tab_name, key_cols):
    """Retrieve existing keys (tuple of columns) for deduplication."""
    res = run_gsheets_cmd(["gsheets", "readonly", "read", SPREADSHEET_ID, f"'{tab_name}'!A:G", "--json"])
    existing = set()
    if res.returncode == 0:
        try:
            data = json.loads(res.stdout)
            for row in data[1:]:  # skip header
                if len(row) >= max(key_cols) + 1:
                    key = tuple(row[col].strip() for col in key_cols)
                    existing.add(key)
        except Exception:
            pass
    return existing

def collect_gifstream_results():
    """
    Parses OUTPUT_DIR structure:
    OUTPUT_DIR / <scene> / GOP_<gop_id> / r0 / stats / compress_step29999.json
    OUTPUT_DIR / <scene> / GOP_<gop_id> / r0 / stats / compress_per_frame_step29999.json
    OUTPUT_DIR / <scene> / GOP_<gop_id> / r0 / compression/ -> compression.zip
    """
    scenes_data = defaultdict(dict)
    
    if not OUTPUT_DIR.exists():
        return scenes_data
        
    for scene_dir in sorted(OUTPUT_DIR.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("."):
            continue
        scene = scene_dir.name
        
        for gop_dir in sorted(scene_dir.glob("GOP_*")):
            if not gop_dir.is_dir():
                continue
            try:
                gop_id = int(gop_dir.name.split("_")[1])
            except ValueError:
                continue
                
            r0_dir = gop_dir / "r0"
            if not r0_dir.is_dir():
                continue
                
            stats_json = r0_dir / "stats" / "compress_step29999.json"
            if not stats_json.exists():
                # Check for train_step29999 if compress_step29999 isn't ready yet
                continue
                
            with open(stats_json) as f:
                stats = json.load(f)
                
            per_frame_json = r0_dir / "stats" / "compress_per_frame_step29999.json"
            per_frame_data = []
            if per_frame_json.exists():
                with open(per_frame_json) as f:
                    per_frame_data = json.load(f)
                    
            # Compute zip size of compression directory
            comp_dir = r0_dir / "compression"
            zip_file = r0_dir / "compression.zip"
            if comp_dir.exists() and comp_dir.is_dir():
                if zip_file.exists():
                    zip_file.unlink()
                with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(comp_dir):
                        for file in files:
                            fp = os.path.join(root, file)
                            zipf.write(fp, os.path.relpath(fp, comp_dir))
                            
            size_mb = (zip_file.stat().st_size / (1024 * 1024)) if zip_file.exists() else 0.0
            
            scenes_data[scene][gop_id] = {
                "stats": stats,
                "per_frame": per_frame_data,
                "size_mb": size_mb,
            }
            
    return scenes_data

def sync_to_sheets():
    if not check_auth():
        print("[SYNC] gsheets is not authenticated.")
        return False
        
    scenes_data = collect_gifstream_results()
    if not scenes_data:
        print("[SYNC] No completed GIFStream results found yet.")
        return True
        
    print(f"[SYNC] Found results for {len(scenes_data)} scenes: {list(scenes_data.keys())}")
    
    # 1. Sync Tier 4: 04_Frame_Quality
    existing_t4 = get_existing_keys("04_Frame_Quality", key_cols=[0, 1, 2, 4]) # Method, Dataset, Scene, Global_Frame_ID
    t4_rows = []
    for scene, gops in scenes_data.items():
        for gop_id, gop_info in gops.items():
            for item in gop_info["per_frame"]:
                global_frame_id = item["frame_idx"]
                rel_frame_id = global_frame_id - (gop_id * 60)
                view_name = item["view_name"]
                key = (METHOD_NAME, DATASET_NAME, scene, str(global_frame_id))
                if key not in existing_t4:
                    t4_rows.append([
                        METHOD_NAME,
                        DATASET_NAME,
                        scene,
                        str(gop_id),
                        str(global_frame_id),
                        str(rel_frame_id),
                        view_name,
                        f"{item['psnr']:.4f}",
                        f"{item['ssim']:.4f}",
                        f"{item['lpips']:.4f}"
                    ])
                    existing_t4.add(key)
                    
    if t4_rows:
        print(f"[SYNC] Appending {len(t4_rows)} rows to 04_Frame_Quality...")
        # Write in chunks of 60 rows (1 GOP per chunk) to avoid CLI argument length limit
        chunk_size = 60
        for i in range(0, len(t4_rows), chunk_size):
            chunk = t4_rows[i:i+chunk_size]
            batch_str = "|".join([",".join(row) for row in chunk])
            res = run_gsheets_cmd(["gsheets", "mutate", "append", SPREADSHEET_ID, "'04_Frame_Quality'", batch_str])
            if res.returncode != 0:
                print(f"[SYNC ERROR] Failed appending to 04_Frame_Quality: {res.stderr}")
    else:
        print("[SYNC] 04_Frame_Quality is up to date.")
        
    # 2. Sync Tier 3: 03_GoF_Details
    existing_t3 = get_existing_keys("03_GoF_Details", key_cols=[0, 1, 2, 3]) # Method, Dataset, Scene, GoF_ID
    t3_rows = []
    for scene, gops in scenes_data.items():
        for gop_id, gop_info in gops.items():
            key = (METHOD_NAME, DATASET_NAME, scene, str(gop_id))
            if key not in existing_t3:
                s = gop_info["stats"]
                size_mb = gop_info["size_mb"]
                frame_count = 60
                init_time = float(s.get("init_time_s", 0.0))
                train_recon_time = float(s.get("train_recon_time_s", s.get("ellipse_time", 0.0)))
                total_recon_time = float(s.get("total_recon_time_s", init_time + train_recon_time))
                peak_vram = float(s.get("peak_vram_gb", s.get("mem", 0.0)))
                render_fps = (1.0 / s.get("ellipse_time", 0.003)) if s.get("ellipse_time", 0.0) > 0 else 300.0
                
                t3_rows.append([
                    METHOD_NAME,
                    DATASET_NAME,
                    scene,
                    str(gop_id),
                    str(gop_id * 60),
                    str((gop_id + 1) * 60 - 1),
                    str(frame_count),
                    f"{init_time:.2f}",
                    f"{train_recon_time:.2f}",
                    f"{total_recon_time:.2f}",
                    f"{init_time / frame_count:.4f}",
                    f"{train_recon_time / frame_count:.4f}",
                    f"{total_recon_time / frame_count:.4f}",
                    f"{peak_vram:.2f}",
                    f"{float(s.get('psnr', 0.0)):.2f}",
                    f"{float(s.get('ssim', 0.0)):.4f}",
                    f"{float(s.get('lpips', 0.0)):.4f}",
                    f"{size_mb:.4f}",
                    f"{size_mb / frame_count:.6f}",
                    f"{render_fps:.1f}"
                ])
                existing_t3.add(key)
                
    if t3_rows:
        print(f"[SYNC] Appending {len(t3_rows)} rows to 03_GoF_Details...")
        batch_str = "|".join([",".join(row) for row in t3_rows])
        res = run_gsheets_cmd(["gsheets", "mutate", "append", SPREADSHEET_ID, "'03_GoF_Details'", batch_str])
        if res.returncode != 0:
            print(f"[SYNC ERROR] Failed appending to 03_GoF_Details: {res.stderr}")
    else:
        print("[SYNC] 03_GoF_Details is up to date.")
        
    # 3. Sync Tier 2: 02_Scene_Summary
    # Only for scenes where all 5 GOPs (300 frames) are complete
    existing_t2 = get_existing_keys("02_Scene_Summary", key_cols=[0, 1, 2]) # Method, Dataset, Scene
    t2_rows = []
    completed_scene_aggregates = []
    
    for scene, gops in scenes_data.items():
        if len(gops) < 5:
            print(f"[SYNC] Scene {scene} has {len(gops)}/5 GOPs complete. Waiting for full 300 frames before summary.")
            continue
            
        key = (METHOD_NAME, DATASET_NAME, scene)
        
        # Aggregate 5 GOPs
        total_init_time = sum(float(g["stats"].get("init_time_s", 0.0)) for g in gops.values())
        total_train_recon_time = sum(float(g["stats"].get("train_recon_time_s", g["stats"].get("ellipse_time", 0.0))) for g in gops.values())
        total_recon_time = total_init_time + total_train_recon_time
        time_per_frame = total_recon_time / 300.0
        
        mean_psnr = sum(float(g["stats"].get("psnr", 0.0)) for g in gops.values()) / 5.0
        mean_ssim = sum(float(g["stats"].get("ssim", 0.0)) for g in gops.values()) / 5.0
        mean_lpips = sum(float(g["stats"].get("lpips", 0.0)) for g in gops.values()) / 5.0
        
        total_model_size_mb = sum(g["size_mb"] for g in gops.values())
        size_per_frame_mb = total_model_size_mb / 300.0
        
        mean_render_fps = sum((1.0 / g["stats"].get("ellipse_time", 0.003)) for g in gops.values() if g["stats"].get("ellipse_time", 0.0) > 0) / 5.0
        max_peak_vram = max(float(g["stats"].get("peak_vram_gb", g["stats"].get("mem", 0.0))) for g in gops.values())
        
        scene_agg = {
            "scene": scene,
            "total_init_time": total_init_time,
            "total_train_recon_time": total_train_recon_time,
            "total_recon_time": total_recon_time,
            "time_per_frame": time_per_frame,
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "lpips": mean_lpips,
            "size_mb": total_model_size_mb,
            "size_per_frame_mb": size_per_frame_mb,
            "render_fps": mean_render_fps,
            "peak_vram": max_peak_vram
        }
        completed_scene_aggregates.append(scene_agg)
        
        if key not in existing_t2:
            t2_rows.append([
                METHOD_NAME,
                DATASET_NAME,
                scene,
                "300",
                f"{total_init_time:.2f}",
                f"{total_train_recon_time:.2f}",
                f"{total_recon_time:.2f}",
                f"{time_per_frame:.4f}",
                f"{mean_psnr:.2f}",
                f"{mean_ssim:.4f}",
                f"{mean_lpips:.4f}",
                f"{total_model_size_mb:.2f}",
                f"{size_per_frame_mb:.4f}",
                f"{mean_render_fps:.1f}",
                f"{max_peak_vram:.2f}",
                datetime.now().strftime("%Y-%m-%d")
            ])
            existing_t2.add(key)
            
    if t2_rows:
        print(f"[SYNC] Appending {len(t2_rows)} scenes to 02_Scene_Summary...")
        batch_str = "|".join([",".join(row) for row in t2_rows])
        res = run_gsheets_cmd(["gsheets", "mutate", "append", SPREADSHEET_ID, "'02_Scene_Summary'", batch_str])
        if res.returncode != 0:
            print(f"[SYNC ERROR] Failed appending to 02_Scene_Summary: {res.stderr}")
    else:
        print("[SYNC] 02_Scene_Summary is up to date.")
        
    # 4. Sync Tier 1: 01_Leaderboard (Row 4: GIFStream (CVPR 2025))
    if completed_scene_aggregates:
        n = len(completed_scene_aggregates)
        overall_time_per_frame = sum(a["time_per_frame"] for a in completed_scene_aggregates) / n
        overall_psnr = sum(a["psnr"] for a in completed_scene_aggregates) / n
        overall_ssim = sum(a["ssim"] for a in completed_scene_aggregates) / n
        overall_lpips = sum(a["lpips"] for a in completed_scene_aggregates) / n
        overall_size_per_frame = sum(a["size_per_frame_mb"] for a in completed_scene_aggregates) / n
        overall_render_fps = sum(a["render_fps"] for a in completed_scene_aggregates) / n
        overall_peak_vram = max(a["peak_vram"] for a in completed_scene_aggregates)
        
        print(f"[SYNC] Updating 01_Leaderboard row for {METHOD_NAME} (based on {n} completed scenes)...")
        # Format: Recon Time (s) ↓, PSNR (dB) ↑, SSIM ↑, LPIPS ↓, File Size (MB) ↓, Render FPS ↑, GPU VRAM (GB) ↓
        leaderboard_values = [
            f"{overall_time_per_frame:.2f}",
            f"{overall_psnr:.2f}",
            f"{overall_ssim:.4f}",
            f"{overall_lpips:.4f}",
            f"{overall_size_per_frame:.4f}",
            f"{int(overall_render_fps)}",
            f"{overall_peak_vram:.2f}"
        ]
        val_csv = ",".join(leaderboard_values)
        res = run_gsheets_cmd(["gsheets", "mutate", "write", SPREADSHEET_ID, "'01_Leaderboard'!B4:H4", val_csv])
        if res.returncode == 0:
            print(f"[SYNC] Successfully updated 01_Leaderboard: {val_csv}")
        else:
            print(f"[SYNC ERROR] Failed updating 01_Leaderboard: {res.stderr}")
            
    return True

if __name__ == "__main__":
    sync_to_sheets()

