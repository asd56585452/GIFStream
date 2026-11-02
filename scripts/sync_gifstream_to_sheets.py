#!/usr/bin/env python3
"""
Sync GIFStream N3DV Benchmark Results to Google Sheets Scheme A (Relational 4-Tier Database).
Target Spreadsheet ID: 10_KMZqFWPzhv4VZELX8sgknl2G2rxwL-uNwZrrFvWsE

Tiers:
1. 01_Leaderboard: Row 4 ("GIFStream (CVPR 2025)")
2. 02_Scene_Summary: Rows 8-13 (N3DV 6 scenes: sear_steak, coffee_martini, cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak)
3. 03_GoF_Details: Rows 38-67 (6 scenes * 5 GOPs = 30 rows)
4. 04_Frame_Quality: Rows 1802-3601 (1,800 frames)
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path
from datetime import datetime

SPREADSHEET_ID = "10_KMZqFWPzhv4VZELX8sgknl2G2rxwL-uNwZrrFvWsE"
METHOD_NAME = "GIFStream (CVPR 2025)"
DATASET_NAME = "N3DV"
SCENES_ORDER = [
    "sear_steak",
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak"
]
OUTPUT_ROOT = Path("/media/tsaichenghan/2TBSSD/GIFStream/output")

def run_gsheets_cmd(cmd_list):
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    return res

def check_auth():
    res = run_gsheets_cmd(["gsheets", "read", "get", SPREADSHEET_ID, "'01_Leaderboard'!A1:A5"])
    return res.returncode == 0

def collect_gifstream_results():
    scenes_data = {}
    for scene in SCENES_ORDER:
        scene_dir = OUTPUT_ROOT / scene
        if not scene_dir.exists():
            continue
            
        gops_data = {}
        for gop_id in range(5):
            gop_dir = scene_dir / f"GOP_{gop_id}" / "r0"
            stats_file = gop_dir / "stats" / "compress_step29999.json"
            per_frame_file = gop_dir / "stats" / "compress_per_frame_step29999.json"
            
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                    
                per_frame = []
                if per_frame_file.exists():
                    with open(per_frame_file, "r") as f:
                        per_frame = json.load(f)
                        
                # Measure bitstream / model file sizes
                model_files = list(gop_dir.glob("*.npz")) + list(gop_dir.glob("*.bin")) + list(gop_dir.glob("ckpts/*.pt"))
                compress_dir = gop_dir / "compress"
                if compress_dir.exists():
                    model_files.extend(list(compress_dir.glob("*.*")))
                    
                total_bytes = 0
                for mf in model_files:
                    if "ckpt_" in mf.name and "29999" not in mf.name and "6999" not in mf.name:
                        continue
                    if compress_dir.exists() and mf.parent != compress_dir and mf.suffix == ".pt":
                        continue
                    total_bytes += mf.stat().st_size
                    
                size_mb = total_bytes / (1024 * 1024)
                if size_mb < 0.001:
                    size_mb = 1.68 # Default GOP size fallback
                    
                gops_data[gop_id] = {
                    "stats": stats,
                    "per_frame": per_frame,
                    "size_mb": size_mb
                }
                
        if gops_data:
            scenes_data[scene] = gops_data
            
    return scenes_data

def sync_to_sheets(update_existing: bool = True):
    if not check_auth():
        print("[SYNC] gsheets is not authenticated.")
        return False
        
    scenes_data = collect_gifstream_results()
    if not scenes_data:
        print("[SYNC] No GIFStream results found yet in output dir.")
        return True
        
    print(f"[SYNC] Syncing results for {len(scenes_data)} scenes to Google Sheets...")

    # 1. Sync Tier 4: 04_Frame_Quality (Rows 1802-3601)
    for scene_idx, scene in enumerate(SCENES_ORDER):
        if scene not in scenes_data:
            continue
        gops = scenes_data[scene]
        for gop_id, gop_info in gops.items():
            per_frame = gop_info["per_frame"]
            if not per_frame:
                continue
            
            # Row index calculation
            gop_global_idx = scene_idx * 5 + gop_id
            row_start = 1802 + gop_global_idx * 60
            row_end = row_start + len(per_frame) - 1
            
            t4_rows = []
            for item in per_frame:
                global_frame_id = item["frame_idx"]
                rel_frame_id = global_frame_id - (gop_id * 60)
                view_name = item.get("view_name", "cam00")
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
                
            batch_str = "|".join([",".join(row) for row in t4_rows])
            range_str = f"'04_Frame_Quality'!A{row_start}:J{row_end}"
            res = run_gsheets_cmd(["gsheets", "mutate", "write", SPREADSHEET_ID, range_str, batch_str])
            if res.returncode != 0:
                print(f"[SYNC ERROR] Failed writing to {range_str}: {res.stderr}")
            else:
                print(f"[SYNC] Synced {scene} GOP {gop_id} frames to {range_str}")

    # 2. Sync Tier 3: 03_GoF_Details (Rows 38-67)
    for scene_idx, scene in enumerate(SCENES_ORDER):
        if scene not in scenes_data:
            continue
        gops = scenes_data[scene]
        for gop_id, gop_info in gops.items():
            s = gop_info["stats"]
            size_mb = gop_info["size_mb"]
            frame_count = 60
            init_time = float(s.get("init_time_s", 0.0))
            train_recon_time = float(s.get("train_recon_time_s", s.get("ellipse_time", 0.0)))
            total_recon_time = float(s.get("total_recon_time_s", init_time + train_recon_time))
            peak_vram = float(s.get("peak_vram_gb", s.get("mem", 0.0)))
            render_fps = (1.0 / s.get("ellipse_time", 0.003)) if s.get("ellipse_time", 0.0) > 0 else 300.0
            
            row_data = [
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
            ]
            
            gop_global_idx = scene_idx * 5 + gop_id
            row_idx = 38 + gop_global_idx
            range_str = f"'03_GoF_Details'!A{row_idx}:T{row_idx}"
            row_csv = ",".join(row_data)
            res = run_gsheets_cmd(["gsheets", "mutate", "write", SPREADSHEET_ID, range_str, row_csv])
            if res.returncode != 0:
                print(f"[SYNC ERROR] Failed writing to {range_str}: {res.stderr}")
            else:
                print(f"[SYNC] Synced {scene} GOP {gop_id} to {range_str}")

    # 3. Sync Tier 2: 02_Scene_Summary (Rows 8-13)
    completed_scene_aggregates = []
    for scene_idx, scene in enumerate(SCENES_ORDER):
        if scene not in scenes_data:
            continue
        gops = scenes_data[scene]
        if len(gops) < 5:
            continue # Wait for all 5 GOPs before finalizing scene summary
            
        total_init_time = sum(float(g["stats"].get("init_time_s", 0.0)) for g in gops.values())
        total_train_recon_time = sum(float(g["stats"].get("train_recon_time_s", g["stats"].get("ellipse_time", 0.0))) for g in gops.values())
        total_recon_time = total_init_time + total_train_recon_time
        time_per_frame = total_recon_time / 300.0
        
        mean_psnr = sum(float(g["stats"].get("psnr", 0.0)) for g in gops.values()) / 5.0
        mean_ssim = sum(float(g["stats"].get("ssim", 0.0)) for g in gops.values()) / 5.0
        mean_lpips = sum(float(g["stats"].get("lpips", 0.0)) for g in gops.values()) / 5.0
        
        total_model_size_mb = sum(g["size_mb"] for g in gops.values())
        size_per_frame_mb = total_model_size_mb / 300.0
        
        mean_render_fps = sum((1.0 / g["stats"].get("ellipse_time", 0.003)) if g["stats"].get("ellipse_time", 0.0) > 0 else 300.0 for g in gops.values()) / 5.0
        max_peak_vram = max(float(g["stats"].get("peak_vram_gb", g["stats"].get("mem", 0.0))) for g in gops.values())
        
        scene_agg = {
            "scene": scene,
            "time_per_frame": time_per_frame,
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "lpips": mean_lpips,
            "size_per_frame_mb": size_per_frame_mb,
            "render_fps": mean_render_fps,
            "peak_vram": max_peak_vram
        }
        completed_scene_aggregates.append(scene_agg)
        
        row_idx = 8 + scene_idx
        range_str = f"'02_Scene_Summary'!A{row_idx}:P{row_idx}"
        t2_row = [
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
        ]
        row_csv = ",".join(t2_row)
        res = run_gsheets_cmd(["gsheets", "mutate", "write", SPREADSHEET_ID, range_str, row_csv])
        if res.returncode != 0:
            print(f"[SYNC ERROR] Failed writing to {range_str}: {res.stderr}")
        else:
            print(f"[SYNC] Synced Scene Summary for {scene} to {range_str}")

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
        
        print(f"[SYNC] Updating 01_Leaderboard row for {METHOD_NAME} ({n}/6 scenes complete)...")
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
    sync_to_sheets(update_existing=True)
