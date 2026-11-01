import os
from pathlib import Path

SOURCE_ROOT = "/media/tsaichenghan/2TBSSD/datasets"
TARGET_ROOT = "/media/tsaichenghan/2TBSSD/GIFStream/datasets"
SCENES = ['sear_steak', 'coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'flame_steak']

def setup_scene(scene_name: str):
    src_scene = os.path.join(SOURCE_ROOT, scene_name)
    tgt_scene = os.path.join(TARGET_ROOT, scene_name)
    os.makedirs(tgt_scene, exist_ok=True)
    
    # 1. Symlink poses_bounds.npy
    src_poses = os.path.join(src_scene, "poses_bounds.npy")
    tgt_poses = os.path.join(tgt_scene, "poses_bounds.npy")
    if os.path.exists(src_poses) and not os.path.exists(tgt_poses):
        os.symlink(src_poses, tgt_poses)
        
    # 2. Symlink colmap directories
    for item in os.listdir(src_scene):
        if item.startswith("colmap_") and os.path.isdir(os.path.join(src_scene, item)):
            src_colmap = os.path.join(src_scene, item)
            tgt_colmap = os.path.join(tgt_scene, item)
            if not os.path.exists(tgt_colmap):
                os.symlink(src_colmap, tgt_colmap)
                
    # 3. Create png/camXX symlinks
    tgt_png_dir = os.path.join(tgt_scene, "png")
    os.makedirs(tgt_png_dir, exist_ok=True)
    
    cams = sorted([d for d in os.listdir(src_scene) if d.startswith("cam") and os.path.isdir(os.path.join(src_scene, d))])
    for cam in cams:
        src_cam_dir = os.path.join(src_scene, cam)
        tgt_cam_dir = os.path.join(tgt_png_dir, cam)
        os.makedirs(tgt_cam_dir, exist_ok=True)
        
        # Link frames 0.png -> 00001.png, ..., 299.png -> 00300.png
        for i in range(300):
            src_img = os.path.join(src_cam_dir, f"{i}.png")
            tgt_img = os.path.join(tgt_cam_dir, f"{i+1:05d}.png")
            if os.path.exists(src_img) and not os.path.exists(tgt_img):
                os.symlink(src_img, tgt_img)
                
    print(f"Setup complete for {scene_name}: {len(cams)} cameras linked in {tgt_png_dir}")

if __name__ == "__main__":
    for s in SCENES:
        setup_scene(s)

