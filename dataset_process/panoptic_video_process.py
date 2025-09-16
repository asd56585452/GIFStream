import os
import subprocess
from argparse import ArgumentParser
import sys
import shutil
import glob
from pre_colmap import COLMAPDatabase
import numpy as np
import json

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def convert_panoptic_to_colmap_db(path, hd_cameras, offset=0):
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    manualfolder = os.path.join(projectfolder, "manual")

    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))
    db.create_tables()

    for i, cam_info in enumerate(hd_cameras):
        R = np.array(cam_info['R'])
        t = np.array(cam_info['t']).flatten()
        K = np.array(cam_info['K'])
        dist = np.array(cam_info['distCoef']).flatten()

        # Correct extrinsic conversion: C = -R^T * t
        T = -np.dot(R.T, t)

        W, H = cam_info['resolution']
        focal_x = K[0,0]
        focal_y = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        # Use OPENCV model which supports distortion
        # It expects fx, fy, cx, cy, k1, k2, p1, p2
        # Panoptic provides k1, k2, p1, p2, k3. We'll use the first 4.
        params = np.array([focal_x, focal_y, cx, cy, dist[0], dist[1], dist[2], dist[3]])

        qvec = rotmat2qvec(R)

        image_id = i + 1
        camera_id = i + 1

        pngname = f"cam_{cam_info['name']}.png"

        line = f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} {camera_id} {pngname}\n\n"
        imagetxtlist.append(line)

        # COLMAP OPENCV camera model is ID 4
        camera_model_id = 4
        db.add_camera(model=camera_model_id, width=W, height=H, params=params, camera_id=camera_id)

        param_str = " ".join(map(str, params))
        cameraline = f"{camera_id} OPENCV {W} {H} {param_str}\n"
        cameratxtlist.append(cameraline)

        db.add_image(name=pngname, camera_id=camera_id, prior_q=qvec, prior_t=T, image_id=image_id)

    db.commit()
    db.close()

    with open(savetxt, "w") as f:
        f.writelines(imagetxtlist)
    with open(savecamera, "w") as f:
        f.writelines(cameratxtlist)
    with open(savepoints, "w") as f:
        pass

def run_colmap(path, offset):
    folder = os.path.join(path, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    manualinputfolder = os.path.join(folder, "manual")

    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    feature_extractor_cmd = f"colmap feature_extractor --database_path {dbfile} --image_path {inputimagefolder}"
    subprocess.run(feature_extractor_cmd, shell=True, check=True)

    feature_matcher_cmd = f"colmap exhaustive_matcher --database_path {dbfile}"
    subprocess.run(feature_matcher_cmd, shell=True, check=True)

    point_triangulator_cmd = f"colmap point_triangulator --database_path {dbfile} --image_path {inputimagefolder} --output_path {distortedmodel} --input_path {manualinputfolder}"
    subprocess.run(point_triangulator_cmd, shell=True, check=True)

    img_undistorter_cmd = f"colmap image_undistorter --image_path {inputimagefolder} --input_path {distortedmodel} --output_path {folder} --output_type COLMAP"
    subprocess.run(img_undistorter_cmd, shell=True, check=True)

    shutil.rmtree(inputimagefolder)

    files = os.listdir(os.path.join(folder, "sparse"))
    os.makedirs(os.path.join(folder, "sparse/0"), exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

if __name__ == "__main__":
    parser = ArgumentParser(description="Panoptic Sport Dataset Processor")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the Panoptic dataset")
    parser.add_argument("--extract_frames", action='store_true', help="Extract frames from videos")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate for extraction")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for extraction")
    parser.add_argument("--end_frame", type=int, default=300, help="End frame for extraction")
    args = parser.parse_args()

    for scene_folder in os.listdir(args.root_dir):
        scene_path = os.path.join(args.root_dir, scene_folder)
        if not os.path.isdir(scene_path):
            continue

        video_folder = os.path.join(scene_path, "hdVideos")
        output_path = os.path.join(scene_path, "png")

        calibration_file = glob.glob(os.path.join(scene_path, "calibration*.json"))
        if not calibration_file:
            print(f"Calibration file not found in {scene_path}, skipping.")
            continue

        with open(calibration_file[0]) as f:
            calibration_data = json.load(f)

        hd_cameras = [cam for cam in calibration_data['cameras'] if cam.get('type') == 'hd']
        hd_cameras = sorted(hd_cameras, key=lambda x: x['name'])

        # --- DEBUG: Test with only the first 31 cameras to isolate crash ---
        print("DEBUG: Testing with only the first 31 cameras.")
        hd_cameras = hd_cameras[:31]
        # --- END DEBUG ---

        if args.extract_frames:
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for cam_info in hd_cameras:
                cam_name = cam_info['name']
                video_path = os.path.join(video_folder, f"hd_{cam_name}.mp4")
                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue

                output_cam_folder = os.path.join(output_path, f"cam_{cam_name}")
                if not os.path.exists(output_cam_folder):
                    os.makedirs(output_cam_folder)

                cmd = (f"ffmpeg -i {video_path} -vf \"select='between(n,{args.start_frame},{args.end_frame})'\" "
                       f"-start_number {args.start_frame + 1} -vsync vfr {output_cam_folder}/%05d.png")
                subprocess.call(cmd, shell=True)

        colmap_offset = args.start_frame
        colmap_path = os.path.join(scene_path, f"colmap_{colmap_offset}")
        input_image_path = os.path.join(colmap_path, "input")
        if not os.path.exists(input_image_path):
            os.makedirs(input_image_path)

        for cam_info in hd_cameras:
            cam_name = cam_info['name']
            cam_folder = os.path.join(output_path, f"cam_{cam_name}")
            first_frame_filename = f"{(args.start_frame + 1):05d}.png"
            frame_file = os.path.join(cam_folder, first_frame_filename)
            if os.path.exists(frame_file):
                shutil.copy(frame_file, os.path.join(input_image_path, f"cam_{cam_name}.png"))
            else:
                print(f"Warning: Could not find {frame_file} for cam {cam_name}")

        convert_panoptic_to_colmap_db(scene_path, hd_cameras, colmap_offset)
        run_colmap(scene_path, colmap_offset)
