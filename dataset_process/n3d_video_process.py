import os
import subprocess
from argparse import ArgumentParser
import sys
import shutil
import glob
from pre_colmap import COLMAPDatabase
import numpy as np

def posetow2c_matrcs(poses):
    tmp = inversestep4(inversestep3(inversestep2(inversestep1(poses))))
    N = tmp.shape[0]
    ret = []
    for i in range(N):
        ret.append(tmp[i])
    return ret

def inversestep4(c2w_mats):
    return np.linalg.inv(c2w_mats)
def inversestep3(newposes):
    tmp = newposes.transpose([2, 0, 1]) # 20, 3, 4 
    N, _, __ = tmp.shape
    zeros = np.zeros((N, 1, 4))
    zeros[:, 0, 3] = 1
    c2w_mats = np.concatenate([tmp, zeros], axis=1)
    return c2w_mats

def inversestep2(newposes):
    return newposes[:,0:4, :]
def inversestep1(newposes):
    poses = np.concatenate([newposes[:, 1:2, :], newposes[:, 0:1, :], -newposes[:, 2:3, :],  newposes[:, 3:4, :],  newposes[:, 4:5, :]], axis=1)
    return poses

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

def convertdynerftocolmapdb(path, offset=0):
    originnumpy = os.path.join(path, "poses_bounds.npy")
    mp4_path = os.path.join(path,"mp4")
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    #sparsefolder = os.path.join(projectfolder, "sparse/0")
    manualfolder = os.path.join(projectfolder, "manual")

    # if not os.path.exists(sparsefolder):
    #     os.makedirs(sparsefolder)
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


    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        llffposes = poses.copy().transpose(1,2,0)
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert (type(w2c_matriclist) == list)


        for i in range(len(poses)):
            cameraname = f"cam{i:02d}" #"cam" + str(i).zfill(2)
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]
            
            H, W, focal = poses[i, :, -1]
            
            colmapQ = rotmat2qvec(colmapR)
            # colmapRcheck = qvec2rotmat(colmapQ)

            imageid = str(i+1)
            cameraid = imageid
            pngname = cameraname + ".png"
            
            line =  imageid + " "

            for j in range(4):
                line += str(colmapQ[j]) + " "
            for j in range(3):
                line += str(T[j]) + " "
            line = line  + cameraid + " " + pngname + "\n"
            empltyline = "\n"
            imagetxtlist.append(line)
            imagetxtlist.append(empltyline)

            focolength = focal
            model, width, height, params = i, W, H, np.array((focolength,  focolength, W//2, H//2,))

            camera_id = db.add_camera(1, width, height, params)
            cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"
            cameratxtlist.append(cameraline)
            
            image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)
            db.commit()
        db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 

# ==============================================================================
# 區塊 2: COLMAP 格式處理函式 (包含更新後的函式)
# ==============================================================================

def update_and_save_images_txt(original_txt_path, new_txt_path):
    """
    讀取原始 images.txt，從舊檔名解析攝影機編號，
    將檔名更新為 camXX.png 格式，並儲存為新檔案。
    """
    new_lines = []
    with open(original_txt_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            new_lines.append(lines[i])
            i += 1
            continue

        parts = line.split(' ')
        pose_info = " ".join(parts[:-1])
        old_filename = parts[-1]

        try:
            base_name = os.path.basename(old_filename)
            camera_num_str = os.path.splitext(base_name)[0]
            camera_num = int(camera_num_str)
            new_filename = f"cam{camera_num:02d}.png"
        except (ValueError, IndexError):
            print(f"警告: 無法從舊檔名 '{old_filename}' 中解析出攝影機編號。將保持原樣。")
            new_filename = old_filename

        new_line = f"{pose_info} {new_filename}\n"
        new_lines.append(new_line)

        if i + 1 < len(lines):
            new_lines.append(lines[i+1])

        i += 2

    with open(new_txt_path, 'w') as f:
        f.writelines(new_lines)
    print(f"已從舊檔名解析並更新，儲存新的 images.txt 到: {new_txt_path}")


def create_db_from_colmap_txt(colmap_txt_dir, db_path):
    """
    從 cameras.txt 和 images.txt 建立 COLMAP 資料庫 (.db)。
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    with open(os.path.join(colmap_txt_dir, 'cameras.txt'), 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(); camera_id = int(parts[0]); model = parts[1]; width, height = int(parts[2]), int(parts[3]); params = np.array(list(map(float, parts[4:])))
            db.add_camera(1, width, height, params, camera_id=camera_id)
            print(model, width, height, params, camera_id)

    with open(os.path.join(colmap_txt_dir, 'images.txt'), 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '': continue
            parts = line.strip().split(); image_id = int(parts[0]); qvec = np.array(list(map(float, parts[1:5]))); tvec = np.array(list(map(float, parts[5:8]))); camera_id = int(parts[8]); image_name = parts[9]
            db.add_image(image_name, camera_id, prior_q=qvec, prior_t=tvec, image_id=image_id)
            try: next(f)
            except StopIteration: pass
    db.commit()
    db.close()
    print(f"從 {colmap_txt_dir} 成功建立資料庫 {db_path}")

# ==============================================================================
# 區塊 3: COLMAP 核心處理函式
# ==============================================================================
def getcolmapsinglen3d(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder + " --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 106384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 --ImageReader.camera_model PINHOLE"

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel  + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
# ==============================================================================
# 區塊 4: 主程式進入點
# ==============================================================================
if __name__ == "__main__" :
    parser = ArgumentParser(description="通用影片/COLMAP資料集處理腳本")
    parser.add_argument("--input_format", type=str, default='nerf', choices=['nerf', 'colmap'], help="輸入資料的格式: 'nerf' (使用 poses_bounds.npy) 或 'colmap' (使用現有 COLMAP 文字檔)")
    parser.add_argument("--colmap_path", type=str, default=None, help="當 input_format 為 'colmap' 時，提供包含 cameras.txt, images.txt 的 COLMAP sparse/0 資料夾路徑")
    parser.add_argument("--root_dir", type=str, required=True, help="資料集根目錄")
    parser.add_argument("--extract_frames", action='store_true', help="是否從影片中提取影格")
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=300, type=int)
    parser.add_argument("--GOP", default=60, type=int)
    args = parser.parse_args()

    if args.input_format == 'colmap' and not args.colmap_path:
        parser.error("--colmap_path 是 --input_format='colmap' 模式下的必需參數。")

    for folder_name in os.listdir(args.root_dir):
        source_path = os.path.join(args.root_dir, folder_name)
        if not os.path.isdir(source_path): continue

        if args.input_format == 'colmap' and not os.path.exists(os.path.join(source_path,args.colmap_path)):
            parser.error(f"提供的 --colmap_path '{os.path.join(source_path,args.colmap_path)}' 不存在。")

        output_path_png = os.path.join(source_path, "png")
        if args.extract_frames:
            print(f"[{folder_name}] 正在從影片提取影格...")
            if not os.path.exists(output_path_png): os.mkdir(output_path_png)
            mp4_files = sorted([name for name in os.listdir(source_path) if name.endswith(".mp4")])
            for idx, file_name in enumerate(mp4_files):
                video_path = os.path.join(source_path, file_name); output_folder = os.path.join(output_path_png, f"cam{idx:02d}")
                if not os.path.exists(output_folder): os.mkdir(output_folder)
                cmd = f"ffmpeg -i {video_path} -vf fps={args.frame_rate} -q:v 2 {output_folder}/%05d.png"; subprocess.call(cmd, shell=True)

        camera_names = sorted([name for name in os.listdir(output_path_png) if os.path.isdir(os.path.join(output_path_png, name))]) if os.path.exists(output_path_png) else []
        frame_list = [args.startframe + x * args.GOP for x in range((args.endframe - args.startframe + args.GOP - 1) // args.GOP)]

        for frame in frame_list:
            print(f"[{folder_name}] 正在處理影格 {frame}...")
            colmap_output_path = os.path.join(source_path, f"colmap_{frame}"); input_image_path = os.path.join(colmap_output_path, "input"); manual_path = os.path.join(colmap_output_path, "manual")
            os.makedirs(input_image_path, exist_ok=True); os.makedirs(manual_path, exist_ok=True)
            
            for ind, cam in enumerate(camera_names):
                image_path = os.path.join(output_path_png, cam, f"{(frame+1):05d}.png"); save_path = os.path.join(input_image_path, f"cam{ind:02d}.png")
                if os.path.exists(image_path): shutil.copy(image_path, save_path)
                else: print(f"警告: 找不到影像 {image_path}，跳過。")

            if args.input_format == 'nerf':
                print(f"模式 'nerf': 從 poses_bounds.npy 產生 COLMAP DB...")
                if not os.path.exists(os.path.join(source_path, "poses_bounds.npy")):
                    print(f"錯誤: 在 {source_path} 中找不到 poses_bounds.npy，但模式設定為 'nerf'。"); continue
                convertdynerftocolmapdb(source_path, frame)
            elif args.input_format == 'colmap':
                print(f"模式 'colmap': 從現有 COLMAP 文字檔產生 COLMAP DB...")
                original_images_txt = os.path.join(source_path,args.colmap_path, 'images.txt')
                new_manual_images_txt = os.path.join(manual_path, 'images.txt')
                if os.path.exists(original_images_txt):
                    update_and_save_images_txt(original_images_txt, new_manual_images_txt)
                else:
                    print(f"錯誤: 在 {os.path.join(source_path,args.colmap_path)} 中找不到 images.txt"); continue
                
                for txt_file in ['cameras.txt', 'points3D.txt']:
                    src_txt = os.path.join(source_path,args.colmap_path, txt_file)
                    if os.path.exists(src_txt): shutil.copy(src_txt, manual_path)
                    else: 
                        print(f"警告: 在 {os.path.join(source_path,args.colmap_path)} 中找不到 {txt_file}。")
                        with open(os.path.join(manual_path, txt_file), "w") as f:
                            pass
                
                db_path = os.path.join(colmap_output_path, "input.db")
                create_db_from_colmap_txt(manual_path, db_path)
            
            print(f"[{folder_name}] 開始執行 COLMAP 處理...")
            getcolmapsinglen3d(source_path, frame)

    print("所有處理已完成！")