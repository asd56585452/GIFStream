import os
import shutil
import argparse
from pathlib import Path

def reorganize_images(source_root, target_root):
    """
    將圖片從 VideoGS/LLFF 格式重組為 n3d_video_process 所需的格式。
    - 來源: .../{data_name}/image_undistortion_white/{frame_num}/{camera_num}.png
    - 目標: .../{data_name}/png/cam{camera_num_padded}/{frame_num_padded}.png
    """
    print(f"來源根目錄: {source_root}")
    print(f"目標根目錄: {target_root}")
    print("開始重組圖片...")

    source_root_path = Path(source_root)
    target_root_path = Path(target_root)

    # 檢查來源目錄是否存在
    if not source_root_path.exists():
        print(f"錯誤：來源目錄 '{source_root}' 不存在。")
        return

    # 遍歷來源根目錄下的所有 data 資料夾 (例如 data1, data2)
    for data_path in source_root_path.iterdir():
        if not data_path.is_dir():
            continue

        data_name = data_path.name
        source_image_dir = data_path / "image_undistortion_white"

        if not source_image_dir.exists():
            print(f"在 '{data_name}' 中找不到 'image_undistortion_white'，跳過。")
            continue

        print(f"\n正在處理資料集: {data_name}")

        # 找出所有的圖片檔案
        # glob 會找到所有符合模式的檔案路徑
        source_files = sorted(source_image_dir.glob('*/*.png'))
        
        if not source_files:
            print(f"在 {source_image_dir} 中找不到任何 .png 檔案。")
            continue

        total_files = len(source_files)
        print(f"找到 {total_files} 個圖片檔案。")

        # 處理每一個找到的圖片檔案
        for i, source_file_path in enumerate(source_files):
            # 從路徑解析出 frame 和 camera 編號
            # source_file_path.parent 是 .../{frame_num}
            # source_file_path.stem 是 {camera_num}
            try:
                frame_num = int(source_file_path.parent.name)
                camera_num = int(source_file_path.stem)
            except ValueError:
                print(f"警告: 無法從 '{source_file_path}' 解析檔名和資料夾名稱，跳過。")
                continue

            # --- 建立目標路徑和檔名 ---
            # 1. 攝影機編號補零 (例如 0 -> cam00)
            camera_padded = f"cam{camera_num:02d}"

            # 2. 影格編號補零 (例如 0 -> 00001)
            # n3d_video_process.py 的 ffmpeg 命令從 1 開始計數
            frame_padded = f"{frame_num + 1:05d}.png"

            # 3. 組合完整的目標路徑
            target_dir = target_root_path / data_name / "png" / camera_padded
            target_file_path = target_dir / frame_padded

            # 建立目標資料夾 (如果不存在)
            target_dir.mkdir(parents=True, exist_ok=True)

            # 複製檔案
            if (i + 1) % 200 == 0 or (i + 1) == total_files: # 每200個檔案或最後一個檔案時印出進度
                 print(f"  ({i+1}/{total_files}) 複製: {source_file_path} \n      -> {target_file_path}")
            shutil.copy(str(source_file_path), str(target_file_path))

    print("\n圖片重組完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="將 VideoGS/LLFF 格式的圖片重組為 n3d_video_process 所需的 png 資料夾結構，以取代影片轉換步驟。"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="包含 {data...} 資料夾的來源根目錄 (例如: Path_to_VideoGS_dir)"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="要建立輸出結構的目標根目錄 (例如: Path_to_Neur3D_dir)"
    )

    args = parser.parse_args()
    reorganize_images(args.source_dir, args.target_dir)