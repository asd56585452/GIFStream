#!/bin/bash
set -euo pipefail

# Standard N3DV Scenes (all 6 dynamic scenes, 300 frames each)
SCENES=("sear_steak" "coffee_martini" "cook_spinach" "cut_roasted_beef" "flame_salmon_1" "flame_steak")

# If specific scenes provided as arguments, override
if [ $# -gt 0 ]; then
    SCENES=("$@")
fi

BASE_DIR="/home/tsaichenghan/GIFStream"
DATASETS_DIR="/media/tsaichenghan/2TBSSD/GIFStream/datasets"
OUTPUT_DIR="/media/tsaichenghan/2TBSSD/GIFStream/output"
TORCH_CACHE="/home/tsaichenghan/.cache/torch"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TORCH_CACHE"

echo "=========================================================="
echo " Starting GIFStream (CVPR 2025) Official N3DV Benchmark Pipeline"
echo " Strict Docker Isolation: asdggg/gifstream:latest"
echo " Strict Read-Only Datasets: /media/tsaichenghan/2TBSSD/datasets (:ro)"
echo " Official COLMAP Triangulation: Mandatory replication"
echo " Target Scenes: ${SCENES[*]}"
echo " Output Directory: $OUTPUT_DIR"
echo "=========================================================="

for SCENE in "${SCENES[@]}"; do
    echo ""
    echo "=========================================================="
    echo " Processing Scene: $SCENE"
    echo "=========================================================="
    
    # Scene Type determination according to official benchmark script
    if [ "$SCENE" = "coffee_martini" ]; then
        TYPE="neur3d_2"
    elif [ "$SCENE" = "flame_salmon_1" ]; then
        TYPE="neur3d_1"
    else
        TYPE="neur3d_0"
    fi
    
    # 5 GOPs of 60 frames each = 300 frames
    for GOP_ID in {0..4}; do
        GOP_START_FRAME=$((GOP_ID * 60))
        EXP_NAME="$OUTPUT_DIR/$SCENE/GOP_$GOP_ID/r0"
        mkdir -p "$EXP_NAME"
        
        # Check if already completed
        if [ -f "$EXP_NAME/stats/compress_step29999.json" ]; then
            echo "[SKIP] $SCENE GOP_$GOP_ID already completed. Skipping training and compression."
            continue
        fi
        
        echo "----------------------------------------------------------"
        echo " Scene: $SCENE | GOP: $GOP_ID (Frames $GOP_START_FRAME-$((GOP_START_FRAME + 59)))"
        echo " Output: $EXP_NAME"
        echo "----------------------------------------------------------"
        
        # 1. Ensure Official COLMAP Triangulation exists in Docker
        echo "[COLMAP] Ensuring official COLMAP triangulation in Docker..."
        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -e QT_QPA_PLATFORM=offscreen \
            -v "$BASE_DIR:/workspace" \
            -v /media/tsaichenghan/2TBSSD/datasets:/media/tsaichenghan/2TBSSD/datasets:ro \
            -v /media/tsaichenghan/2TBSSD/GIFStream:/media/tsaichenghan/2TBSSD/GIFStream \
            asdggg/gifstream:latest python3 /workspace/scripts/generate_official_colmap_in_docker.py --scenes "$SCENE" --gops "$GOP_ID"
            
        # 2. Check if 30k checkpoint exists to skip training
        if [ ! -f "$EXP_NAME/ckpts/ckpt_29999_rank0.pt" ]; then
            if [ "$GOP_ID" -eq 0 ]; then
                echo "[TRAIN] GOP 0: Training from scratch (30,000 steps)..."
                docker run --rm --gpus all \
                    --user "$(id -u):$(id -g)" \
                    -v "$BASE_DIR:/workspace" \
                    -v /media/tsaichenghan/2TBSSD/datasets:/media/tsaichenghan/2TBSSD/datasets:ro \
                    -v /media/tsaichenghan/2TBSSD/GIFStream:/media/tsaichenghan/2TBSSD/GIFStream \
                    -v "$TORCH_CACHE:/tmp/.cache/torch" \
                    -e TORCH_HOME=/tmp/.cache/torch \
                    -e MPLCONFIGDIR=/tmp/mpl_config \
                    -e HOME=/tmp \
                    asdggg/gifstream:latest python3 examples/simple_trainer_GIFStream.py "$TYPE" \
                    --disable_viewer --data_factor 2 --render_traj_path ellipse \
                    --data_dir "$DATASETS_DIR/$SCENE/" --result_dir "$EXP_NAME" \
                    --eval_steps 7000 30000 --save_steps 7000 30000 \
                    --compression_sim --rd_lambda 0.0005 --entropy_model_opt --rate 0 \
                    --batch_size 1 --GOP_size 60 --knn --start_frame "$GOP_START_FRAME"
            else
                echo "[TRAIN] GOP $GOP_ID: Continuing training from GOP 0 checkpoint (30,000 steps)..."
                CKPT_GOP0="$OUTPUT_DIR/$SCENE/GOP_0/r0/ckpts/ckpt_6999_rank0.pt"
                if [ ! -f "$CKPT_GOP0" ]; then
                    echo "[ERROR] Required checkpoint $CKPT_GOP0 does not exist!"
                    exit 1
                fi
                
                docker run --rm --gpus all \
                    --user "$(id -u):$(id -g)" \
                    -v "$BASE_DIR:/workspace" \
                    -v /media/tsaichenghan/2TBSSD/datasets:/media/tsaichenghan/2TBSSD/datasets:ro \
                    -v /media/tsaichenghan/2TBSSD/GIFStream:/media/tsaichenghan/2TBSSD/GIFStream \
                    -v "$TORCH_CACHE:/tmp/.cache/torch" \
                    -e TORCH_HOME=/tmp/.cache/torch \
                    -e MPLCONFIGDIR=/tmp/mpl_config \
                    -e HOME=/tmp \
                    asdggg/gifstream:latest python3 examples/simple_trainer_GIFStream.py "$TYPE" \
                    --disable_viewer --data_factor 2 --render_traj_path ellipse \
                    --data_dir "$DATASETS_DIR/$SCENE/" --result_dir "$EXP_NAME" \
                    --eval_steps 7000 30000 --save_steps 7000 30000 \
                    --compression_sim --rd_lambda 0.0005 --entropy_model_opt --rate 0 \
                    --batch_size 1 --GOP_size 60 --knn --start_frame "$GOP_START_FRAME" \
                    --ckpt "$CKPT_GOP0" --continue_training
            fi
        fi
        
        # 3. Run End-to-End Compression & Validation
        echo "[COMPRESS] Running end-to-end compression and decompression evaluation..."
        docker run --rm --gpus all \
            --user "$(id -u):$(id -g)" \
            -v "$BASE_DIR:/workspace" \
            -v /media/tsaichenghan/2TBSSD/datasets:/media/tsaichenghan/2TBSSD/datasets:ro \
            -v /media/tsaichenghan/2TBSSD/GIFStream:/media/tsaichenghan/2TBSSD/GIFStream \
            -v "$TORCH_CACHE:/tmp/.cache/torch" \
            -e TORCH_HOME=/tmp/.cache/torch \
            -e MPLCONFIGDIR=/tmp/mpl_config \
            -e HOME=/tmp \
            asdggg/gifstream:latest python3 examples/simple_trainer_GIFStream.py "$TYPE" \
            --disable_viewer --data_factor 2 --render_traj_path ellipse \
            --data_dir "$DATASETS_DIR/$SCENE/" --result_dir "$EXP_NAME" \
            --ckpt "$EXP_NAME/ckpts/ckpt_29999_rank0.pt" \
            --compression end2end --rate 0 \
            --GOP_size 60 --knn --start_frame "$GOP_START_FRAME"
            
        echo "[SYNC] Streaming completed GOP_$GOP_ID to Google Sheets..."
        python3 "$BASE_DIR/scripts/sync_gifstream_to_sheets.py" || echo "[WARN] Sync failed, will retry later."
    done
    
    echo "=========================================================="
    echo " Scene $SCENE Complete! Running Full Scene Sync..."
    echo "=========================================================="
    python3 "$BASE_DIR/scripts/sync_gifstream_to_sheets.py" || echo "[WARN] Sync failed, will retry later."
done

echo ""
echo "=========================================================="
echo " All GIFStream N3DV Benchmark Runs Finished Successfully!"
echo "=========================================================="
