# Set the directory containing the scenes
SCENE_DIR="/home/cgvmis418/GIFStream/dataset/VideoGS"
# Set the directory to store results
RESULT_DIR="/home/cgvmis418/GIFStream/gscodec/GIFStream_branch_videogs_more-refine"
# Set the rendering trajectory path
RENDER_TRAJ_PATH="ellipse"
# List of scenes to process
SCENE_LIST="4K_Actor1_Greeting"
# List of entropy lambda values (rate-distortion tradeoff parameter)
ENTROPY_LAMBDA_LIST=(0.0005 )
# Data factor for training
DATA_FACTOR=2
# Number of frames per GOP (Group of Pictures)
GOP=50
# The index of the first frame to process
FIRST_FRAME=0
# Total number of frames to process
TOTAL_FRAME=200

SCALE=1

# Loop over each scene in the scene list
for SCENE in $SCENE_LIST;
do
    # Set TYPE based on the scene name
    if [ "$SCENE" = "coffee_martini" ]; then
        TYPE=neur3d_2
    elif [ "$SCENE" = "flame_salmon_1" ]; then
        TYPE=neur3d_1
    else
        TYPE=neur3d_0
    fi

    # Loop over each entropy lambda (rate)
    for ((RATE=0; RATE<${#ENTROPY_LAMBDA_LIST[@]}; RATE++));
    do
        # Loop over each GOP segment
        for ((GOP_ID=0; GOP_ID < $(((TOTAL_FRAME + GOP - 1)/GOP)) ; GOP_ID++));
        do
            echo "Running $SCENE"
            # Set experiment name and output directory
            EXP_NAME=$RESULT_DIR/${SCENE}/GOP_$GOP_ID/r$RATE
            # Calculate the starting frame for this GOP
            GOP_START_FRAME=$((FIRST_FRAME + GOP_ID * GOP ))
            # Calculate the maximum number of frames for this GOP
            MAX_GOP=$((TOTAL_FRAME + FIRST_FRAME - GOP_START_FRAME))
            if ((GOP_ID == 0)); then

                # Run evaluation and rendering after training
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --ckpt $EXP_NAME/ckpts/ckpt_$(( 30000 * SCALE - 1))_rank0.pt \
                    --compression end2end  --rate $RATE \
                    --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME --random-bkgd
            else

                # Run evaluation and rendering after training
                CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                    --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                    --ckpt $EXP_NAME/ckpts/ckpt_$(( 30000 * SCALE - 1))_rank0.pt \
                    --compression end2end  --rate $RATE \
                    --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME --random-bkgd
            fi
        done
    done
done

# Run the summary script to aggregate results
python examples/summary.py --root_dir $RESULT_DIR
