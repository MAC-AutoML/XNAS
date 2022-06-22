OUT_NAME="OFA_trial_25"
TASKS="normal_1 kernel_1 depth_1 depth_2 expand_1 expand_2"

for loop in $TASKS
do
    # echo `torchrun --nproc_per_node 2 scripts/search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3/$loop.yaml OUT_DIR exp/search/$OUT_NAME/$loop OPTIM.MAX_EPOCH 2 OPTIM.WARMUP_EPOCH 2 LOADER.BATCH_SIZE 128`
    echo "using gpus: $@"
    echo `python scripts/search/OFA/train_supernet.py --cfg configs/search/OFA/mbv3/$loop.yaml OUT_DIR exp/search/$OUT_NAME/$loop NUM_GPUS $@`
    echo `sleep 5s`
done

# # full supernet
# python scripts/search/OFA/train_supernet.py --cfg configs/OFA/mbv3/OFA_normal_phase1.yaml OUT_DIR exp/$OUT_NAME/normal_1
# # elastic kernel size
# python scripts/search/OFA/train_supernet.py --cfg configs/OFA/mbv3/OFA_kernel_phase1.yaml OUT_DIR exp/$OUT_NAME/kernel_1
# # elastic depth
# python scripts/search/OFA/train_supernet.py --cfg configs/OFA/mbv3/OFA_depth_phase1.yaml OUT_DIR exp/$OUT_NAME/depth_1
# python scripts/search/OFA/train_supernet.py --cfg configs/OFA/mbv3/OFA_depth_phase2.yaml OUT_DIR exp/$OUT_NAME/depth_2
# # elastic width
# python scripts/search/OFA/train_supernet.py --cfg configs/OFA/mbv3/OFA_expand_phase1.yaml OUT_DIR exp/$OUT_NAME/expand_1
# python scripts/search/OFA/train_supernet.py --cfg configs/OFA/mbv3/OFA_expand_phase1.yaml OUT_DIR exp/$OUT_NAME/expand_1
