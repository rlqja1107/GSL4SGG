export CUDA_VISIBLE_DEVICES="0"
export num_gpu=3
export use_multi_gpu=false
export task='predcls'

export use_obj_refine=False 
export resampling=True
export REPEAT_FACTOR=0.13
export INSTANCE_DROP_RATE=1.6
export output_dir="checkpoints/${task}-BGNN-resample_${resampling}-repeat_${REPEAT_FACTOR}_pretrained"
export faster_path=/home/public/Datasets/CV/faster_ckpt/vg_faster_det.pth # /home/public/Datasets/CV/faster_ckpt/vg_faster_det.pth
if $use_multi_gpu;then
    # Multi GPU -sgcls Task
    python -m torch.distributed.launch --master_port 10030 --nproc_per_node=$num_gpu tools/relation_train_net.py --config-file "configs/relRGCN_vg.yaml" \
        SOLVER.IMS_PER_BATCH 9 \
        SOLVER.MAX_ITER 50000 \
        SOLVER.VAL_PERIOD 2000 \
        LOSS_PERIOD 150 \
        TEST.IMS_PER_BATCH 6 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.GEOMETRIC_FEATURES True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling}
else
    # Single GPU
    python  tools/relation_train_net.py --config-file "configs/e2e_relBGNN_vg.yaml" \
        SOLVER.IMS_PER_BATCH 6 \
        SOLVER.MAX_ITER 50000 \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        LOSS_PERIOD 150 \
        TEST.IMS_PER_BATCH 1 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.GEOMETRIC_FEATURES True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        MODEL.PRETRAINED_DETECTOR_CKPT ${faster_path}
fi