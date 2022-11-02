export CUDA_VISIBLE_DEVICES="0,2,3,4"
export num_gpu=4
export use_multi_gpu=true
export task='sgdet'

export use_obj_refine=False
export sampling_method=bilvl
export set_on=True
export resampling=True
export predictor=AGRCNNPredictor # CausalAnalysisPredictor, AGRCNNPredictor, GPSNetPredictor, BGNNPredictor
export REPEAT_FACTOR=0.1 # mylvl : 0.025(v2), ours : 0.014
export INSTANCE_DROP_RATE=0.9

export output_dir="checkpoints/${task}_Unbiased_Motifs_Sum"
export output_dir="checkpoints/${task}_${predictor}_bilvl_no_val"

if $use_multi_gpu;then
    # Multi GPU -sgcls Task
    python -m torch.distributed.launch --master_port 10030 --nproc_per_node=$num_gpu tools/relation_train_net.py --config-file "configs/relRGCN_vg.yaml" \
        SOLVER.MAX_ITER 50000 \
        SOLVER.IMS_PER_BATCH 24 \
        SOLVER.TO_VAL False \
        TEST.IMS_PER_BATCH 12 \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING True \
        MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON ${set_on} \
        MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD 'rel_pn' \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_METHOD ${sampling_method} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        OUTPUT_DIR ${output_dir}
        # MODEL.PRETRAINED_DETECTOR_CKPT '' \
        # MODEL.WEIGHT "model_0002000.pth"
else
    # Single GPU
    python  tools/relation_train_net.py --config-file "configs/relRGCN_vg.yaml" \
        SOLVER.MAX_ITER 50000 \
        SOLVER.IMS_PER_BATCH 6 \
        TEST.IMS_PER_BATCH 3 \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_METHOD ${sampling_method} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        OUTPUT_DIR ${output_dir}
fi