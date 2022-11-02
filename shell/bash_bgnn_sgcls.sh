export CUDA_VISIBLE_DEVICES="4"
export num_gpu=1
export use_multi_gpu=false
export task='sgcls'

export use_obj_refine=True
export resampling=False
export REPEAT_FACTOR=0.1
export INSTANCE_DROP_RATE=0.9
export config=vg

export set_on=True
export apply_gt=False
export use_global_context=True
export use_obj_or_rel_context='obj'
export output_dir="checkpoints/${task}_BGNNPredictor_bilvl_${set_on}_refine_${use_obj_refine}_${config}_is_rp(${is_reporting})_rp_thres(${reporting_thres})_rp_criteria(${report_criteria})_gt_(${apply_gt})_consideration_non_gt_${consideration_non_gt}"
export output_dir="checkpoints/${task}_BGNNPredictor_bilvl_${resampling}_refine_${use_obj_refine}_${config}_global_${use_global_context}(${use_obj_or_rel_context})"

# export output_dir="checkpoints/sgcls_compare_bgnn"
if $use_multi_gpu;then
    # Multi GPU -sgcls Task
    python -m torch.distributed.launch --master_port 10034 --nproc_per_node=$num_gpu tools/relation_train_net.py --config-file "configs/e2e_relBGNN_vg.yaml" \
        SOLVER.IMS_PER_BATCH 18 \
        SOLVER.MAX_ITER 50000 \
        SOLVER.VAL_PERIOD 2500 \
        SOLVER.CHECKPOINT_PERIOD 2500 \
        LOSS_PERIOD 150 \
        TEST.IMS_PER_BATCH 2 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.GEOMETRIC_FEATURES True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE}
else
    # Single GPU
    python  tools/relation_train_net.py --config-file "configs/e2e_relBGNN_vg.yaml" \
        SOLVER.IMS_PER_BATCH 6 \
        SOLVER.MAX_ITER 50000 \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        LOSS_PERIOD 200 \
        TEST.IMS_PER_BATCH 1 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.GEOMETRIC_FEATURES True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON ${set_on} \
        MODEL.ROI_RELATION_HEAD.BGNN_MODULE.APPLY_GT ${apply_gt} \
        MODEL.ROI_RELATION_HEAD.BGNN_MODULE.USE_GLOBAL_CONTEXT ${use_global_context} \
        MODEL.ROI_RELATION_HEAD.BGNN_MODULE.USE_OBJ_OR_REL_CONTEXT ${use_obj_or_rel_context}
fi