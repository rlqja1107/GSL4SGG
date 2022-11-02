export CUDA_VISIBLE_DEVICES="5"
export num_gpu=1
export use_multi_gpu=false
export task='sgdet'

export use_obj_refine=False
export resampling=True
export REPEAT_FACTOR=0.1
export INSTANCE_DROP_RATE=0.9
export config=vg

export set_on=True
export is_reporting=True
export report_criteria=original
export apply_gt=True
export consideration_non_gt=False
export pretrain_relness=True

# Density
export fill_criteria="zero" # zero, reverse, uniform, iou_filter
export below_zero=0.3 # reverse

export iou_bg_filter=False # BG를 IoU로 Filtering할 것인지
export iou_multi=0.168
export zero_constant=0.828

# export output_dir="checkpoints/${task}_BGNNPredictor_bilvl_${set_on}_refine_${use_obj_refine}_${config}_is_rp(${is_reporting})_rp_thres(${reporting_thres})_rp_criteria(${report_criteria})_gt_(${apply_gt})_consideration_non_gt_${consideration_non_gt}"
# export output_dir="checkpoints/${task}_Density_BGNNPredictor_bilvl_${resampling}_reporting(${is_reporting})_gt_(${apply_gt})_fill_criteria(${fill_criteria})_below(${below_zero})_15percen"
# export output_dir="checkpoints/${task}_Density_BGNNPredictor_bilvl_${resampling}_reporting(${is_reporting})_gt_(${apply_gt})_fill_criteria(${fill_criteria})_filter_bg(${iou_bg_filter})_multi(${iou_multi})_17.5percent"
export output_dir="checkpoints/${task}_Density_BGNNPredictor_bilvl_${resampling}_reporting(${is_reporting})_gt_(${apply_gt})_fill_criteria(${fill_criteria})_filter_bg(${iou_bg_filter})_zero_constant(${zero_constant})_20percent"

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
    python  tools/relation_train_net.py --config-file "configs/e2e_relBGNN_vg_Reporting.yaml" \
        SOLVER.IMS_PER_BATCH 6 \
        SOLVER.MAX_ITER 24000 \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        LOSS_PERIOD 150 \
        TEST.IMS_PER_BATCH 1 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.GEOMETRIC_FEATURES True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON ${set_on} \
        IS_REPORTING ${is_reporting} \
        REPORTING_CRITERIA ${report_criteria} \
        MODEL.ROI_RELATION_HEAD.BGNN_MODULE.APPLY_GT ${apply_gt} \
        CONSIDERATION_NON_GT ${consideration_non_gt} \
        MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_RELNESS_MODULE ${pretrain_relness} \
        FILL_CRITERIA ${fill_criteria} \
        BELOW_ZERO ${below_zero} \
        BG_FILTER ${iou_bg_filter} \
        IOU_MULTI ${iou_multi} \
        FOR_ZERO_CONSTANT ${zero_constant}

fi