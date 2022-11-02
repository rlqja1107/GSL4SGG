export CUDA_VISIBLE_DEVICES="7"
export num_gpu=1
export use_multi_gpu=false
export task='sgcls'

export resampling=False
export REPEAT_FACTOR=0.1
export INSTANCE_DROP_RATE=0.9
export config=vg

export use_obj_refine=True
export predictor="GSL4SGG_Predictor"
export link_loss_type="bce" # bce, focal
export filter_valid=False
export input_format='concat' # concat, minus
export residual=False
export output_dir="checkpoints/${task}_${predictor}_refine_${use_obj_refine}_loss_type(${link_loss_type})_filter_valid(${filter_valid})_input(${input_format})_residual(${residual})_${config}"

# export output_dir="checkpoints/sgcls_compare_bgnn"
if $use_multi_gpu;then
    # Multi GPU -sgcls Task
    python -m torch.distributed.launch --master_port 10034 --nproc_per_node=$num_gpu tools/relation_train_net.py --config-file "configs/relGSL4SGG_vg.yaml" \
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
    python  tools/relation_train_net.py --config-file "configs/relGSL4SGG_vg.yaml" \
        SOLVER.IMS_PER_BATCH 6 \
        SOLVER.MAX_ITER 50000 \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        LOSS_PERIOD 200 \
        TEST.IMS_PER_BATCH 3 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.GEOMETRIC_FEATURES True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
        MODEL.ROI_RELATION_HEAD.GSL_MODULE.FILTER_VALID_NODE ${filter_valid} \
        MODEL.ROI_RELATION_HEAD.GSL_MODULE.LINK_LOSS_TYPE ${link_loss_type} \
        MODEL.ROI_RELATION_HEAD.GSL_MODULE.INPUT_FORMAT ${input_format} \
        MODEL.ROI_RELATION_HEAD.GSL_MODULE.RESIDUAL_CONNECTION ${residual}
fi