export CUDA_VISIBLE_DEVICES="4"
export num_gpu=2
export use_multi_gpu=false
export task='sgcls'

export bases=8
export n_relation=9

export use_obj_refine=True
export set_on=False
export resampling=True
export sampling_method=bilvl
export predictor=HetSGG_Predictor_1 # ARGCN_Predictor, CausalAnalysisPredictor, HetSGG_Predictor, Hetero_RGCN
export REPEAT_FACTOR=0.1 # mylvl : 0.025(v2), ours : 0.014
export INSTANCE_DROP_RATE=0.9
export category=VG-SGG-Category_v2
export IS_ASSIGN_GT_CAT=False
export agg=mean
export exclude_ablation="edge"
export output_dir="checkpoints/${task}-${predictor}-Bases${bases}-Dim128-sampling_${resampling}-method_${sampling_method}_drop_${INSTANCE_DROP_RATE}_repeat_${REPEAT_FACTOR}_v2_assign_gt_${IS_ASSIGN_GT_CAT}_agg_${agg}_${exclude_ablation}"

if $use_multi_gpu;then
    # Multi GPU -sgcls Task
    python -m torch.distributed.launch --master_port 10032 --nproc_per_node=$num_gpu tools/relation_train_net.py --config-file "configs/relRGCN_vg.yaml" \
        SOLVER.IMS_PER_BATCH 24 \
        SOLVER.MAX_ITER 60000 \
        SOLVER.VAL_PERIOD 1500 \
        SOLVER.CHECKPOINT_PERIOD 1500 \
        LOSS_PERIOD 300 \
        TEST.IMS_PER_BATCH 16 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
        MODEL.ROI_RELATION_HEAD.RGCN.N_BASES ${bases} \
        MODEL.ROI_RELATION_HEAD.RGCN.H_DIM 128 \
        MODEL.ROI_RELATION_HEAD.RGCN.CLASS_AGG ${agg} \
        MODEL.ROI_RELATION_HEAD.RGCN.IS_ASSIGN_GT_CAT ${IS_ASSIGN_GT_CAT} \
        MODEL.ROI_RELATION_HEAD.RGCN.NUM_RELATION ${n_relation} \
        MODEL.ROI_RELATION_HEAD.RGCN.CATEGORY_FILE ${category} \
        MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON ${set_on} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_METHOD ${sampling_method} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE}
else
    # Single GPU
    python  tools/relation_train_net.py --config-file "configs/relRGCN_vg.yaml" \
        SOLVER.IMS_PER_BATCH 9 \
        SOLVER.MAX_ITER 50000 \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        LOSS_PERIOD 300 \
        TEST.IMS_PER_BATCH 6 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
        MODEL.ROI_RELATION_HEAD.RGCN.N_BASES ${bases} \
        MODEL.ROI_RELATION_HEAD.RGCN.H_DIM 128 \
        MODEL.ROI_RELATION_HEAD.RGCN.EXCLUDE_ABLATION ${exclude_ablation} \
        MODEL.ROI_RELATION_HEAD.RGCN.CLASS_AGG ${agg} \
        MODEL.ROI_RELATION_HEAD.RGCN.IS_ASSIGN_GT_CAT ${IS_ASSIGN_GT_CAT} \
        MODEL.ROI_RELATION_HEAD.RGCN.NUM_RELATION ${n_relation} \
        MODEL.ROI_RELATION_HEAD.RGCN.CATEGORY_FILE ${category} \
        MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON ${set_on} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING ${resampling} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_METHOD ${sampling_method} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE}
fi