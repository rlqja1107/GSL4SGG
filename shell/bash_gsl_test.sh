export CUDA_VISIBLE_DEVICES="5"
export num_gpu=1
export use_multi_gpu=false

# export test_list=("0008000" "0010000") sgdet_BGNN_bilvl_False_refine_False_seton(True)_gt(True)_consideration_gt(False)_reporting(True)_divide_ratio(1.5)_17_5percent
export test_list=("0008000" "0010000" "0012000" "0014000" "0016000" "0018000")
export test_list=("0030000" "0032000" "0034000" "0036000" "0038000" "0040000" "0042000" "0044000" "0046000" "0048000")
export output_dir="checkpoints/sgcls_GSL4SGG_Predictor_refine_True_loss_type(bce)_filter_valid(False)_input(minus)_residual(True)_vg"
# export test_list=("0012000")
export save_result=False

if $use_multi_gpu;then
    # Multi GPU -sgcls Taskbsgdet_BGNNPredictor_bilvl_True_refine_False_vg_is_rp(False)_rp_thres(0.2)_rp_criteria(max)_gt_(True)_consideration_non_gt_False_rel_rel
    python -m torch.distributed.launch --master_port 10030 --nproc_per_node=$num_gpu tools/relation_test_net.py --config-file "${output_dir}/configs.yml" \
            TEST.IMS_PER_BATCH 1 \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/model_${t}.pth"
else
    for t in ${test_list[@]}
    do
        python  tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
            TEST.IMS_PER_BATCH 6 \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/model_${t}.pth" \
            TEST.SAVE_RESULT ${save_result}
    done
fi