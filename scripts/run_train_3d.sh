python -m pdb train_3d.py \
    --exp_suffix train_3d_robotiq_116019 \
    --model_version model_3d \
    --primact_type grasp \
    --data_dir_prefix datasets/gt_data \
    --offline_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_robotiq \
    --val_data_dir /media/zhou/软件/博士/具身/grasp_train_data_gspwth_robotiq \
    --val_data_fn data_tuple_list.txt \
    --train_shape_fn datasets/grasp_stats/data_small.txt \
    --ins_cnt_fn datasets/grasp_stats/ins_cnt_15cats.txt \
    --buffer_max_num 200 \
    --num_processes_for_datagen 5 \
    --num_interaction_data_offline 1 \
    --num_interaction_data 1 \
    --sample_succ \
    --pretrained_critic_ckpt logs/exp-model_3d_critic-train_3d_critic_robotiq/ckpts/165-network.pth \
    --resume

