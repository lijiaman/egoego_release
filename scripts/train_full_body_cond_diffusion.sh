python trainer_amass_cond_motion_diffusion.py \
--data_root_folder="data" \
--window=120 \
--batch_size=32 \
--project="exp/stage2_motion_diffusion_amass_runs/train" \
--exp_name="stage2_cond_motion_diffusion_amass_set1" \
--wandb_pj_name="stage2_cond_motion_diffusion_amass" \
--entity="jiamanli" \
--use_min_max \
--canonicalize_init_head 