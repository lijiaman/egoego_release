python eval_stage2.py \
--data_root_folder="data" \
--diffusion_window=120 \
--diffusion_batch_size=32 \
--diffusion_project="exp/stage2_motion_diffusion_amass_runs/train" \
--diffusion_exp_name="stage2_cond_motion_diffusion_amass_set1" \
--use_min_max \
--canonicalize_init_head