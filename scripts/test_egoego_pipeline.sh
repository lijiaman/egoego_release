python run_egoego.py \
--normal_window=120 \
--normal_n_dec_layers=2 \
--normal_n_head=4 \
--normal_d_k=256 \
--normal_d_v=256 \
--normal_d_model=256 \
--window=60 \
--freeze_of_cnn \
--input_of_feats \
--diffusion_window=120 \
--diffusion_batch_size=32 \
--use_min_max \
--canonicalize_init_head \
--gen_vis 