python trainer_head_estimation.py \
--data_root_folder="data" \
--window=60 \
--batch_size=8 \
--epochs=1000 \
--project="exp/stage1_headnetnet_ares_runs/train" \
--exp_name="stage1_headnet_ares_set1" \
--wandb_pj_name="stage1_headnet_ares" \
--entity="jiamanli" \
--save_interval=50 \
--validation_iter=3 \
--image_save_iter=200 \
--input_of_feats \
--train_on_ares 