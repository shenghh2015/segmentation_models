python single_train_v4.py --net_type Unet --backbone efficientnetb1 --pre_train True --batch_size 4 --dim 1024 --epoch 200 --lr 5e-4 --dataset cell_cycle_1984_v2 --ext True --train 1100 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 0.8 --bk 1.0 --focal_weight 4 --feat_version 1

