python single_train_v5.py --net_type Unet --backbone efficientnetb5 --pre_train True --batch_size 4 --dim 800 --epoch 2400 --lr 5e-3 --dataset cell_cycle_1984_v2 --ext True --train 1100 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --ext True --reduce_factor 1.0 --bk 1.0 --focal_weight 4 --feat_version 1

