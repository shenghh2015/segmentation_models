python single_train_v3.py --net_type Unet --backbone efficientnetb2 --pre_train True --batch_size 3 --dim 1024 --epoch 300 --lr 5e-4 --dataset cell_cycle_1984_v2 --ext True --train 1100 --gpu 2 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 0.8 --bk 1.0 --focal_weight 4

