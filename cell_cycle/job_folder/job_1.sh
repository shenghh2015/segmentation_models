python train_viability.py --net_type Unet --backbone efficientnetb1 --pre_train True --batch_size 4 --dim 800 --epoch 2400 --lr 1e-4 --dataset viability2_832x832 --data_version 3 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 

