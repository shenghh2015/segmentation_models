# Dec. 24, 2020
# JOB: python train_model.py --net_type Unet --backbone efficientnetb0 --pre_train True --batch_size 14 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 
# JOB: python train_model.py --net_type Unet --backbone efficientnetb1 --pre_train True --batch_size 14 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 
# JOB: python train_model.py --net_type Unet --backbone efficientnetb2 --pre_train True --batch_size 14 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 2 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4
# JOB: python train_model.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 14 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 3 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4

JOB: python train_model.py --net_type Unet --backbone efficientnetb4 --pre_train True --batch_size 8 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 
JOB: python train_model.py --net_type Unet --backbone efficientnetb5 --pre_train True --batch_size 8 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 
JOB: python train_model.py --net_type Unet --backbone efficientnetb6 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 2 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4
JOB: python train_model.py --net_type Unet --backbone efficientnetb7 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 3 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4