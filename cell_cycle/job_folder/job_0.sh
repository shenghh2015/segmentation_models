python train_model.py --net_type Unet --backbone efficientnetb0 --pre_train True --batch_size 12 --dim 640 --epoch 2400 --lr 5e-4 --dataset cycle_736x752 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4 

