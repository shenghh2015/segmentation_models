python train_model.py --net_type Unet --backbone efficientnetb6 --pre_train True --batch_size 3 --dim 800 --epoch 4800 --lr 1e-4 --dataset cyc2_1488x1512 --gpu 2 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --bk 1.0 --focal_weight 4

