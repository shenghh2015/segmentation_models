python single_train_v5.py --net_type AtUnet --backbone efficientnetb5 --pre_train True --batch_size 4 --dim 800 --epoch 2400 --lr 1e-4 --dataset live_dead --ext False --train 1100 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --ext False --reduce_factor 1.0 --bk 1.0 --focal_weight 4 

