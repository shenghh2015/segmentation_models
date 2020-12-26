python single_train_v6.py --net_type AtUnet --backbone efficientnetb5 --pre_train True --batch_size 8 --dim 512 --epoch 2400 --lr 5e-4 --dataset live_dead --ext False --train 1100 --gpu 1 --loss focal --filters 256 --upsample upsampling --ext False --reduce_factor 1.0 --bk 1.0

