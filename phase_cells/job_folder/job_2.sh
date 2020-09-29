python single_train_v3.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 14 --dim 512 --epoch 200 --lr 5e-4 --dataset live_dead --ext False --train 700 --gpu 2 --loss focal --filters 256 --upsample upsampling --reduce_factor 0.8 --bk 1.0

