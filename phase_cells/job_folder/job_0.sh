python single_train_v3.py --net_type Unet --backbone efficientnetb2 --pre_train True --batch_size 14 --dim 512 --epoch 400 --lr 5e-4 --dataset live_dead --ext False --train 900 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 0.8

