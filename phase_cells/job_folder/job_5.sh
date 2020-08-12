python single_train.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 10 --dim 512 --epoch 150 --lr 5e-4 --dataset live_dead --train 900 --gpu 1 --loss focal+dice --filters 512 --upsample upsampling

