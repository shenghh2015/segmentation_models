python single_train.py --net_type Unet --backbone efficientnetb0 --pre_train True --batch_size 6 --dim 800 --epoch 150 --lr 5e-4 --dataset live_dead --train 900 --gpu 2 --loss focal+dice --filters 512 --upsample transpose

