python single_train.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 10 --dim 512 --epoch 120 --lr 1e-3 --dataset live_dead --train 900 --gpu 3 --loss focal+dice
