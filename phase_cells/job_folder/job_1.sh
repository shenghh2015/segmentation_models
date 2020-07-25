python live_dead.py --net_type Unet --backbone efficientnetb2 --pre_train True --batch_size 14 --dim 512 --epoch 100 --lr 5e-4 --dataset live_dead --down_factor 1 --train 900 --class_balanced False --gpu 1 --frozen True --loss focal+dice

