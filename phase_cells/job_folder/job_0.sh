python biFPN_train.py --net_type BiFPN --backbone efficientnetb0 --pre_train True --batch_size 8 --dim 512 --epoch 120 --lr 5e-4 --dataset live_dead --train 900 --gpu 0 --loss focal+dice

