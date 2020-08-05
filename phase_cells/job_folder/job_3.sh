python BiFPN_train.py --net_type BiFPN --backbone efficientnetb3 --pre_train True --batch_size 8 --dim 512 --epoch 120 --lr 1e-3 --dataset live_dead --train 900 --gpu 3 --loss focal+dice
