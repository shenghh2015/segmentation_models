python biFPN_train.py --net_type BiFPN --backbone efficientnetb2 --pre_train True --batch_size 3 --dim 896 --epoch 120 --lr 5e-4 --dataset cell_cycle_1984_v2 --train 900 --gpu 2 --loss focal+dice

