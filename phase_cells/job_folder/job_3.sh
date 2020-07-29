python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 1024 --down_factor 1 --epoch 150 --dataset cell_cycle_1984_v2 --lr 5e-4 --train 1100 --gpu 3 --loss focal+jaccard

