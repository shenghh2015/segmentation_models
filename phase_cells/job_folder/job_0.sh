python phase_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 800 --epoch 80 --lr 5e-4 --train 1100 --filtered True --gpu 0 --loss mse+focal --act_fun sigmoid --channels combined --dataset cell_cycle_1984_v2 --ext True

