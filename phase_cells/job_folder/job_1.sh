python phase_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 800 --epoch 60 --lr 5e-4 --train 1100 --filtered True --gpu 1 --loss mse --act_fun sigmoid --channels fl2 --dataset cell_cycle_1984_v2 --ext False

