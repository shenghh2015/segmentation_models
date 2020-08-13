python phase_fl_train.py --net_type Unet --backbone efficientnetb3 --pre_train False --batch_size 4 --dim 800 --epoch 60 --lr 5e-4 --train 1100 --filtered True --gpu 0 --loss mse --act_fun sigmoid --channels combined --dataset cell_cycle_1984 --ext False

