# July 30
# JOB: python cell_cycle.py --net_type Unet --backbone efficientnetb0 --pre_train True --batch_size 3 --dim 1024 --down_factor 1 --ext True --epoch 120 --dataset cell_cycle_1984_v2 --lr 5e-4 --train 1100 --gpu 0 --loss focal+dice
# JOB: python cell_cycle.py --net_type Unet --backbone efficientnetb1 --pre_train True --batch_size 3 --dim 1024 --down_factor 1 --ext True --epoch 120 --dataset cell_cycle_1984_v2 --lr 5e-4 --train 1100 --gpu 1 --loss focal+dice
# JOB: python cell_cycle.py --net_type Unet --backbone efficientnetb2 --pre_train True --batch_size 3 --dim 1024 --down_factor 1 --ext True --epoch 120 --dataset cell_cycle_1984_v2 --lr 5e-4 --train 1100 --gpu 2 --loss focal+dice
# JOB: python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 1024 --down_factor 1 --ext True --epoch 120 --dataset cell_cycle_1984_v2 --lr 5e-4 --train 1100 --gpu 3 --loss focal+dice
 
python life_cycle_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 1024 --epoch 100 --lr 5e-4 --train 1100 --filtered True --gpu 0 --loss mse --act_fun sigmoid --channels combined --ext True --dataset cell_cycle_1984_v2
python life_cycle_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 1024 --epoch 100 --lr 5e-4 --train 1100 --filtered True --gpu 1 --loss mse --act_fun relu --channels combined --ext True --dataset cell_cycle_1984_v2
python life_cycle_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 1024 --epoch 100 --lr 5e-4 --train 1100 --filtered True --gpu 2 --loss mse --act_fun sigmoid --channels combined --ext False --dataset cell_cycle_1984_v2
python life_cycle_flu.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 1024 --epoch 100 --lr 5e-4 --train 1100 --filtered True --gpu 3 --loss mse --act_fun relu --channels combined --ext False --dataset cell_cycle_1984_v2