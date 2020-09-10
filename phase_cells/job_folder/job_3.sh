python single_train_v2.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --dim 992 --epoch 200 --lr 5e-4 --dataset cell_cycle_1984 --ext False --train 1100 --gpu 3 --loss focal+dice --filters 256 --upsample upsampling

