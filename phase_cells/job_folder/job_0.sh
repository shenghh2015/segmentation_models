python single_train_v2.py --net_type Unet --backbone efficientnetb4 --pre_train True --batch_size 1 --dim 992 --epoch 200 --lr 5e-4 --dataset cell_cycle_1984_v2 --ext True --train 1100 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling

