python single_train.py --net_type Unet --backbone efficientnetb2 --pre_train True --batch_size 4 --dim 800 --epoch 150 --lr 5e-4 --dataset cell_cycle_1984_v2 --train 900 --gpu 2 --loss focal+dice --filters 512 --upsample transpose

