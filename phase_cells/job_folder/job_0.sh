python single_train_v3.py --net_type Nestnet --backbone efficientnetb0 --pre_train True --batch_size 2 --dim 1024 --epoch 400 --lr 5e-4 --dataset cell_cycle_1984_v2 --ext True --train 1100 --gpu 0 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0

