python single_train_v3.py --net_type Unet --backbone resnet34 --pre_train True --batch_size 4 --dim 1024 --epoch 200 --lr 5e-4 --dataset cell_cycle_1984_v2 --ext True --train 1100 --gpu 1 --loss focal --filters 256 --upsample upsampling --reduce_factor 0.8

