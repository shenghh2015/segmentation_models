python single_train_v3.py --net_type FPN --backbone efficientnetb1 --pre_train True --batch_size 8 --dim 512 --epoch 400 --lr 5e-4 --dataset live_dead --ext False --train 900 --gpu 1 --loss focal+dice --filters 256 --upsample upsampling --reduce_factor 1.0 --pyramid_agg sum

