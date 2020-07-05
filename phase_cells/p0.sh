## June 18
# python live_dead.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 8 --dim 320 --epoch 600 --lr 5e-4 --gpu 0

python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --epoch 5 --lr 5e-4 --train 100 --gpu 0