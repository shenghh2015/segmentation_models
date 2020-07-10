# python cell_cycle.py --net_type FPN --backbone efficientnetb2 --pre_train True --batch_size 8 --dim 320 --epoch 600 --lr 5e-4 --gpu 1
# python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --train 100 --epoch 5 --lr 5e-4 --gpu 1

# python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 5 --dim 512 --epoch 200 --lr 5e-4 --train 1100 --bk_weight 0.7 --gpu 1
# python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 512 --epoch 300 --lr 1e-4 --train 1100 --gpu 1
python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 2 --epoch 300 --lr 5e-4 --dataset live_dead_1664 --train 900 --dim 800 --rot 0 --gpu 1
