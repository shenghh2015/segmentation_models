## July 2, 2020
# JOB: python live_dead.py --net_type FPN --backbone efficientnetb2 --pre_train True --batch_size 10 --epoch 300 --lr 5e-4 --gpu 0
# JOB: python live_dead.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 10 --epoch 300 --lr 5e-4 --gpu 1
# JOB: python live_dead.py --net_type FPN --backbone efficientnetb4 --pre_train True --batch_size 10 --epoch 300 --lr 5e-4 --gpu 2
# JOB: python live_dead.py --net_type FPN --backbone efficientnetb6 --pre_train True --batch_size 6 --epoch 300 --lr 5e-4 --gpu 3

# JOB: python live_dead.py --net_type Unet --backbone efficientnetb2 --pre_train True --batch_size 14 --epoch 300 --lr 5e-4 --gpu 0
# JOB: python live_dead.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 14 --epoch 300 --lr 5e-4 --gpu 1
# JOB: python live_dead.py --net_type Unet --backbone efficientnetb4 --pre_train True --batch_size 14 --epoch 300 --lr 5e-4 --gpu 2
# JOB: python live_dead.py --net_type Unet --backbone efficientnetb6 --pre_train True --batch_size 10 --epoch 300 --lr 5e-4 --gpu 3

JOB: python cell_cycle.py --net_type FPN --backbone efficientnetb2 --pre_train True --batch_size 10 --epoch 300 --lr 5e-4 --gpu 0
JOB: python cell_cycle.py --net_type FPN --backbone efficientnetb3 --pre_train True --batch_size 10 --epoch 300 --lr 5e-4 --gpu 1
JOB: python cell_cycle.py --net_type FPN --backbone efficientnetb4 --pre_train True --batch_size 10 --epoch 300 --lr 5e-4 --gpu 2
JOB: python cell_cycle.py --net_type FPN --backbone efficientnetb6 --pre_train True --batch_size 6 --epoch 300 --lr 5e-4 --gpu 3

# JOB: python cell_cycle.py --net_type Unet --backbone efficientnetb2 --pre_train True --batch_size 14 --epoch 300 --lr 5e-4 --gpu 0
# JOB: python cell_cycle.py --net_type Unet --backbone efficientnetb3 --pre_train True --batch_size 14 --epoch 300 --lr 5e-4 --gpu 1
# JOB: python cell_cycle.py --net_type Unet --backbone efficientnetb4 --pre_train True --batch_size 14 --epoch 300 --lr 5e-4 --gpu 2
# JOB: python cell_cycle.py --net_type Unet --backbone efficientnetb6 --pre_train True --batch_size 10 --epoch 300 --lr 5e-4 --gpu 3