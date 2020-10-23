# July 30
# JOB: python phase_fl_train.py --net_type Unet --backbone efficientnetb2 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 0
# JOB: python phase_fl_train.py --net_type Unet --backbone efficientnetb3 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 1
# JOB: python phase_fl_train.py --net_type Unet --backbone efficientnetb4 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 8 --rot 20 --lr 5e-4 --pre_train True --gpu 2
# JOB: python phase_fl_train.py --net_type Unet --backbone efficientnetb5 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 3

JOB: python phase_fl_train.py --net_type AtUnet --backbone efficientnetb2 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 0
JOB: python phase_fl_train.py --net_type AtUnet --backbone efficientnetb3 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 1
JOB: python phase_fl_train.py --net_type AtUnet --backbone efficientnetb4 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 2
JOB: python phase_fl_train.py --net_type AtUnet --backbone efficientnetb5 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 3

# JOB: python phase_fl_train.py --net_type Unet --backbone efficientnetb6 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 0
# JOB: python phase_fl_train.py --net_type Unet --backbone efficientnetb7 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 1
# JOB: python phase_fl_train.py --net_type AtUnet --backbone efficientnetb0 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 2
# JOB: python phase_fl_train.py --net_type AtUnet --backbone efficientnetb1 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 14 --rot 20 --lr 5e-4 --pre_train True --gpu 3