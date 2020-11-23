## Nov. 21, 2020
# compton 
python train_model.py --net_type Unet --backbone efficientnetb3 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3 --docker False
python train_model.py --net_type Unet --backbone efficientnetb3 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 1 --fl_ch fl1 --ch_in 3 --ch_out 3 --docker False
python train_model.py --net_type Unet --backbone efficientnetb3 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 2 --fl_ch fl2 --ch_in 3 --ch_out 3 --docker False

python train_model.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 4 --fl_ch fl12 --ch_in 3 --ch_out 3 --docker False
python train_model.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 5 --fl_ch fl1 --ch_in 3 --ch_out 3 --docker False
python train_model.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 6 --fl_ch fl2 --ch_in 3 --ch_out 3 --docker False

python train_model.py --net_type Unet --backbone efficientnetb0 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 50 --lr 5e-4 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3 --docker False
python train_model.py --net_type Unet --backbone efficientnetb0 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 50 --lr 5e-4 --pre_train True --gpu 1 --fl_ch fl12 --ch_in 3 --ch_out 3 --docker False

python train_model.py --net_type Unet --backbone efficientnetb0 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 50 --lr 5e-4 --pre_train True --gpu 6 --fl_ch fl2 --ch_in 3 --ch_out 3 --docker False

## Nov. 23, 2020
# turing
python train_model.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_v3 --subset train --epoch 800 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 0 --fl_ch fl1 --ch_in 3 --ch_out 3 --docker False
python train_model.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_v3 --subset train --epoch 800 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 7 --fl_ch fl2 --ch_in 3 --ch_out 3 --docker False

# turing
 