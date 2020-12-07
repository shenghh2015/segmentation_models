# python phase_fl_train.py --net_type Unet --backbone efficientnetb0 --dataset bead_dataset_v2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 0
# python phase_fl_train.py --net_type Unet --backbone efficientnetb0 --dataset bead_dataset_v2 --epoch 100 --dim 320 --loss mse --batch_size 6 --lr 5e-4 --pre_train True --gpu 0
# python phase_fl_train.py --net_type Unet --backbone efficientnetb0 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 0
# python phase_fl_train3.py --net_type Unet --backbone efficientnetb0 --dataset neuron_x2 --epoch 1 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 0 --ch_in 3 --ch_out 1
# python phase_fl_train3.py --net_type Unet --backbone efficientnetb0 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 0 --ch_in 3 --ch_out 1
# python phase_fl_train3.py --net_type Unet --backbone efficientnetb0 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 0 --ch_in 1 --ch_out 1
# python phase_fl_train3.py --net_type Unet --backbone efficientnetb0 --dataset neuron_x2 --epoch 1 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 1 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train.py --net_type Unet --backbone efficientnetb7 --dataset neuron_wbx1 --subset train --epoch 1 --dim 512 --loss mse --batch_size 1 --rot 0 --lr 1e-4 --pre_train True --gpu 0 --fl_ch fl12
# python phase_fl1_fl2_train.py --net_type Unet --backbone efficientnetb7 --dataset neuron_wbx1 --subset train --epoch 1 --dim 512 --loss mse --batch_size 1 --rot 0 --lr 1e-4 --pre_train True --gpu 0 --fl_ch fl1
# python phase_fl1_fl2_train.py --net_type Unet --backbone efficientnetb7 --dataset neuron_wbx1 --subset train --epoch 1 --dim 512 --loss mse --batch_size 1 --rot 0 --lr 1e-4 --pre_train True --gpu 0 --fl_ch fl2
# python phase_fl1_fl2_train3.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb0 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 50 --lr 1e-4 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb0 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 50 --lr 5e-5 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb0 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 50 --lr 1e-5 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3
# python parallel_phase_fl1_fl2_train.py --net_type Unet --backbone efficientnetb7 --dataset neuron_trn_tst --subset train --epoch 100 --dim 416 --loss mse --batch_size 1 --rot 50 --lr 5e-5 --pre_train True --gpu 0,1 --fl_ch fl12 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb3 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3
# python train_model.py --net_type Unet --backbone efficientnetb0 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 2 --fl_ch fl12 --ch_in 3 --ch_out 3
# python train_neuron.py --net_type Unet --backbone efficientnetb0 --dataset neuron_trn_tst_v2 --subset train --epoch 2 --dim 320 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 2 --fl_ch fl1 --ch_in 3 --ch_out 3
# python train_neuron.py --net_type Unet --backbone efficientnetb3 --dataset neuron_trn_tst_v2 --subset train --epoch 400 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3

python train_unet.py --net_type unet1 --backbone xxx --dataset spheroids_v4 --subset train --epoch 400 --dim 512 --loss mse --batch_size 6 --rot 50 --lr 1e-4 --bn True --gpu 0 --fl_ch fl1 --ch_in 3 --ch_out 3