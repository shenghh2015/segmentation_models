# python phase_fl_train.py --net_type Unet --backbone efficientnetb0 --dataset bead_dataset --epoch 200 --dim 512 --loss mae --batch_size 6 --lr 5e-4 --pre_train True --gpu 1
# python phase_fl_train.py --net_type Unet --backbone efficientnetb1 --dataset neuron_x2 --epoch 200 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 1
# python phase_fl_train3.py --net_type Unet --backbone efficientnetb1 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 1 --ch_in 3 --ch_out 1
# python phase_fl_train3.py --net_type Unet --backbone efficientnetb1 --dataset neuron_x2 --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 20 --lr 5e-4 --pre_train True --gpu 1 --ch_in 1 --ch_out 1
# python phase_fl1_fl2_train3.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 1 --fl_ch fl12 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 1 --fl_ch fl12 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-5 --pre_train True --gpu 1 --fl_ch fl12 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb3 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-5 --pre_train True --gpu 1 --fl_ch fl12 --ch_in 3 --ch_out 3
# python phase_fl1_fl2_train2.py --net_type Unet --backbone efficientnetb0 --dataset neuron_trn_tst --subset train --epoch 100 --dim 512 --loss mse --batch_size 6 --rot 50 --lr 1e-4 --pre_train True --gpu 1 --fl_ch fl12 --ch_in 3 --ch_out 3 --best True
# python train_model.py --net_type Unet --backbone efficientnetb0 --dataset spheroids_dataset_x1 --subset train --epoch 10 --dim 320 --loss mse --batch_size 6 --rot 50 --lr 1e-4 --pre_train True --gpu 1 --fl_ch fl2 --ch_in 3 --ch_out 3 --best True --train 100
python train_model.py --net_type Unet --backbone efficientnetb1 --dataset neuron_trn_tst --subset train --epoch 200 --dim 512 --loss mse --batch_size 5 --rot 50 --lr 1e-4 --pre_train True --gpu 1 --fl_ch fl2 --ch_in 3 --ch_out 3 --best_select True