python train_neuron.py --net_type Unet --backbone efficientnetb7 --dataset neuron_trn_tst_v2 --subset train --epoch 800 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 2 --fl_ch fl1 --ch_in 3 --ch_out 3 --best_select True --extra False

