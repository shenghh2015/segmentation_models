python train_model.py --net_type Unet --backbone efficientnetb7 --dataset neuron_trn_tst --subset train --epoch 200 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 1 --fl_ch fl12 --ch_in 3 --ch_out 3 --best False

