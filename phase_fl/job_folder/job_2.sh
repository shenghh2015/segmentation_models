python phase_fl1_fl2_train3.py --net_type Unet --backbone efficientnetb7 --dataset spheroids_dataset_x1 --subset train --epoch 100 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 5e-4 --pre_train True --gpu 2 --fl_ch fl12 --ch_in 3 --ch_out 3

