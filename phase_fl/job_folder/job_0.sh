python train_model.py --net_type Unet --backbone efficientnetb7 --dataset spheroids_v3 --subset train --epoch 800 --dim 512 --loss mse --batch_size 4 --rot 50 --lr 1e-4 --pre_train True --gpu 0 --fl_ch fl12 --ch_in 3 --ch_out 3 --best_select True --scale 10.0

