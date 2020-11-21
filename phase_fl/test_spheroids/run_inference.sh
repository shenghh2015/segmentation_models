# python predict_spheroids.py --gpu 0 --model_index 0 --save False
# python predict_spheroids.py --gpu 0 --model_index 1 --save False
# python predict_spheroids.py --gpu 0 --model_index 2 --save False
python predict_spheroids.py --gpu 2 --model_index 3 --save False --epoch 150
python predict_spheroids.py --gpu 2 --model_index 4 --save False --epoch 150
python predict_spheroids.py --gpu 2 --model_index 3 --save False --epoch 140
python predict_spheroids.py --gpu 2 --model_index 4 --save False --epoch 140
python predict_spheroids.py --gpu 2 --model_index 3 --save False --epoch 130
python predict_spheroids.py --gpu 2 --model_index 4 --save False --epoch 130
python predict_spheroids.py --gpu 2 --model_index 3 --save False --epoch 120
python predict_spheroids.py --gpu 2 --model_index 4 --save False --epoch 120
