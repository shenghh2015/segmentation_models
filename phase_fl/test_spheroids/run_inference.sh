# python predict_spheroids.py --gpu 0 --model_index 0 --save False
# python predict_spheroids.py --gpu 0 --model_index 1 --save False
# python predict_spheroids.py --gpu 0 --model_index 2 --save False
# python predict_spheroids.py --gpu 2 --model_index 5 --save False --epoch 49
# python predict_spheroids.py --gpu 2 --model_index 4 --save False --epoch 145
# python predict_spheroids.py --gpu 0 --model_index 6 --save True --train False
# python predict_spheroids.py --gpu 2 --model_index 7 --save False --epoch 86
# python predict_spheroids.py --gpu 2 --model_index 8 --save False --epoch 62  ## OK so far
# python predict_spheroids.py --gpu 2 --model_index 9 --save False --epoch 92  ## best so far
python predict_spheroids.py --gpu 2 --model_index 8 --save True --epoch 62  --train True
python predict_spheroids.py --gpu 2 --model_index 9 --save True --epoch 92  --train True