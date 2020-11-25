# python predict_spheroids.py --gpu 0 --model_index 0 --save False
# python predict_spheroids.py --gpu 0 --model_index 1 --save False
# python predict_spheroids.py --gpu 0 --model_index 2 --save False
# python predict_spheroids.py --gpu 2 --model_index 5 --save False --epoch 49
# python predict_spheroids.py --gpu 2 --model_index 4 --save False --epoch 145
# python predict_spheroids.py --gpu 0 --model_index 6 --save True --train False
# python predict_spheroids.py --gpu 2 --model_index 7 --save False --epoch 86
# python predict_spheroids.py --gpu 2 --model_index 8 --save False --epoch 62  ## OK so far
# python predict_spheroids.py --gpu 2 --model_index 9 --save False --epoch 92  ## best so far
# python predict_spheroids.py --gpu 2 --model_index 8 --save True --epoch 62  --train True
# python predict_spheroids.py --gpu 2 --model_index 9 --save True --epoch 92  --train True
# python predict_spheroids.py --gpu 2 --model_index 5 --save False --epoch 86
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 0 --save False --epoch 25
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 1 --save False --epoch 52
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 2 --save False --epoch 11
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 3 --save False --epoch 51
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 4 --save False --epoch 34
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 1 --save False --epoch 52

# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 0 --save False --epoch 42
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 1 --save False --epoch 83
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 3 --save False --epoch 63
# python predict_spheroids.py --model_file model_list2.txt --gpu 2 --model_index 5 --save False --epoch 84

## check results
# python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 0 --save False --epoch 102
# python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 1 --save False --epoch 142
# python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 2 --save False --epoch 75
# python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 3 --save False --epoch 150
# python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 4 --save False --epoch 64
# python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 1 --save False --epoch 171

## generate results
python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 2 --save True --train True --epoch 75
python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 3 --save True --train True --epoch 150
# python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 1 --save True --train True --epoch 83
# python predict_spheroids.py --model_file model_list2.txt --gpu 0 --model_index 3  --save True --train True --epoch 63