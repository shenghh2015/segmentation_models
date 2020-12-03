# parser = argparse.ArgumentParser()
# parser.add_argument("--gpu", type=str, default = '0')
# parser.add_argument("--docker", type=str2bool, default = True)
# parser.add_argument("--dataset", type=str, default = 'spheroids_x1')
# parser.add_argument("--filters", type=int, default = 16)
# parser.add_argument("--levels", type=int, default = 5)
# parser.add_argument("--dim", type=int, default = 256)
# parser.add_argument("--dep", type=int, default = 16)
# parser.add_argument("--val_dim", type=int, default = 256)
# parser.add_argument("--val_dep", type=int, default = 32)
# parser.add_argument("--epoch", type=int, default = 10)
# parser.add_argument("--batch_size", type=int, default = 3)
# parser.add_argument("--lr", type=float, default = 5e-6)
# parser.add_argument("--scale", type=float, default = 1.0)
# parser.add_argument("--decay", type=float, default = 0.8)
# args = parser.parse_args()

## Nov 30, 2020
python train_3D.py --docker False --filters 16 --levels 5 --dim 512 --dep 16 --val_dim 512 --val_dep 32 --epoch 10000 --batch_size 3 --lr 5e-6 --gpu 7
python train_3D.py --docker False --filters 16 --levels 5 --dim 512 --dep 16 --val_dim 512 --val_dep 32 --epoch 10000 --batch_size 3 --lr 5e-6 --gpu 6 --scale 100
python train_3D.py --docker False --filters 16 --levels 5 --dim 512 --dep 16 --val_dim 512 --val_dep 48 --epoch 10000 --batch_size 3 --lr 1e-5 --gpu 5 --scale 10

## Dec. 1, 2020
python train_3D.py --docker False --filters 16 --levels 5 --dim 512 --dep 16 --val_dim 512 --val_dep 48 --epoch 10000 --batch_size 3 --lr 1e-6 --gpu 7 --scale 10
python train_3D.py --docker False --filters 32 --levels 5 --dim 256 --dep 16 --val_dim 256 --val_dep 48 --epoch 10000 --batch_size 3 --lr 1e-6 --gpu 4 --scale 10
python train_DS3D.py --docker False --filters 16 --levels 5 --dim 512 --dep 16 --val_dim 512 --val_dep 16 --epoch 10000 --batch_size 1 --lr 5e-6 --gpu 6 --scale 10

## Dec. 4, 2020
python train_VNet.py --docker False --dim 512 --dep 16 --val_dim 512 --val_dep 32 --epoch 10000 --batch_size 2 --lr 1e-5 --gpu 7 --scale 1
python train_VNet.py --docker False --dim 512 --dep 16 --val_dim 512 --val_dep 32 --epoch 10000 --batch_size 2 --lr 1e-5 --gpu 6 --scale 100.
python train_3D.py --docker False --filters 16 --levels 6 --dim 256 --dep 16 --val_dim 256 --val_dep 48 --epoch 10000 --batch_size 3 --lr 1e-5 --gpu 5 --scale 10
python train_3D.py --docker False --filters 16 --levels 6 --dim 256 --dep 16 --val_dim 256 --val_dep 48 --epoch 10000 --batch_size 3 --lr 1e-5 --gpu 4 --scale 100.