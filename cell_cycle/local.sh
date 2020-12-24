# parser = argparse.ArgumentParser()
# parser.add_argument("--docker", type=str2bool, default = True)
# parser.add_argument("--gpu", type=str, default = '2')
# parser.add_argument("--net_type", type=str, default = 'unet1')  #Unet, Linknet, PSPNet, FPN
# parser.add_argument("--backbone", type=str, default = 'xxxx')
# parser.add_argument("--feat_version", type=int, default = None)
# parser.add_argument("--epoch", type=int, default = 2)
# parser.add_argument("--dim", type=int, default = 512)
# parser.add_argument("--batch_size", type=int, default = 4)
# parser.add_argument("--dataset", type=str, default = 'live_dead')
# parser.add_argument("--ext", type=str2bool, default = False)
# parser.add_argument("--upsample", type=str, default = 'upsampling')
# parser.add_argument("--pyramid_agg", type=str, default = 'sum')
# parser.add_argument("--filters", type=int, default = 16)
# parser.add_argument("--rot", type=float, default = 0)
# parser.add_argument("--lr", type=float, default = 1e-3)
# parser.add_argument("--bk", type=float, default = 0.5)
# parser.add_argument("--focal_weight", type=float, default = 1)
# parser.add_argument("--bn", type=str2bool, default = True)
# parser.add_argument("--train", type=int, default = None)
# parser.add_argument("--loss", type=str, default = 'focal+dice')
# parser.add_argument("--reduce_factor", type=float, default = 0.1)
# args = parser.parse_args()
python train_unet.py --gpu 1 --epoch 400 --batch_size 6 --dataset cell_cycle_1984_v2 --ext True --rot 20 --lr 5e-4 --loss focal --bn True --reduce_factor 0.9