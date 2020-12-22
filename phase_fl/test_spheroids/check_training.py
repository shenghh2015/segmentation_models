import os
import cv2
from skimage import io
import sys
# import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
sys.path.append('../')
import segmentation_models_v1 as sm
from segmentation_models_v1 import Unet, Linknet, PSPNet, FPN, AtUnet, ResUnet
sm.set_framework('tf.keras')

from helper_function import plot_history_flu2, save_phase_fl_history, plot_flu_prediction, plot_set_prediction
from helper_function import save_history_for_callback, plot_history_for_callback
from helper_function import precision, recall, f1_score, calculate_psnr, calculate_pearsonr
from sklearn.metrics import confusion_matrix

def str2bool(value):
    return value.lower() == 'true'

def generate_folder(folder_name):
	if not os.path.exists(folder_name):
		os.system('mkdir -p {}'.format(folder_name))


model_name = 'Cor-FL1_FL2-net-Unet-bone-efficientnetb7-pre-True-epoch-800-batch-6-lr-0.0005-dim-512-train-None-rot-50.0-set-spheroids_v5-subset-train-loss-mse-act-relu-scale-50.0-decay-0.8-delta-10-chi-3-cho-1-chf-fl2-bselect-True-Scr-extra-True'

splits = model_name.split('-')

for v in range(len(splits)):
	if splits[v] == 'net':
		net_arch = splits[v+1]
	elif splits[v] == 'bone':
		backbone = splits[v+1]
	elif splits[v] == 'scale':
		scale = float(splits[v+1])
	elif splits[v] == 'act':
		act_fun = splits[v+1]
	elif splits[v] == 'set':
		dataset = splits[v+1]
	elif splits[v] == 'subset':
		subset = splits[v+1]
	elif splits[v] == 'chi':
		chi = int(splits[v+1])
	elif splits[v] == 'cho':
		cho = int(splits[v+1])
	elif splits[v] == 'chf':
		fl_ch = splits[v+1]
	elif splits[v] == 'best':
		best_flag = str2bool(splits[v+1])
	elif splits[v] == 'loss':
		loss_name = splits[v+1]
	elif splits[v] == 'dim':
		dim = int(splits[v+1])
	
model_dir = '/data/2d_models/{}'.format(dataset)

# parser = argparse.ArgumentParser()
# parser.add_argument("--gpu", type=str, default = '2')
# parser.add_argument("--docker", type=str2bool, default = True)
# parser.add_argument("--net_type", type=str, default = 'Unet')  #Unet, Linknet, PSPNet, FPN
# parser.add_argument("--backbone", type=str, default = 'efficientnetb0')
# parser.add_argument("--dataset", type=str, default = 'neuron_wbx1')
# parser.add_argument("--subset", type=str, default = 'train')
# parser.add_argument("--extra", type=str2bool, default = True)
# parser.add_argument("--epoch", type=int, default = 10)
# parser.add_argument("--dim", type=int, default = 512)
# parser.add_argument("--ch_in", type=int, default = 3)
# parser.add_argument("--ch_out", type=int, default = 3)
# parser.add_argument("--fl_ch", type=str, default = 'fl12')
# parser.add_argument("--rot", type=float, default = 0)
# parser.add_argument("--scale", type=float, default = 100)
# parser.add_argument("--train", type=int, default = None)
# parser.add_argument("--act_fun", type=str, default = 'relu')
# parser.add_argument("--loss", type=str, default = 'mse')
# parser.add_argument("--batch_size", type=int, default = 6)
# parser.add_argument("--lr", type=float, default = 5e-4)
# parser.add_argument("--decay", type=float, default = 0.8)
# parser.add_argument("--delta", type=float, default = 10)
# parser.add_argument("--best_select", type=str2bool, default = True)  ## cancel the selection of best model
# parser.add_argument("--pre_train", type=str2bool, default = True)
# args = parser.parse_args()
# print(args)

## screen the fl1
# model_name = 'Cor-FL1_FL2-net-{}-bone-{}-pre-{}-epoch-{}-batch-{}-lr-{}-dim-{}-train-{}-rot-{}-set-{}-subset-{}-loss-{}-act-{}-scale-{}-decay-{}-delta-{}-chi-{}-cho-{}-chf-{}-bselect-{}-Scr-extra-{}'.format(net_type, backbone, pre_train,\
# 		 epoch, batch_size, lr, dim, train, rot, dataset, subset, loss, act_fun, scale, decay, delta, ch_in, ch_out, fl_ch, best_select, extra)
# print(model_name)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
docker = True

DATA_DIR = '/data/datasets/{}'.format(dataset) if docker else './datasets/{}'.format(dataset)
train_dim = dim

x_train_dir = os.path.join(DATA_DIR, '{}/phase'.format(subset))
y1_train_dir = os.path.join(DATA_DIR, '{}/fl1'.format(subset))
y2_train_dir = os.path.join(DATA_DIR, '{}/fl2'.format(subset))

x_valid_dir = os.path.join(DATA_DIR, 'test/phase')
y1_valid_dir = os.path.join(DATA_DIR, 'test/fl1')
y2_valid_dir = os.path.join(DATA_DIR, 'test/fl2')

x_test_dir = os.path.join(DATA_DIR, 'test/phase')
y1_test_dir = os.path.join(DATA_DIR, 'test/fl1')
y2_test_dir = os.path.join(DATA_DIR, 'test/fl2')

val_dim = 896 if 'x2' in dataset else 1792 
# if dataset == 'neuron_wbx1' or dataset == 'neuron_trn_tst':
# 	val_dim = 1792
# elif dataset == 'spheroids_dataset_x1':
# 	val_dim = 1792
# elif dataset == 'neuron_wbx2':
# 	val_dim = 896

print(x_train_dir)
print(y1_train_dir)
print(y2_train_dir)

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir1,
            masks_dir2,
            extra = True,
            fl_ch = None,
            scale = 1.0,
            channels = [3,3],
            nb_data=None,
            augmentation=None, 
            preprocessing=None,
    ):
        #self.ids = os.listdir(images_dir)
        id_list = natsorted(os.listdir(images_dir))
        if (not fl_ch == 'fl2'):
        		screen_set = ['T-33', 'T-34', 'T-35', 'T-36', 'T-37', 'T-38', 
        									'T-39']
        		id_list = [id for id in id_list if not 'T-{}'.format(id.split('-')[1]) in screen_set]
        if not extra:
        		screen_set = ['T-33', 'T-34', 'T-35', 'T-36', 'T-37', 'T-38', 
        									'T-39', 'T-40', 'T-41','T-42', 'T-43','T-44']
        		id_list = [id for id in id_list if not 'T-{}'.format(id.split('-')[1]) in screen_set]        
        if nb_data ==None:
            self.ids = id_list
        else:
            self.ids = id_list[:int(min(nb_data,len(id_list)))]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks1_fps = [os.path.join(masks_dir1, image_id) for image_id in self.ids]
        self.masks2_fps = [os.path.join(masks_dir2, image_id) for image_id in self.ids]
        print('Load files: image {}, fl1: {}, fl2:{}'.format(len(self.images_fps),len(self.masks1_fps),len(self.masks2_fps)))      
        self.scale = scale
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.channels = channels
        self.fl_ch = fl_ch
    
    def __getitem__(self, i):
        
        # load image and fl1 or fl2 or both
        image = io.imread(self.images_fps[i])
        if self.fl_ch == 'fl1':
            mask  = io.imread(self.masks1_fps[i])
            mask = mask/255.*self.scale
        elif self.fl_ch == 'fl2':
            mask  = io.imread(self.masks2_fps[i])
            mask = mask/255.*self.scale        
        elif self.fl_ch == 'fl12':
            mask1  = io.imread(self.masks1_fps[i])
            mask2  = io.imread(self.masks2_fps[i])
            mask = np.stack([mask1[:,:,1], mask2[:,:,1]], axis = -1)
            mask = mask/255.*self.scale
        
        # decide the input and output channels
        if self.channels[0] == 1:
            image[:,:,0], image[:,:,2] = image[:,:,1], image[:,:,1]
        elif self.channels[0] == 2:
            image[:,:,2] = image[:,:,1]		
        
        if self.channels[1] == 1 and not (self.fl_ch=='fl12'):
            mask = mask[:,:,1:2]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
        
    def __len__(self):
        return len(self.ids)
    

class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return (batch[0], batch[1])
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

import albumentations as A

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(dim, rot = 0):
    train_transform = [

        A.HorizontalFlip(p=0.5),
# 
#         A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=rot, shift_limit=0.1, p=1, border_mode=0),
# 
        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
        A.RandomCrop(height=dim, width=dim, always_apply=True),
# 
#         A.IAAAdditiveGaussianNoise(p=0.2),
#         A.IAAPerspective(p=0.5),
# 
#         A.OneOf(
#             [
#                 A.CLAHE(p=1),
#                 A.RandomBrightness(p=1),
#                 A.RandomGamma(p=1),
#             ],
#             p=0.9,
#         ),
# 
#         A.OneOf(
#             [
#                 A.IAASharpen(p=1),
#                 A.Blur(blur_limit=3, p=1),
#                 A.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.9,
#         ),
# 
#         A.OneOf(
#             [
#                 A.RandomContrast(p=1),
#                 A.HueSaturationValue(p=1),
#             ],
#             p=0.9,
#         ),
#         A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(dim = 992):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(dim, dim)
#         A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

## create models
BACKBONE = backbone
BATCH_SIZE = batch_size
LR = lr
EPOCHS = epoch

# processing configuration
preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = cho if fl_ch == 'fl1' or fl_ch == 'fl2' else 2
activation = '{}'.format(act_fun)

#create model
net_func = globals()[net_type]

# encoder_weights='imagenet' if pre_train else None

encoder_weights = None
model = net_func(BACKBONE, encoder_weights=encoder_weights, classes=n_classes, activation=activation)

# load the weights
weight_file = os.path.join(model_dir, model_name, 'best_model-{}.h5'.format(epoch))
model.load_weights(weight_file)

# define optomizer
optim = tf.keras.optimizers.Adam(LR)

if loss_name == 'mse':
	loss = tf.keras.losses.MSE
elif loss_name == 'mae':
	loss = tf.keras.losses.MAE
elif loss_name == 'huber':
	loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

from tensorflow.keras import backend as K
def pearson(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    return r

metrics = [sm.metrics.PSNR(max_val=scale), pearson]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, loss, metrics)

# Dataset for train images
train = None
extra = True
rot = 50
train_dataset = Dataset(
    x_train_dir, 
    y1_train_dir,
    y2_train_dir,
    extra = extra,
    fl_ch = fl_ch,
    channels = [chi, cho],
    scale = scale,
    nb_data=train, 
    augmentation=get_training_augmentation(train_dim, rot),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y1_valid_dir,
    y2_valid_dir,
    extra = extra,
    fl_ch = fl_ch,
    scale = scale,
    channels = [chi, cho],
    augmentation=get_validation_augmentation(val_dim),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

print(train_dataloader[0][0].shape)
print(train_dataloader[0][1].shape)
# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, train_dim, train_dim, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, train_dim, train_dim, n_classes)

model_folder = '/data/2d_models/{}/{}'.format(dataset, model_name) if docker else './2d_models/{}/{}'.format(dataset, model_name)
generate_folder(model_folder)

# model evaluation
# [8.248620156122714, 25.8897, 0.24172623]

def calculate_pearsonr(vol1, vol2):
	SMOOTH_ = 1e-6
	numerator = np.sum((vol1-vol1.mean())*(vol2-vol2.mean()))
	denominator = np.linalg.norm(vol1-vol1.mean())*np.linalg.norm(vol2-vol2.mean())
	return (numerator+SMOOTH_)/(denominator+SMOOTH_)

def calculate_psnr(gt_vol,pr_vol):
	gt_max, gt_min = gt_vol.max(), gt_vol.min()
	pr_max, pr_min = pr_vol.max(), pr_vol.min()
	mse = np.mean(np.square(gt_vol-pr_vol))
	psnr_score = 20*np.log10(255/np.sqrt(mse))
	return psnr_score

def calculate_mse(vol1, vol2):
	return np.mean(np.square(vol1-vol2))

# prediction
val_loss = []; 
gts = []; prs = []
for i in range(len(valid_dataloader)):
		img_input = valid_dataloader[i][0]
		prs.append(model.predict(valid_dataloader[i]))
		gts.append(valid_dataloader[i][1])

gts_arr = np.concatenate(gts,axis =0)
prs_arr = np.concatenate(prs, axis =0)

cor_score = calculate_pearsonr(gts_arr, prs_arr)
mse_score = calculate_mse(gts_arr, prs_arr)
psnr_score = calculate_psnr(gts_arr, prs_arr)

print('mMSE {:4f}, mCor:{:.4f}, mPSNR {:.4f}'.format(mse_score, cor_score, psnr_score))

# scale back and compute
gts_arr_1 = gts_arr/scale*255
prs_arr_1 = prs_arr/scale*255
cor_score_1 = calculate_pearsonr(gts_arr_1, prs_arr_1)
mse_score_1 = calculate_mse(gts_arr_1, prs_arr_1)
psnr_score_1 = calculate_psnr(gts_arr_1, prs_arr_1)

## mse
mse_scores = [calculate_mse(gts_arr_1[i,:], prs_arr_1[i,:]) for i in range(prs_arr_1.shape[0])]
cor_scores = [calculate_pearsonr(gts_arr_1[i,:], prs_arr_1[i,:]) for i in range(prs_arr_1.shape[0])]
psnr_scores = [calculate_psnr(gts_arr_1[i,:], prs_arr_1[i,:]) for i in range(prs_arr_1.shape[0])]

print('mMSE {:4f}, mCor:{:.4f}, mPSNR {:.4f}'.format(mse_score_1, cor_score_1, psnr_score_1))

mse_scores = [calculate_mse(gts_arr[i,:], prs_arr[i,:]) for i in range(prs_arr.shape[0])]
cor_scores = [calculate_pearsonr(gts_arr[i,:], prs_arr[i,:]) for i in range(prs_arr.shape[0])]
psnr_scores = [calculate_psnr(gts_arr[i,:], prs_arr[i,:]) for i in range(prs_arr.shape[0])]

print('mMSE {:4f}, mCor:{:.4f}, mPSNR {:.4f}'.format(mse_score, cor_score_1, psnr_score_1))

offset = 24
mse_scores = [calculate_mse(gts_arr[i,offset:-offset,offset:-offset], prs_arr[i,offset:-offset,offset:-offset]) for i in range(prs_arr.shape[0])]
cor_scores = [calculate_pearsonr(gts_arr[i,offset:-offset,offset:-offset], prs_arr[i,offset:-offset,offset:-offset]) for i in range(prs_arr.shape[0])]
psnr_scores = [calculate_psnr(gts_arr[i,offset:-offset,offset:-offset], prs_arr[i,offset:-offset,offset:-offset]) for i in range(prs_arr.shape[0])]

print('mMSE {:4f}, mCor:{:.4f}, mPSNR {:.4f}'.format(np.mean(mse_scores), np.mean(cor_scores), np.mean(psnr_scores)))

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def save_images(file_name, vols):
		vols = vols[:,:,:,1] if vols.shape[-1] >= 2 else vols[:,:,:,0]
		shp = vols.shape
		ls, lx, ly = shp
		sx, sy = int(lx/128), int(ly/128)
		vols = vols[:,::sx,::sy]
		slice_list, rows = [], []
		for si in range(vols.shape[0]):
				slice = vols[si,:,:]
				rows.append(slice)
				if si%8 == 7 and not si == vols.shape[0]-1:
						slice_list.append(rows)
						rows = []
		save_img = concat_tile(slice_list)		
		cv2.imwrite(file_name, save_img)

class HistoryPrintCallback(tf.keras.callbacks.Callback):
		def __init__(self):
				super(HistoryPrintCallback, self).__init__()
				self.history = {}

		def on_epoch_end(self, epoch, logs=None):
				if logs:
						for key in logs.keys():
								if epoch == 0:
										self.history[key] = []
								self.history[key].append(logs[key])
				if epoch%5 == 0:
						plot_history_for_callback(model_folder+'/train_history.png', self.history)
						save_history_for_callback(model_folder, self.history)
						img_vols, gt_vols, pr_vols = [],[],[]
						for i in range(0, len(valid_dataset),int(len(valid_dataset)/64)):
								img_vols.append(io.imread(valid_dataloader.dataset.images_fps[i]))
								gt_vols.append(valid_dataloader[i][1])
								pr_vols.append(self.model.predict(valid_dataloader[i]))
						img_vols = np.stack(img_vols, axis = 0)
						gt_vols = np.concatenate(gt_vols, axis = 0)
						pr_vols = np.concatenate(pr_vols, axis = 0)
						save_images(model_folder+'/epoch-{}-img.png'.format(epoch), img_vols)
						save_images(model_folder+'/epoch-{}-gt.png'.format(epoch), gt_vols/scale*255)
						save_images(model_folder+'/epoch-{}-pr.png'.format(epoch), pr_vols/scale*255)
				
#     		if epoch%5 == 0:
#     				plot_history_for_callback(model_folder+'/train_history.png', logs)

# define callbacks for learning rate scheduling and best checkpoints saving
if not best_select:
		callbacks = [
				tf.keras.callbacks.ModelCheckpoint(model_folder+'/weights_{epoch:02d}.h5', save_weights_only=True, save_best_only=False, period=5),
				tf.keras.callbacks.ReduceLROnPlateau(factor=decay),
				HistoryPrintCallback(),
		]
else:
		callbacks = [
				tf.keras.callbacks.ModelCheckpoint(model_folder+'/best_model-{epoch:03d}.h5', monitor='val_pearson', save_weights_only=True, save_best_only=True, mode='max'),
				tf.keras.callbacks.ReduceLROnPlateau(factor=decay),
				HistoryPrintCallback(),
		]

# train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)