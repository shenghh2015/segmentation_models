import tensorflow as tf
import os
import sys
import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
import random
from model import unet_model_3d

from helper_function import plot_history_for_callback, save_history_for_callback

def str2bool(value):
		return value.lower() == 'true'

def generate_folder(folder_name):
	  if not os.path.exists(folder_name):
				os.system('mkdir -p {}'.format(folder_name))

def read_txt(txt_dir):
    lines = []
    with open(txt_dir, 'r+') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default = '0')
parser.add_argument("--docker", type=str2bool, default = True)
parser.add_argument("--dataset", type=str, default = 'spheroids_x1')
parser.add_argument("--filters", type=int, default = 16)
parser.add_argument("--levels", type=int, default = 5)
parser.add_argument("--dim", type=int, default = 256)
parser.add_argument("--dep", type=int, default = 16)
parser.add_argument("--val_dim", type=int, default = 256)
parser.add_argument("--val_dep", type=int, default = 32)
parser.add_argument("--epoch", type=int, default = 10)
parser.add_argument("--batch_size", type=int, default = 3)
parser.add_argument("--lr", type=float, default = 5e-6)
parser.add_argument("--scale", type=float, default = 1.0)
parser.add_argument("--decay", type=float, default = 0.8)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

## load the training and testing list
dataset = args.dataset
# dataset = 'spheroids_x4'
DATA_DIR = '/data/datasets/{}'.format(dataset) if args.docker else './datasets/{}'.format(dataset)
train_file = DATA_DIR+'/spheroids_v3_train.txt'
test_file = DATA_DIR+'/spheroids_v3_test.txt'
trn_fns = read_txt(train_file)
tst_fns = read_txt(test_file)
val_fns = tst_fns

X_dir = os.path.join(DATA_DIR, 'phase')
Y_dir = os.path.join(DATA_DIR, 'fl1')

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            file_names,
            scale = 1.0,
            hsizes = [160, 160, 10],
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = ['{}.npy'.format(fn) for fn in file_names]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        print(len(self.images_fps)); print(len(self.masks_fps))        
        self.scale = scale
        self.hsizes = hsizes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = np.load(self.images_fps[i]).transpose([1,2,0])
        mask  = np.load(self.masks_fps[i]).transpose([1,2,0])
        mask = mask/255.*self.scale    

				# random cropping: (z, x, y)
        hsizes = self.hsizes
        vol_shp = mask.shape
        x_bd = [0,vol_shp[0]-hsizes[0]]
        y_bd = [0,vol_shp[1]-hsizes[1]]
        z_bd = [20, 80-hsizes[2]]
        xi = random.randint(x_bd[0], x_bd[1])
        yi = random.randint(y_bd[0], y_bd[1])
        zi = random.randint(z_bd[0], z_bd[1])
        X_sample = image[xi:xi+hsizes[0],yi:yi+hsizes[1],zi:zi+hsizes[2]]
        Y_sample = mask[xi:xi+hsizes[0],yi:yi+hsizes[1],zi:zi+hsizes[2]]
        
#         X_sample = X_sample.transpose([2,0,1])
#         Y_sample = Y_sample.transpose([2,0,1])
        X_sample = np.expand_dims(X_sample, axis = 0)
        Y_sample = np.expand_dims(Y_sample, axis = 0)
        # apply augmentations
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']

        # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']

        return X_sample, Y_sample
        
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

## parameters
dim, dep, val_dim, val_dep, scale = args.dim, args.dep, args.val_dim, args.val_dep, args.scale
BATCH_SIZE, EPOCHS, LR, decay = args.batch_size, args.epoch, args.lr, args.decay
levels, filters = args.levels, args.filters
model_name = '3d-set-{}-dim-{}-dep-{}-val_dim-{}-val_dep-{}-bz-{}-lr-{}-level-{}-filters-{}-ep-{}-decay-{}-scale-{}'.format(dataset,
							dim, val_dep, val_dim, val_dep, BATCH_SIZE, LR, levels, filters, EPOCHS, decay, scale)
model_folder = '/data/3d_models/{}/{}'.format(dataset, model_name) if args.docker else './3d_models/{}/{}'.format(dataset, model_name)
generate_folder(model_folder)

## network architecture
model = unet_model_3d(input_shape = (1, None, None, None), n_base_filters=filters, pool_size=(2, 2, 1), depth = levels)

## testing training dataset
train_dataset = Dataset(X_dir, Y_dir, trn_fns, scale = scale, hsizes = [dim, dim, dep])
valid_dataset = Dataset(X_dir, Y_dir, tst_fns, scale = scale, hsizes = [val_dim, val_dim, val_dep])
print(train_dataset[0][0].shape, train_dataset[0][1].shape)
print(valid_dataset[0][0].shape, valid_dataset[0][1].shape)
train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

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

def psnr(y_true, y_pred):
		mse = K.mean(K.square(y_true-y_pred))
		psnr_score = 20*K.log(1./K.sqrt(mse))
		return psnr_score

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def save_images(file_name, vols):
		shp = vols.shape
		bx, by, bz = shp[-3], shp[-2], shp[-1]
		sx, sy, sz = int(bx/64), int(by/64), int(bz/10)
		if len(shp) == 4:
				tile_list = []
				for vi in range(shp[0]):
						vol = vols[vi,::sx,::sy,::sz]
						slice_list = [vol[:,:,si] for si in range(vol.shape[-1])]
						tile_list.append(slice_list)
		elif len(shp) == 3:
				tile_list = []
				vol = vols[::sx,::sy,::sz]
				slice_list = [vol[:,:,si] for si in range(vol.shape[-1])]
				tile_list.append(slice_list)
		save_img = concat_tile(tile_list)		
		cv2.imwrite(file_name, save_img)		

metrics = [psnr, pearson]

loss = tf.keras.losses.MSE
optim = tf.keras.optimizers.Adam(LR)
model.compile(optim, loss, metrics)

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
						ph_vols, fl_gts, fl_prs = [], [], []
						for i in range(len(valid_dataloader)):
								ph_vol, fl_gt = valid_dataloader[i]
								fl_pr = self.model.predict(ph_vol)
								ph_vols.append(ph_vol); fl_gts.append(fl_gt); fl_prs.append(fl_pr)
						ph_vols = np.stack(ph_vols).squeeze()
						fl_gts = np.stack(fl_gts).squeeze()
						fl_prs = np.stack(fl_prs).squeeze(); print('max voxel value:{:.2f}'.format(fl_prs.max()))
						print(ph_vols.shape)
						save_images(model_folder+'/phase_{}.png'.format(epoch), ph_vols)
						save_images(model_folder+'/gt_{}.png'.format(epoch), fl_gts*255./scale)
						save_images(model_folder+'/pr_{}.png'.format(epoch), fl_prs*255./scale)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_folder+'/best_model-{epoch:03d}.h5', monitor='val_pearson', save_weights_only=True, save_best_only=True, mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(factor=decay),
		HistoryPrintCallback(),
]

# EPOCHS = 20
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)
