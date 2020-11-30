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
import random
from model import unet_model_3d

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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

## load data
dataset = 'spheroids_x1'
# dataset = 'spheroids_x4'
DATA_DIR = '/data/datasets/{}'.format(dataset)

## load the training and testing list
train_file = DATA_DIR+'/train_list.txt'
test_file = DATA_DIR+'/test_list.txt'
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
        x_bd = [hsizes[0],vol_shp[0]-hsizes[0]]
        y_bd = [hsizes[1],vol_shp[1]-hsizes[1]]
        z_bd = [20, 80]
        xc = random.randint(x_bd[0], x_bd[1])
        yc = random.randint(y_bd[0], y_bd[1])
        zc = random.randint(z_bd[0], z_bd[1])
        X_sample = image[xc-hsizes[0]:xc+hsizes[0],yc-hsizes[1]:yc+hsizes[1],zc-hsizes[2]:zc+hsizes[2]]
        Y_sample = mask[xc-hsizes[0]:xc+hsizes[0],yc-hsizes[1]:yc+hsizes[1],zc-hsizes[2]:zc+hsizes[2]]
        
        X_sample = X_sample.transpose([2,0,1])
        Y_sample = Y_sample.transpose([2,0,1])
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


## 
BATCH_SIZE = 3
LR = 5e-6
decay = 0.8

## network architecture
model = unet_model_3d(input_shape = (1, None, None, None), n_base_filters=8, pool_size=(2, 2, 1), depth = 4)

## testing training dataset
train_dataset = Dataset(X_dir, Y_dir, trn_fns, scale = 100, hsizes = [256, 256, 4])
valid_dataset = Dataset(X_dir, Y_dir, tst_fns, scale = 100, hsizes = [256, 256, 4])
print(train_dataset[0][0].shape, train_dataset[0][1].shape)
train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

loss = tf.keras.losses.MSE
optim = tf.keras.optimizers.Adam(LR)
model.compile(optim, loss)
model_name = '3D_phase_x1'
model_folder = '/data/models_fl/{}'.format(model_name)
generate_folder(model_folder)
# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_folder+'/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(factor=decay),
]

EPOCHS = 100000
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)

## 
# nb_sampling = 10
# for epoch in range(1):
# 		trn_ph_vois, trn_fl_vois = [],[]
# 		for iv in range(len(train_dataset)):
# 				for j in range(nb_sampling):
# 						ph_voi, fl_voi = train_dataset[iv]
# 						trn_ph_vois.append(ph_voi)
# 						trn_fl_vois.append(fl_voi)
# 		trn_phs = np.stack(trn_ph_vois, axis = 0)
# 		trn_fls = np.stack(trn_fl_vois, axis = 0)
# 		model.fit(tran_phs, trn_fls, batch_size= BATCH_SIZE)

test_dataset = Dataset(X_dir, Y_dir, tst_fns, scale = 100, hsizes = [192,192,16])
test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
ph_vols = np.stack([test_dataset[i][0].squeeze() for i in range(len(test_dataset))])
gt_vols = np.stack([test_dataset[i][1].squeeze() for i in range(len(test_dataset))])

pr_vols = np.concatenate([model.predict(np.expand_dims(np.expand_dims(ph_vols[i,:,:,:], axis=0),axis=0)) for i in range(ph_vols.shape[0])])
pr_vols = pr_vols.squeeze()
SMOOTH= 1e-6; seed = 0
def calculate_pearsonr(vol1, vol2):
	from scipy import stats
	import numpy as np
	SMOOTH_ = np.random.random(vol1.flatten().shape)*SMOOTH
	flat1 = vol1.flatten(); flat2 = vol2.flatten().flatten()
	flat1 = SMOOTH_ + flat1; flat2 = SMOOTH_ + flat2
	score, _= stats.pearsonr(flat1, flat2)
	return score

for i in range(pr_vols.shape[0]):
  cor = calculate_pearsonr(gt_vols[i,:,:,:], pr_vols[i,:,:,:])
  print('Pearsonr {:.2f}'.format(cor))

# test_dataset = Dataset(X_dir, Y_dir, tst_fns, hsizes = [256,256,8])
# test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
# pr_vol = model.predict(test_dataloader)


    