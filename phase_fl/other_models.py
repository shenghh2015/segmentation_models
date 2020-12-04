import numpy as np 
import os

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

def down_conv2(x, filters, n_convs = 2, bn = False):
		if bn:
				x = LeakyReLU(alpha=0.3)(BatchNorm(axis = -1)(Conv2D(filters, 3, padding = 'same')(x)))
				x = LeakyReLU(alpha=0.3)(BatchNorm(axis = -1)(Conv2D(filters, 3, padding = 'same')(x)))
		else:
				x = LeakyReLU(alpha=0.3)(Conv2D(filters, 3, padding = 'same')(x))
				x = LeakyReLU(alpha=0.3)(Conv2D(filters, 3, padding = 'same')(x))
		pool = MaxPooling2D(pool_size=(2, 2))(x)
		return pool, x
		
def up_conv2(x1, x2, filters, n_convs = 2, bn = False):
		x = concatenate([UpSampling2D(size = (2,2))(x1), x2], axis = -1)
		if bn:
				x = LeakyReLU(alpha=0.3)(BatchNorm(axis = -1)(Conv2D(filters, 3, padding = 'same')(x)))
				x = LeakyReLU(alpha=0.3)(BatchNorm(axis = -1)(Conv2D(filters, 3, padding = 'same')(x)))
		else:
				x = LeakyReLU(alpha=0.3)(Conv2D(filters, 3, padding = 'same')(x))
				x = LeakyReLU(alpha=0.3)(Conv2D(filters, 3, padding = 'same')(x))
		return x		

def gunet3(input_size = (None,None,None), base_filters = 16, level = 5, bn = False):
    inputs = Input(input_size)
    down1, add1 = down_conv2(x, base_filters, n_convs = 2, bn = bn)
    down2, add2 = down_conv2(down1, base_filters*2, n_convs = 2, bn = bn)
    down3, add3 = down_conv2(down2, base_filters*4, n_convs = 2, bn = bn)
    down4, add4 = down_conv2(down3, base_filters*8, n_convs = 2, bn = bn)
    down5, add5 = down_conv2(down4, base_filters*16, n_convs = 2, bn = bn)
    down6, add6 = down_conv2(down5, base_filters*32, n_convs = 2, bn = bn)
    down7, add6 = down_conv2(down5, base_filters*32, n_convs = 2, bn = bn)

    conv1 = Conv2D(16)(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
​
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)
​
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
​
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(pool2))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
​
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
​
    model4 = Model(input = inputs, output = conv10)
​
    return model4 