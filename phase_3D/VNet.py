"""
Diogo Amorim, 2018-07-10
V-Net implementation in Keras 2
https://arxiv.org/pdf/1606.04797.pdf
"""

import functools

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import activations
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import initializers

K.set_image_data_format("channels_first")

def downward_layer(input_layer, n_convolutions, n_output_channels):
    inl = input_layer

    for _ in range(n_convolutions):
        inl = ReLU()(BatchNormalization(axis=1)(
            Conv3D(filters=(n_output_channels // 2), kernel_size=5,
                   padding='same', kernel_initializer='he_normal')(inl))
        )
    add_l = add([inl, input_layer])
    downsample = Conv3D(filters=n_output_channels, kernel_size=2, strides =(2, 2, 1),
                        padding='same', kernel_initializer='he_normal')(add_l)
    downsample = BatchNormalization(axis=1)(downsample)
    downsample = ReLU()(downsample)
    return downsample, add_l


def upward_layer(input0, input1, n_convolutions, n_output_channels):
    merged = concatenate([input0, input1], axis=1)
    inl = merged
#     print(n_output_channels)
    for _ in range(n_convolutions):
        inl = ReLU()(BatchNormalization(axis=1)(
            Conv3D((n_output_channels*4), kernel_size=5,
                   padding='same', kernel_initializer='he_normal')(inl))
        )
#     print(inl.get_shape().as_list(), merged.get_shape().as_list())
    add_l = add([inl, merged])
    upsample = BatchNormalization(axis=1)(Conv3DTranspose(n_output_channels, (2, 2, 2), strides = (2, 2, 1), padding='same', kernel_initializer='he_normal')(add_l))
    return ReLU()(upsample)

def vnet(input_shape=(1, 128, 128, 64), activation_name = 'relu'):
         # loss='categorical_crossentropy', metrics=['categorical_accuracy']):
    # Layer 1
    inputs = Input(input_shape)
    conv1 = Conv3D(16, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = ReLU()(conv1)
    repeat1 = concatenate(16 * [inputs], axis=1)
    add1 = add([conv1, repeat1])
    down1 = Conv3D(32, 2, strides=(2,2,1), padding='same', kernel_initializer='he_normal')(add1)
    down1 = BatchNormalization(axis=1)(down1)
    down1 = ReLU()(down1)

    # Layer 2,3,4
    down2, add2 = downward_layer(down1, 2, 64)
    down3, add3 = downward_layer(down2, 3, 128)
    down4, add4 = downward_layer(down3, 3, 256)

    # Layer 5
    # !Mudar kernel_size=(5, 5, 5) quando imagem > 64!
    conv_5_1 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(down4)
    conv_5_1 = BatchNormalization(axis=1)(conv_5_1)
    conv_5_1 = ReLU()(conv_5_1)
    conv_5_2 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(conv_5_1)
    conv_5_2 = BatchNormalization(axis=1)(conv_5_2)
    conv_5_2 = ReLU()(conv_5_2)
    conv_5_3 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(conv_5_2)
    conv_5_3 = BatchNormalization(axis=1)(conv_5_3)
    conv_5_3 = ReLU()(conv_5_3)
    add5 = add([conv_5_3, down4])
    upsample_5 = Conv3DTranspose(128, (2, 2, 2), strides = (2, 2, 1), padding='same', kernel_initializer='he_normal')(add5)
    upsample_5 = BatchNormalization(axis=1)(upsample_5)
    upsample_5 = ReLU()(upsample_5)

    # Layer 6,7,8
    upsample_6 = upward_layer(upsample_5, add4, 3, 64)
    upsample_7 = upward_layer(upsample_6, add3, 3, 32)
    upsample_8 = upward_layer(upsample_7, add2, 2, 16)

    # Layer 9
    merged_9 = concatenate([upsample_8, add1], axis=1)
    conv_9_1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal')(merged_9)
    conv_9_1 = ReLU()(conv_9_1)
    add_9 = add([conv_9_1, merged_9])
    # conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
    conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
    conv_9_2 = ReLU()(conv_9_2)

    # softmax = Softmax()(conv_9_2)
    output = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',
                     activation=activation_name)(conv_9_2)
    model = Model(inputs=inputs, outputs=output)

    return model

# model = vnet()
# model.summary(line_length=133)