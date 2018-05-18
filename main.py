
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import cv2
import tensorflow as tf
#from utils import load, save
#from layers import Deconv2D
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Activation, Conv2D, LeakyReLU, Flatten, BatchNormalization as BN
from keras.models import Sequential, Model
#from keras import initializations

learning_rate = 0.0003
beta1 = .5
z_dim = 128
#df_dim = 64
def encoder(num_filters, ch, rows, cols):

    model = Sequential()
    X = Input(shape=(rows[-1], cols[-1], ch))

    model = Conv2D(num_filters, kernel_size=(5,5), strides=(2,2), padding='same', name='enc_conv2D_01', input_shape=(rows, cols, ch))(X)
    model = BN(axis=3, name="enc_bn_01",  epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Conv2D(num_filters*2,kernel_size=(5,5), strides=(2,2), padding='same', name='enc_conv2D_02')(model)
    model = BN(axis=3, name="enc_bn_02",  epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Conv2D(num_filters*4,kernel_size=(5,5), strides=(2,2), padding='same', name='enc_conv2D_03')(model)
    model = BN(axis=3, name="enc_bn_03",  epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    #model = Reshape((8,8,256))(model)
    model = Flatten()(model)
    model = Dense(2048, name="enc_dense_01")(model)
    model = BN(name="enc_bn_04",  epsilon=1e-5)(model)
    encoded_model = LeakyReLU(.2)(model)

    mean = Dense(z_dim, name="e_h3_lin")(encoded_model)
    logsigma = Dense(z_dim, name="e_h4_lin", activation="tanh")(encoded_model)
    meansigma = Model([X], [mean, logsigma])


    #X_decode = Input(shape=(8,8,256))
    #model = Dense(256, name="dec_dense_01")(encoded_model)

#    enc_model = Model(X, encoded_model)
#    dec_model = Model(X, model)
    return meansigma


df_dim = 64
batch_size = 64
channels = 3
height = np.array([64])
width = np.array([64])

vae_encoder = encoder(num_filters=df_dim, ch=channels, rows=height, cols=width)
vae_encoder.compile(optimizer='RMSProp', loss='binary_crossentropy')
vae_encoder.summary()
