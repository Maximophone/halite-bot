from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Reshape, Flatten, Merge, Input, merge, Lambda
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AtrousConvolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
import numpy as np

import argparse

input_shape=(51,51,4)
input_alt_shape=(51,51,5)

def main(old_model_file,new_model_file):

    if old_model_file.lower() != 'none':
        old_model = load_model(old_model_file)
        weights = [layer.get_weights() for layer in old_model.layers]

    input_ = Input(shape=input_shape)

    cv0 = Convolution2D(16,3,3,border_mode='same')(input_)
    bn0 = BatchNormalization()(cv0)
    act0 = Activation("relu")(bn0)

    cv1 = AtrousConvolution2D(32,3,3,atrous_rate=(2,2),border_mode='same')(act0)
    bn1 = BatchNormalization()(cv1)
    act1 = Activation("relu")(bn1)

    cv2 = AtrousConvolution2D(64,3,3,atrous_rate=(4,4),border_mode='same')(act1)
    bn2 = BatchNormalization()(cv2)
    act2 = Activation("relu")(bn2)

    cv3 = AtrousConvolution2D(128,3,3,atrous_rate=(8,8),border_mode='same')(act2)
    bn3 = BatchNormalization()(cv3)
    act3 = Activation("relu")(bn3)

    output = AtrousConvolution2D(5,3,3,atrous_rate=(16,16),border_mode='same',activation='relu')(act3)

    # output = Convolution2D(5,3,3,activation='relu',border_mode='same')(cv4)

    opt = RMSprop(lr=0.0001)

    # model = Model(input=input_, output=output)
    model = Model(input=input_, output=output)

    # for layer in model.layers[:23]:
    #     layer.trainable = False
        
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    model.save(new_model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('oldmodel')
    parser.add_argument('newmodel')

    args = parser.parse_args()

    main(args.oldmodel,args.newmodel)
