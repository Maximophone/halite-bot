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
    input_prev = Input(shape=input_alt_shape)

    merged_input = merge([input_, input_prev], mode='concat')

    #Branch 1

    cv0 = Convolution2D(32,3,3,border_mode='same')(merged_input)
    bn0 = BatchNormalization()(cv0)
    act0 = Activation("relu")(bn0)

    cv1 = AtrousConvolution2D(32,3,3,atrous_rate=(2,2),border_mode='same')(act0)
    bn1 = BatchNormalization()(cv1)
    act1 = Activation("relu")(bn1)

    cv2 = AtrousConvolution2D(32,3,3,atrous_rate=(4,4),border_mode='same')(act1)
    bn2 = BatchNormalization()(cv2)
    act2 = Activation("relu")(bn2)

    cv3 = AtrousConvolution2D(32,3,3,atrous_rate=(8,8),border_mode='same')(act2)
    bn3 = BatchNormalization()(cv3)
    act3 = Activation("relu")(bn3)

    #Branch 2

    cv0_2 = Convolution2D(32,3,3,border_mode='same', activation='relu')(merged_input)

    cv1_2 = AtrousConvolution2D(32,3,3,atrous_rate=(2,2),border_mode='same', activation='relu')(cv0_2)

    cv2_2 = AtrousConvolution2D(32,3,3,atrous_rate=(4,4),border_mode='same', activation='relu')(cv1_2)

    cv3_2 = AtrousConvolution2D(32,3,3,atrous_rate=(8,8),border_mode='same', activation='relu')(cv2_2)

    #Branch 3

    cv0_11 = Convolution2D(4,1,1,activation='relu',border_mode='same')(merged_input)
    cv0_33 = Convolution2D(8,3,3,activation='relu',border_mode='same')(merged_input)
    cv0_77 = Convolution2D(4,9,9,activation='relu',border_mode='same')(merged_input)

    merged_0 = merge([cv0_11, cv0_33, cv0_77], mode='concat')

    cv1_11 = Convolution2D(4,1,1,activation='relu',border_mode='same')(merged_0)
    cv1_33 = Convolution2D(8,3,3,activation='relu',border_mode='same')(merged_0)
    cv1_77 = Convolution2D(4,9,9,activation='relu',border_mode='same')(merged_0)

    merged_1 = merge([cv1_11, cv1_33, cv1_77], mode='concat')

    cv2_3 = Convolution2D(32,5,5,activation='relu',border_mode='same')(merged_1)

    cv3_3 = Convolution2D(32,5,5,activation='relu',border_mode='same')(cv2_3)
    
    cv4_3 = Convolution2D(5,3,3,border_mode='same')(cv3_3)

    merged_final = merge([act3, cv3_2, cv4_3], mode='concat')

    output = AtrousConvolution2D(5,3,3,atrous_rate=(16,16),border_mode='same',activation='relu')(merged_final)

    # Branch 

    opt = RMSprop(lr=0.0001)

    # model = Model(input=input_, output=output)
    model = Model(input=[input_,input_prev], output=output)

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
