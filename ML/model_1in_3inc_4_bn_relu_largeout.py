from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Reshape, Flatten, Merge, Input, merge, Lambda
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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

    cv0_1 = Convolution2D(4,1,1,border_mode='same',weights=weights[1])(input_)
    cv0_3 = Convolution2D(8,3,3,border_mode='same',weights=weights[2])(input_)
    cv0_11 = Convolution2D(4,11,11,border_mode='same',weights=weights[3])(input_)

    merged_0 = merge([cv0_1, cv0_3, cv0_11], mode='concat')
    bn0 = BatchNormalization(weights=weights[5])(merged_0)
    act0 = Activation("relu")(bn0)

    cv1_1 = Convolution2D(4,1,1,border_mode='same',weights=weights[7])(act0)
    cv1_3 = Convolution2D(8,3,3,border_mode='same',weights=weights[8])(act0)
    cv1_11 = Convolution2D(4,11,11,border_mode='same',weights=weights[9])(act0)

    merged_1 = merge([cv1_1, cv1_3, cv1_11], mode='concat')
    bn1 = BatchNormalization(weights=weights[11])(merged_1)
    act1 = Activation("relu")(bn1)

    cv2_1 = Convolution2D(4,1,1,border_mode='same',weights=weights[13])(act1)
    cv2_3 = Convolution2D(8,3,3,border_mode='same',weights=weights[14])(act1)
    cv2_11 = Convolution2D(4,11,11,border_mode='same',weights=weights[15])(act1)

    merged_2 = merge([cv2_1, cv2_3, cv2_11], mode='concat')
    bn2 = BatchNormalization(weights=weights[17])(merged_2)
    act2 = Activation("relu")(bn2)

    cv3 = Convolution2D(32,3,3,border_mode='same',weights=weights[19])(act2)
    bn3 = BatchNormalization(weights=weights[20])(cv3)
    act3 = Activation("relu")(bn3)

    cv4 = Convolution2D(64,3,3,border_mode='same',weights=weights[22])(act3)
    # bn4 = BatchNormalization(weights=weights[23])(cv4)
    act4 = Activation("relu")(cv4)

    cv5 = Convolution2D(128,3,3,border_mode='same',weights=weights[24])(act4)
    bn5 = BatchNormalization(weights=weights[25])(cv5)
    act5 = Activation("relu")(bn5)

    cv6 = Convolution2D(8,3,3,activation='relu',border_mode='same',weights=weights[27])(act5)
    bn6 = BatchNormalization(weights=weights[28])(cv6)
    act6 = Activation("relu")(bn6)

    output = Convolution2D(5,20,20,activation='relu',border_mode='same',weights=weights[30])(act6)

    # input_alt = Input(shape=input_alt_shape)

    # merged_2 = merge([cv5, input_alt, input_], mode='concat')

    # cv6 = Convolution2D(16,3,3,border_mode='same',weights=weights[25])(merged_2)
    # bn6 = BatchNormalization(weights=weights[26])(cv6)
    # lr6 = LeakyReLU(alpha=0.01)(bn6)

    # cv7 = Convolution2D(32,3,3,border_mode='same',weights=weights[28])(lr6)
    # bn7 = BatchNormalization(weights=weights[29])(cv7)
    # lr7 = LeakyReLU(alpha=0.01)(bn7)

    # cv8 = Convolution2D(64,3,3,border_mode='same')(lr7)
    # bn8 = BatchNormalization()(cv8)
    # lr8 = LeakyReLU(alpha=0.01)(bn8)
    # cv7 = Convolution2D(5,3,3,activation='relu',border_mode='same',weights=weights[16])(cv6)

    # special_weights = np.zeros((3,3,14,16))
    # special_biases = np.zeros((16,))
    # special_weights[1,1,0,0] = 1.
    # special_weights[1,1,1,1] = 1.
    # special_weights[1,1,2,2] = 1.
    # special_weights[1,1,3,3] = 1.
    # special_weights[1,1,4,4] = 1.

    # cv6 = Convolution2D(16,3,3,activation='relu',border_mode='same',weights=weights[25])(merged_2)

    # special_weights = np.zeros((3,3,16,32))
    # special_biases = np.zeros((32,))
    # for i in range(16):
    #     special_weights[1,1,i,i] = 1.

    # cv7 = Convolution2D(32,3,3,activation='relu',border_mode='same',weights=(special_weights,special_biases))(cv6)

    # special_weights = np.zeros((3,3,32,5))
    # special_weights[:,:,:16,:] = weights[26][0]
    # special_biases = weights[26][1]

    # output = Convolution2D(5,3,3,activation='relu',border_mode='same')(cv5)

    opt = RMSprop(lr=0.0001)

    # model = Model(input=input_, output=output)
    model = Model(input=input_, output=output)

    # for layer in model.layers[:26]:
    #     layer.trainable = False
        
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    model.save(new_model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('oldmodel')
    parser.add_argument('newmodel')

    args = parser.parse_args()

    print args

    main(args.oldmodel,args.newmodel)
