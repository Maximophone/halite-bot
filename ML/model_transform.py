from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Reshape, Flatten, Merge, Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU

import argparse

input_shape=(51,51,4)

def main(old_model_file,new_model_file):

    old_model = load_model(old_model_file)
    weights = [layer.get_weights() for layer in old_model.layers]

    input_ = Input(shape=input_shape)

    cv0_11 = Convolution2D(4,1,1,activation='relu',border_mode='same',weights=weights[1])(input_)
    cv0_33 = Convolution2D(8,3,3,activation='relu',border_mode='same',weights=weights[2])(input_)
    cv0_77 = Convolution2D(4,9,9,activation='relu',border_mode='same',weights=weights[3])(input_)

    merged_0 = merge([cv0_11, cv0_33, cv0_77], mode='concat')


    cv1_11 = Convolution2D(4,1,1,activation='relu',border_mode='same',weights=weights[5])(merged_0)
    cv1_33 = Convolution2D(8,3,3,activation='relu',border_mode='same',weights=weights[6])(merged_0)
    cv1_77 = Convolution2D(4,9,9,activation='relu',border_mode='same',weights=weights[7])(merged_0)

    merged_1 = merge([cv1_11, cv1_33, cv1_77], mode='concat')

    cv2_11 = Convolution2D(4,1,1,activation='relu',border_mode='same')(merged_1)
    cv2_33 = Convolution2D(8,3,3,activation='relu',border_mode='same')(merged_1)
    cv2_77 = Convolution2D(4,9,9,activation='relu',border_mode='same')(merged_1)

    merged_2 = merge([cv2_11, cv2_33, cv2_77], mode='concat')

    cv2 = Convolution2D(32,5,5,activation='relu',border_mode='same',weights=weights[9])(merged_2)
    # cv2 = Convolution2D(32,5,5,activation='relu',border_mode='same')(merged_1)
    
    output = Convolution2D(5,3,3,activation='relu',border_mode='same',weights=weights[10])(cv2)

    opt = RMSprop(lr=0.0001)

    model = Model(input=input_, output=output)

    for layer in model.layers[:12]:
        layer.trainable = False

    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    model.save(new_model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('oldmodel')
    parser.add_argument('newmodel')

    args = parser.parse_args()

    print args

    main(args.oldmodel,args.newmodel)
