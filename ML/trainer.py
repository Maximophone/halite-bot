import datetime
import os
import sys

import numpy as np
import json

from delay_run import wait_till

print('Waiting...')
# wait_till('19:30')
print('Ting!')

from keras.models import Sequential,load_model
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

import tensorflow as tf
import math
from mlutils import get_frames,center_frame,get_centroid_1D,get_centroid

if not hasattr(tf,'merge_all_summaries'):
    tf.merge_all_summaries = tf.summary.merge_all
    tf.train.SummaryWriter = tf.summary.FileWriter

import argparse

input_shape = (51,51)

def get_winner(frames):
    player=frames[:,:,:,0]
    players,counts = np.unique(player[-1],return_counts=True)
    target_id = players[counts.argmax()]
    return target_id

def main(model_file,replays,output_model, start, end, batch_size):
    if start is None:
        start = 0
    if end is None:
        end = 1e6

    training_input = []
    training_target = []    

    np.random.seed(0)

    model = load_model(model_file)

    n_replays = len(os.listdir(replays))
    print('Loading')
    for i,replay_name in enumerate(os.listdir(replays)):
        if i<=start or i>end:
            continue
        if replay_name[-4:]!='.hlt':continue
        print('Loading {}/{}'.format(i+1,n_replays))

        replay = json.load(open('{}/{}'.format(replays,replay_name)))

        frames = get_frames(replay)

        target_id = get_winner(frames)
        if target_id == 0: continue
        if 'nmalaguti' not in replay['player_names'][target_id-1].lower():
            continue

        moves = np.array(replay['moves'])

        is_player = frames[:,:,:,0]==target_id
        filtered_moves = np.where(is_player[:-1],moves,np.zeros_like(moves))
        categorical_moves = (np.arange(5) == filtered_moves[:,:,:,None]).astype(int)
        
        wrapped_frames = np.empty(shape=(frames.shape[0],input_shape[0],input_shape[1],frames.shape[3]))
        wrapped_moves = np.empty(shape=(categorical_moves.shape[0],input_shape[0],input_shape[1],categorical_moves.shape[3]))

        iframes = np.empty(shape=frames.shape[:3]+(4,))

        iframes[:,:,:,0] = frames[:,:,:,0] == target_id
        iframes[:,:,:,1] = (frames[:,:,:,0] != target_id) & (frames[:,:,:,0] != 0)
        iframes[:,:,:,2] = frames[:,:,:,1]/20.
        iframes[:,:,:,3] = frames[:,:,:,2]/255.

        for i,(frame,move) in enumerate(zip(iframes,categorical_moves)):
            centroid = get_centroid(frame)
            wframe = center_frame(frame,centroid,wrap_size=input_shape)
            wmoves = center_frame(move,centroid,wrap_size=input_shape)
            training_input.append(wframe)
            training_target.append(wmoves)

    print('Loaded')

    now = datetime.datetime.now()
    tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
    training_input = np.array(training_input)
    training_target = np.array(training_target)
    indices = np.arange(len(training_input))

    np.random.shuffle(indices) #shuffle training samples
    training_input = training_input[indices]
    training_target = training_target[indices]

    model.fit(training_input,training_target,validation_split=0.2,
              callbacks=[EarlyStopping(patience=500),ModelCheckpoint(model_file,verbose=1,save_best_only=True),
                         tensorboard],
              batch_size=batch_size, nb_epoch=10000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('replays')
    parser.add_argument('-o','--output')
    parser.add_argument('-s','--start',type=int,help="Starting replay in database")
    parser.add_argument('-e','--end',type=int,help="Last replay in database")
    parser.add_argument('-bs','--batch_size',type=int,default=256)

    args = parser.parse_args()

    main(args.model, args.replays, args.output if args.output else args.model, args.start, args.end, args.batch_size)



# import ipdb
# ipdb.set_trace()
# print('STILL accuracy:',model.evaluate(training_input[still_mask],training_target[still_mask],verbose=0)[1])
# print('MOVE accuracy:',model.evaluate(training_input[~still_mask],training_target[~still_mask],verbose=0)[1])