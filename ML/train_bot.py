import datetime
import os
import sys

import numpy as np
import json

from keras.models import Sequential,load_model
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

import tensorflow as tf
import math

if not hasattr(tf,'merge_all_summaries'):
    tf.merge_all_summaries = tf.summary.merge_all
    tf.train.SummaryWriter = tf.summary.FileWriter

def get_frames(replay):
    frames = np.array(replay['frames'])
    prods = np.repeat(np.array(replay['productions']).reshape((1,frames.shape[1],frames.shape[2],1)),frames.shape[0],axis=0)
    return np.concatenate([frames,prods],axis = 3)

def center_frame(frame,position,wrap_size=None):
    if not wrap_size:
        h = frame.shape[0]
        w = frame.shape[1]
    else:
        h,w = wrap_size
    return np.take(np.take(frame,
                np.arange(-int(h/2),int(h/2) + 1)+position[0],axis=0,mode='wrap'),
                np.arange(-int(w/2),int(w/2) + 1)+position[1],axis=1,mode='wrap')

def get_centroid_1D(X,L):
    n = len(X)
    mu_x = 1./n*sum([math.cos(x/float(L)*2*math.pi) for x in X])
    mu_y = 1./n*sum([math.sin(x/float(L)*2*math.pi) for x in X])
    if mu_x == 0:
        if mu_y > 0:
            return math.pi/2.
        else:
            return -math.pi/2.
    return int(round(L/math.pi/2.*math.atan2(mu_y,mu_x)%L))

def get_centroid(frame):
    X,Y = np.where(frame[:,:,0]==1)
    mX = get_centroid_1D(X,frame.shape[0])
    mY = get_centroid_1D(Y,frame.shape[1])
    return (mX,mY)

REPLAY_FOLDER = sys.argv[1]
input_shape = (51,51)

training_input = []
training_target = []

np.random.seed(0)

MODEL_FILE = "model_0.h5"

if os.path.isfile(MODEL_FILE):
    model = load_model(MODEL_FILE)
else:

    model = Sequential()

    model.add(Convolution2D(16,5,5,activation='relu',border_mode='same',input_shape=input_shape+(4,)))
    model.add(Convolution2D(32,5,5,activation='relu',border_mode='same'))
    model.add(Convolution2D(64,5,5,activation='relu',border_mode='same'))
    model.add(Convolution2D(5,5,5,activation='relu',border_mode='same'))

    opt = Adam(lr=0.00001)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


n_replays = len(os.listdir(REPLAY_FOLDER))
for i,replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    if i>50:
        break
    if replay_name[-4:]!='.hlt':continue
    print('Loading {}/{}'.format(i+1,n_replays))
    replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name)))

    frames = get_frames(replay)

    player=frames[:,:,:,0]
    players,counts = np.unique(player[-1],return_counts=True)
    target_id = players[counts.argmax()]
    if target_id == 0: continue
    # if 'erdman' not in replay['player_names'][target_id-1].lower():
    #     continue

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

now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
training_input = np.array(training_input)
training_target = np.array(training_target)
indices = np.arange(len(training_input))

np.random.shuffle(indices) #shuffle training samples
training_input = training_input[indices]
training_target = training_target[indices]

model.fit(training_input,training_target,validation_split=0.2,
          callbacks=[EarlyStopping(patience=10),ModelCheckpoint(MODEL_FILE,verbose=1,save_best_only=True),
                     tensorboard],
          batch_size=1024, nb_epoch=1000)

model = load_model(MODEL_FILE)

still_mask = training_target[:,:,:,0].astype(bool)

predictions = model.predict(training_input)

# import ipdb
# ipdb.set_trace()
# print('STILL accuracy:',model.evaluate(training_input[still_mask],training_target[still_mask],verbose=0)[1])
# print('MOVE accuracy:',model.evaluate(training_input[~still_mask],training_target[~still_mask],verbose=0)[1])