import datetime
import os, sys

import numpy as np

from delay_run import wait_till

print('Waiting...')
# wait_till('19:30')
print('Ting!')

from keras.models import load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

import tensorflow as tf
from mlutils import get_frames,center_frame,get_centroid_1D,get_centroid

if not hasattr(tf,'merge_all_summaries'):
    tf.merge_all_summaries = tf.summary.merge_all
    tf.train.SummaryWriter = tf.summary.FileWriter
    tf.histogram_summary = tf.summary.histogram

import argparse

input_shape = (51,51)

dict_rot90 = {
    0:0,
    1:4,
    2:1,
    3:2,
    4:3
}
dict_fliplr = {
    0:0,
    1:1,
    2:4,
    3:3,
    4:2
}
dict_flipud = {
    0:0,
    1:3,
    2:2,
    3:1,
    4:4
}
dir_rot90 = np.vectorize(dict_rot90.get)
dir_fliplr = np.vectorize(dict_fliplr.get)
dir_flipud = np.vectorize(dict_flipud.get)

def rot90(arr):
    rot_arr = np.rot90(arr)
    dir_rot_arr = (np.arange(5) == dir_rot90(np.argmax(rot_arr,axis=2))[:,:,None]).astype(int)
    return dir_rot_arr

def fliplr(arr):
    flip_arr = np.fliplr(arr)
    dir_flip_arr = (np.arange(5) == dir_fliplr(np.argmax(flip_arr,axis=2))[:,:,None]).astype(int)
    return dir_flip_arr

def flipud(arr):
    flip_arr = np.flipud(arr)
    dir_flip_arr = (np.arange(5) == dir_flipud(np.argmax(flip_arr,axis=2))[:,:,None]).astype(int)
    return dir_flip_arr

def dir_identity(arr):
    return arr

def dir_rot1(arr):
    return rot90(arr)

def dir_rot2(arr):
    return rot90(rot90(arr))

def dir_rot3(arr):
    return rot90(rot90(rot90(arr)))

def dir_flip1(arr):
    return fliplr(arr)

def dir_diag1(arr):
    return rot90(fliplr(arr))

def dir_flip2(arr):
    return flipud(arr)

def dir_diag2(arr):
    return rot90(flipud(arr))

dir_transforms = [dir_identity,dir_rot1,dir_rot2,dir_rot3,dir_flip1,dir_diag1,dir_flip2,dir_diag2]

def identity(arr):
    return arr

def rot1(arr):
    return np.rot90(arr)

def rot2(arr):
    return np.rot90(np.rot90(arr))

def rot3(arr):
    return np.rot90(np.rot90(np.rot90(arr)))

def flip1(arr):
    return np.fliplr(arr)

def diag1(arr):
    return np.rot90(np.fliplr(arr))

def flip2(arr):
    return np.flipud(arr)

def diag2(arr):
    return np.rot90(np.flipud(arr))

transforms = [identity,rot1,rot2,rot3,flip1,diag1,flip2,diag2]

def transform_input(frames, directions, prev_moves=None):
    it = np.random.randint(8)
    new_frames = np.array([transforms[it](frame) for frame in frames])
    new_directions = np.array([dir_transforms[it](dirmap) for dirmap in directions])
    if prev_moves is not None:
        new_prev_moves = np.array([dir_transforms[it](prev_move) for prev_move in prev_moves])
        return [new_frames, new_prev_moves], new_directions
    else:
        return new_frames, new_directions

def get_data_generator(training_input, training_target, batch_size, training_input_alt=None):
    cursor = 0
    L = len(training_input)
    perm = np.arange(L)
    while True:
        if cursor+batch_size>L:
            np.random.shuffle(perm)
            training_input[:] = training_input[perm]
            training_target[:] = training_target[perm]
            if training_input_alt is not None:
                training_input_alt[:] = training_input_alt[perm]
            cursor = 0
        yield transform_input(
            training_input[cursor:cursor+batch_size],
            training_target[cursor:cursor+batch_size],
            training_input_alt[cursor:cursor+batch_size] if training_input_alt is not None else None
            )
        cursor += batch_size


def main(model_file,replays,output_model, batch_size, replays_chunk, patience, log_dir, double_input):

    if replays[-1]=='/': 
        replays = replays[:-1]

    np.random.seed(876864)

    print('Loading...')

    training_input = np.load('{}/training_input_{}.npy'.format(replays,replays_chunk))
    training_target = np.load('{}/training_target_{}.npy'.format(replays,replays_chunk))
    test_input = np.load('{}/test_input_{}.npy'.format(replays,replays_chunk))
    test_target = np.load('{}/test_target_{}.npy'.format(replays,replays_chunk))

    if double_input:
        training_input_alt = np.load('{}/training_input_alt_{}.npy'.format(replays,replays_chunk))
        test_input_alt = np.load('{}/test_input_alt_{}.npy'.format(replays,replays_chunk))

    print('Features Loaded')

    model = load_model(model_file)

    print('Model Loaded')

    now = datetime.datetime.now()
    logname = '{}_{}_rc{}_{}'.format(model_file.split('/')[-1],replays.split('/')[-1],replays_chunk,now.strftime('%Y.%m.%d %H.%M'))
    tensorboard = TensorBoard(log_dir='./{}/{}'.format(log_dir,logname))

    if double_input:
        data_generator = get_data_generator(training_input, training_target, batch_size, training_input_alt=training_input_alt)
    else:
        data_generator = get_data_generator(training_input, training_target, batch_size)

    model.fit_generator(
        data_generator,
        samples_per_epoch=len(training_input),
        validation_data=(test_input,test_target) if not double_input else ([test_input,test_input_alt], test_target),
        callbacks=[
            EarlyStopping(patience=patience),
            ModelCheckpoint(output_model,verbose=1,save_best_only=True),
            tensorboard],
            nb_epoch=10000
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('replays')
    parser.add_argument('-rc','--replays_chunk',type=int,default=0)
    parser.add_argument('-o','--output')
    parser.add_argument('-bs','--batch_size',type=int,default=256)
    parser.add_argument('-p','--patience',type=int,default=500)
    parser.add_argument('-lg','--log_dir',type=str,default='logs')
    parser.add_argument('-di','--double_input', action="store_true", help="Use previous moves as additional input")

    args = parser.parse_args()

    main(
        args.model, 
        args.replays, 
        args.output if args.output else args.model, 
        args.batch_size, 
        args.replays_chunk,
        args.patience,
        args.log_dir,
        args.double_input
        )

