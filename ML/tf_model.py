import tensorflow as tf
import mlutils
import os
import json
import numpy as np
import datetime

REPLAY_FOLDER = './replays'
input_shape = (51,51)

MODEL_SAVED = 'saved_models/tf_model_0'

split_ratio = 0.8

input_ = []
target_ = []

np.random.seed(0)

n_replays = len(os.listdir(REPLAY_FOLDER))
for i,replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    if i>300:
        break
    if replay_name[-4:]!='.hlt':continue
    print('Loading {}/{}'.format(i+1,n_replays))
    replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name)))

    frames = mlutils.get_frames(replay)

    player=frames[:,:,:,0]
    players,counts = np.unique(player[-1],return_counts=True)
    target_id = players[counts.argmax()]
    if target_id == 0: continue
    # if 'erdman' not in replay['player_names'][target_id-1].lower():
    #     continue
    n_max = min(len(replay['frames']),20)
    frames = frames[:n_max]

    moves = np.array(replay['moves'])[:n_max-1]

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
        centroid = mlutils.get_centroid(frame)
        wframe = mlutils.center_frame(frame,centroid,wrap_size=input_shape)
        wmoves = mlutils.center_frame(move,centroid,wrap_size=input_shape)
        input_.append(wframe)
        target_.append(wmoves)

input_ = np.array(input_)
target_ = np.array(target_)
indices = np.arange(len(input_))

np.random.shuffle(indices) #shuffle training samples
input_ = input_[indices]
target_ = target_[indices]

total_size = input_.shape[0]
train_input = input_[:int(split_ratio*total_size)]
test_input = input_[int(split_ratio*total_size):]

train_target = target_[:int(split_ratio*total_size)]
test_target = target_[int(split_ratio*total_size):]

sess = tf.Session()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 4, 16])
b_conv1 = bias_variable([16])

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])

W_conv3 = weight_variable([5, 5, 32, 64])
b_conv3 = bias_variable([64])

W_conv4 = weight_variable([5, 5, 64, 5])
b_conv4 = bias_variable([5])

x = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], 4], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], 5], name='y_')
mask = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], 5], name='mask')

tf.add_to_collection("x",x)
tf.add_to_collection("y_",y_)
tf.add_to_collection("mask",mask)


h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1,name='h_conv1')
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2,name='h_conv2')
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3,name='h_conv3')
y = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4,name='y')

tf.add_to_collection("y",y)

output_masked = tf.mul(y,mask,name='output_masked')
target_masked = tf.mul(y_,mask,name='target_masked')

mse = tf.reduce_sum(tf.square(tf.sub(target_masked, output_masked)))/tf.reduce_sum(mask)
tf.add_to_collection("mse",mse)
tf.summary.scalar('mse', mse)

correct_prediction = tf.equal(tf.argmax(output_masked,axis=3), tf.argmax(target_masked,axis=3))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))/tf.reduce_sum(mask)
tf.add_to_collection("accuracy",accuracy)
tf.summary.scalar('accuracy', accuracy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)
merged = tf.summary.merge_all()

now = datetime.datetime.now()
train_writer = tf.summary.FileWriter('tflogs/{}/train'.format(now.strftime('%Y.%m.%d %H.%M')),sess.graph)
test_writer = tf.summary.FileWriter('tflogs/{}/test'.format(now.strftime('%Y.%m.%d %H.%M')))

saver = tf.train.Saver()

if os.path.isfile(MODEL_SAVED+'.meta'):
    print("Model found, loading...")
    saver.restore(sess, MODEL_SAVED)
else:
    sess.run(tf.global_variables_initializer())


best_mse = 1e9
batch_size = 128
n_epoch = 1000
step = 0
for epoch in range(n_epoch):
    print("Epoch {}".format(epoch))
    cursor = 0
    while cursor<train_input.shape[0]:
        batch_input = train_input[cursor:cursor+batch_size]
        batch_target = train_target[cursor:cursor+batch_size]
        mask_ = np.repeat(batch_input[:,:,:,0:1],5,axis=3)
        cursor += batch_size
        
        _,summary = sess.run([train_step,merged],feed_dict={x:batch_input,y_:batch_target,mask:mask_})
        train_writer.add_summary(summary, step)
        step += 1

    test_indices = np.arange(len(test_input))
    np.random.shuffle(test_indices) #shuffle test samples
    test_input = test_input[test_indices]
    test_target = test_target[test_indices]

    batch_test_input = test_input[:batch_size]
    batch_test_target = test_target[:batch_size]
    mask_ = np.repeat(batch_test_input[:,:,:,0:1],5,axis=3)
    accuracy_,mse_,summary = sess.run([accuracy,mse,merged],feed_dict={x:batch_test_input,y_:batch_test_target,mask:mask_})
    print("MSE {}".format(mse_))
    print("Accuracy {}".format(accuracy_))
    if mse_<best_mse:
        best_mse = mse_
        print("MSE Improved, saving model...")
        saver.save(sess, MODEL_SAVED)
    test_writer.add_summary(summary, step)