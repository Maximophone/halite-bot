from hlt2 import *
import utils2
import logging
import random
import math
import json
import os,sys
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

random.seed(0)
logging.basicConfig(filename='rlbot.log.perm',level=logging.DEBUG)

logging.debug("Start logging")

stdout, stderr = sys.stdout, sys.stderr

sys.stdout, sys.stderr = open("fout","wb"), open("ferr","wb")
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import keras.backend as K
sys.stdout, sys.stderr = stdout, stderr

import numpy as np
from utils2 import Dumper

import signal

# import tensorflow as tf

# def huber_loss(y_true, y_pred):
#     return tf.reduce_sum(tf.sqrt(1 + tf.square(y_pred - y_true)) -1)

def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

logging.debug("Imports done")

MEMORY_FILE = "memory_save_conv_7"
BRAIN_FILE = "brain_save_conv_7"
EPSILON_FILE = "epsilon_save_conv_7"

class Brain:
    def __init__(self, stateCnt, actionCnt,load=False, dim=None):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.dim = dim

        self.model = self._createModel()
        self.frozen_model = self._createModel()
        if load:
            self.load()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = Sequential()

        # model.add(Reshape(target_shape=dim, input_shape=(self.stateCnt,)))
        # model.add(Convolution2D(1, 3, 3, activation="relu", border_mode='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Convolution2D(1, 3, 3, activation="relu", border_mode='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Flatten())
        # model.add(Dense(16, 3, 3, border_mode='same', input_shape=dim))
        model.add(Dense(output_dim=64, activation='relu', input_shape=(self.stateCnt,)))
        model.add(Dense(output_dim=64, activation='relu'))
        model.add(Dense(output_dim=64, activation='relu'))
        model.add(Dense(output_dim=self.actionCnt, activation='linear'))

        # opt = Adam(lr=0.00025)
        model.compile(loss=hubert_loss, optimizer='nadam')

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, frozen=False):
        if frozen:
            return self.frozen_model.predict(s)
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

    def save(self):
        self.model.save_weights(BRAIN_FILE,overwrite=True)

    def load(self):
        self.model.load_weights(BRAIN_FILE)
        self.frozen_model.load_weights(BRAIN_FILE)



#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity, load=False):
        self.capacity = capacity
        if load:
            self.load()

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def save(self):
        with open(MEMORY_FILE,"wb") as f:
            pickle.dump(self.samples,f,protocol=2)

    def load(self):
        with open(MEMORY_FILE,"rb") as f:
            self.samples = pickle.load(f)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 500000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1.
MIN_EPSILON = 0.01
LAMBDA = 0.00008      # speed of decay

class Agent:

    def __init__(self, stateCnt, actionCnt, load_memory=False, load_brain=False, load_epsilon=True, dim=None):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt


        self.brain = Brain(stateCnt, actionCnt, load=load_brain, dim=dim)
        self.memory = Memory(MEMORY_CAPACITY, load=load_memory)
        if load_epsilon:
            with open(EPSILON_FILE,"rb") as f:
                self.steps,self.epsilon = [float(x) for x in f.read().split(',')]
        else:
            self.steps,self.epsilon = 0.,MAX_EPSILON

    def save(self):
        self.memory.save()
        self.brain.save()
        with open(EPSILON_FILE,"wb") as f:
            f.write(str(self.steps)+','+str(self.epsilon))
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, frozen=True)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s,a,r,s_ = o
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

def get_state(gamemap,my_id,square,dist):
    x0,y0 = square.x,square.y
    w,h = gamemap.width,gamemap.height
    arr = np.array([
            (
                sq.owner==my_id,
                (sq.owner!=0) & (sq.owner!=my_id),
                sq.production/20.,
                sq.strength/255.,
            )
            for sq in gamemap]).reshape((w,h,4))
    return np.take(np.take(arr,
        np.arange(-dist,dist + 1)+x0,axis=1,mode='wrap'),
        np.arange(-dist,dist + 1)+y0,axis=0,mode='wrap').flatten()

def compute_reward(gamemap,my_id):
    return sum([
        square.production/20. 
        for square 
        in gamemap 
        if square.owner == my_id])# * len([sq for sq in gamemap if sq.owner == my_id])

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self,visible_distance):
        self.visible_distance = visible_distance
        self.done = True
        self.total_reward = 0

    def reset(self):
        self.done = False
        my_id,gamemap = get_init()
        logging.debug("Initialisation")
        send_init("RLBot1")
        gamemap.get_frame()
        return my_id,gamemap
        
    def step(self,gamemap,my_id,moves):
        send_frame(moves)
        gamemap.get_frame()
        return gamemap, compute_reward(gamemap,my_id)

    def run(self, agent):
        dist = self.visible_distance
        my_id, gamemap = self.reset()
        R = 0 
        done = False

        dumper = Dumper('gamemap','rlbot1',on=True)

        turn = 0
        while True:
            dumper.dump((my_id,gamemap),turn)
            moves = []
            states = []
            actions = []
            squares = []
            move_blocked = False
            for square in gamemap:
                if square.owner==my_id:
                    squares.append(square)

                    s = get_state(gamemap,my_id,square,dist)
                    states.append(s)

                    a = agent.act(s)
                    actions.append(a)

                    # if should_move(square,a,gamemap):
                    moves.append(Move(square,a))
                    # else:
                    #     moves.append(Move(square,STILL))
                    #     move_blocked = True

            # logging.debug("Actions: "+str(actions))
            gamemap, r = self.step(gamemap,my_id,moves)
            # logging.debug("Reward: "+str(r))

            if done: # terminal state
                s_ = None

            for square,s,a in zip(squares,states,actions):
                if done:
                    s_ = None
                else:
                    s_ = get_state(gamemap,my_id,square,dist)
                # if not move_blocked:
                agent.observe( (s, a, r, s_) )
            
            agent.replay()

            s = s_
            self.total_reward += r

            if done:
                break

            # agent.save()
            turn+=1

def should_move(square,d,gamemap):
    if square.strength <= 6*square.production:
        return False
    new_square = gamemap.get_target(square,d)
    if square.strength < new_square.strength:
        return False
    return True
#-------------------- MAIN ----------------------------
if __name__ == '__main__':
    
    VISIBLE_DISTANCE = 4

    env = Environment(visible_distance=VISIBLE_DISTANCE)

    stateCnt  = (2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)*4
    actionCnt = 5
    dim = (2*VISIBLE_DISTANCE+1,2*VISIBLE_DISTANCE+1,4)

    logging.debug("Creating agent")
    agent = Agent(
        stateCnt, 
        actionCnt,
        load_memory = os.path.isfile(MEMORY_FILE),
        load_brain = os.path.isfile(BRAIN_FILE),
        load_epsilon = os.path.isfile(EPSILON_FILE),
        dim = dim)

    logging.debug("Running")

    def signal_term_handler(signal, frame):
        logging.debug("Total Reward: {}".format(env.total_reward))
        logging.debug("Saving agent")
        agent.save()
        sys.exit(0)
     
    signal.signal(signal.SIGTERM, signal_term_handler)

    env.run(agent)