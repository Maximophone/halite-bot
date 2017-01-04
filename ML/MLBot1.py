import numpy as np
from hlt2 import *
import os
import logging

import mlutils
import utils2
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(filename='mlbot.logperm',level=logging.DEBUG)

input_shape = (51,51)
input_shape_start = (15,15)

MODEL = sys.argv[1]

def attractiveness_start(square,my_id):
    return square.production/float(square.strength)

def cost(square):
    return square.strength/float(square.production+1)


def gamemap_to_frame(gamemap):
    return np.array([[(
        square.owner,
        square.strength,
        square.production
        ) for square in row] for row in gamemap.contents])

def rebase_moves(moves,centroid,wrap_size,original_shape):
    hstart = wrap_size[0]/2 - centroid[0]
    hend = hstart + original_shape[0]

    wstart = wrap_size[1]/2 - centroid[1]
    wend = wstart + original_shape[1]
    
    return np.take(np.take(moves,
                np.arange(hstart,hend),axis=0,mode='wrap'),
                np.arange(wstart,wend),axis=1,mode='wrap')

if __name__ == '__main__':

    momentum_map = {}
    
    logging.debug("Start logging")

    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = open("fout","wb"), open("ferr","wb")
    from keras.models import load_model
    sys.stdout, sys.stderr = stdout, stderr

    logging.debug("Keras imported")

    model = load_model(MODEL)
    model_start = load_model("ML/model_start_1.h5")

    logging.debug("Models loaded")


    my_id, gamemap = get_init()
    logging.debug("Got init")

    start = utils2.find_start(my_id,gamemap)
    momentum_map[(start.x,start.y)] = STILL
    others_start = [utils2.find_start(player_id,gamemap) for player_id in range(gamemap.starting_player_count+1) if player_id not in (0,my_id)]
    logging.debug("Found Starting Positions")
    logging.debug(start)
    logging.debug(others_start)

    min_distance_to_others = utils2.get_min_distance(start,others_start,gamemap)
    logging.debug("Found Minimum Distance")
    logging.debug(min_distance_to_others)

    attr_map = utils2.map_attractiveness(my_id,gamemap,attractiveness_start)
    logging.debug("Mapped Attractiveness")

    smoothed_attr_map = utils2.smooth_map(attr_map,gamemap,kernel=[1.,1.5])
    logging.debug("Mapped Smoothed Attractiveness")

    target = utils2.find_local_max(start,int(min_distance_to_others/2),smoothed_attr_map,gamemap)
    logging.debug("Found Target")
    logging.debug(target)

    directions_dict,path = utils2.a_star(target,start,gamemap,cost)
    logging.debug("Found Path")

    send_init(MODEL.split('/')[-1])
    logging.debug("Sent init")

    target_reached = False

    turn = 0
    while True:
        if gamemap.get_target(target,4).owner!=0:
            target_reached = True

        turn += 1
        gamemap.get_frame()

        frame = gamemap_to_frame(gamemap)

        iframe = np.empty(shape=frame.shape[:2]+(4,))

        iframe[:,:,0] = frame[:,:,0] == my_id
        iframe[:,:,1] = (frame[:,:,0] != my_id) & (frame[:,:,0] != 0)
        iframe[:,:,2] = frame[:,:,1]/20.
        iframe[:,:,3] = frame[:,:,2]/255.

        centroid = mlutils.get_centroid(iframe)

        if not target_reached:
            moves = []

            for square in gamemap:
                if square.owner != my_id:
                    continue

                wanted_d = directions_dict.get(square)
                if wanted_d == None:
                    wanted_d = STILL
                new_square = gamemap.get_target(square,wanted_d)

                if square.strength <= 6*square.production:
                    d = STILL

                elif new_square.owner != my_id and new_square.strength > square.strength:
                    d = STILL

                else:
                    d = wanted_d

                moves.append(Move(square,d))

        else:

            wframe = mlutils.center_frame(iframe,centroid,wrap_size=input_shape)

            moves_pred = model.predict(np.array([wframe]))
            moves_pred[:,:,:,0]/=5.

            moves_mat = np.argmax(moves_pred,axis=3)[0]

            moves_mat_rebased = rebase_moves(moves_mat,centroid,input_shape,frame.shape)

            moves = [
                Move(square,(moves_mat_rebased[square.y,square.x]-1)%5) 
                for square in gamemap if square.owner==my_id]

            # logging.debug(turn)
            # logging.debug(np.where(moves_mat_rebased>0))
            # logging.debug([(square.x,square.y) for square in gamemap if square.owner==my_id])

        send_frame(moves)

