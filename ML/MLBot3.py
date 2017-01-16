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

MODEL = sys.argv[1]

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
    
    hrange = np.arange(hstart,hend)
    wrange = np.arange(wstart,wend)
    
    return np.take(np.take(moves,
                [x%original_shape[0] for x in hrange],axis=0),
                [x%original_shape[1] for x in wrange],axis=1)

def should_move(square,new_square,map_uptodate,my_id):
    if square.strength <= 3*square.production: 
    #if strength <= 5*production
        return False
    if map_uptodate[new_square].strength > square.strength and map_uptodate[new_square].owner != my_id: 
    #if foreign tile of superior strength
        return False
    if map_uptodate[new_square].strength + square.strength > 255 and map_uptodate[new_square].owner == my_id:
        # logging.debug('prevented move')
    #if friendly tile and sum>255
        return False
    return True

def reroute_moves(moves_map,gamemap,momentum_map):
    map_uptodate = {square:Square(
        square.x, 
        square.y, 
        square.owner, 
        square.strength if square.owner!=my_id else 0., 
        square.production) for square in gamemap}
    sorted_squares = sorted(gamemap,key=lambda sq:sq.strength, reverse=True)
    new_moves = []
    for square in sorted_squares:
        if square.owner != my_id:
            continue
        if square not in moves_map:
            continue
        wanted_move = moves_map[square]
        new_square = gamemap.get_target(square,wanted_move)
        if wanted_move == opp_cardinal[momentum_map[square]]:
            new_move = STILL
        elif should_move(square,new_square,map_uptodate,my_id):
            new_strength = square.strength + map_uptodate[new_square].strength if map_uptodate[new_square].owner==my_id else square.strength-map_uptodate[new_square].strength
            #updating mapping
            map_uptodate[new_square] = map_uptodate[new_square]._replace(strength=new_strength, owner=my_id)
            
            new_move = wanted_move
        else:
            new_move = STILL
        new_moves.append(Move(square,new_move))
    return new_moves

def process_moves(moves,momentum_map,gamemap):
    for move in moves:
        square = move.square
        d = move.direction
        if d == STILL:
            momentum_map[square] = STILL
        else:
            new_square = gamemap.get_target(square,d)
            momentum_map[new_square] = d


if __name__ == '__main__':

    momentum_map = {}
    
    logging.debug("Start logging")

    with open(os.devnull, 'w') as sys.stderr:
        from keras.models import load_model
        model = load_model(MODEL)

    logging.debug("Keras imported")

    model.predict(np.random.randn(1,*(input_shape+(4,)))).shape
    
    logging.debug("Model loaded")

    my_id, gamemap = get_init()
    logging.debug("Got init")

    start = utils2.find_start(my_id,gamemap)
    momentum_map[start] = STILL
    logging.debug("Found Starting Position")
    logging.debug(start)

    send_init(MODEL.split('/')[-1])
    logging.debug("Sent init")

    turn = 0
    while True:
        turn += 1
        gamemap.get_frame()

        frame = gamemap_to_frame(gamemap)

        iframe = np.empty(shape=frame.shape[:2]+(4,))

        iframe[:,:,0] = frame[:,:,0] == my_id
        iframe[:,:,1] = (frame[:,:,0] != my_id) & (frame[:,:,0] != 0)
        iframe[:,:,2] = frame[:,:,1]/20.
        iframe[:,:,3] = frame[:,:,2]/255.

        centroid = mlutils.get_centroid(iframe)

        wframe = mlutils.center_frame(iframe,centroid,wrap_size=input_shape)

        moves_pred = model.predict(np.array([wframe]))
        moves_pred[:,:,:,0]/=100.

        moves_mat = np.argmax(moves_pred,axis=3)[0]

        moves_mat_rebased = rebase_moves(moves_mat,centroid,input_shape,frame.shape)

        moves_map = {
            square:(moves_mat_rebased[square.y,square.x]-1)%5
            for square in gamemap
            if square.owner==my_id
            }

        moves = reroute_moves(moves_map,gamemap,momentum_map)

        process_moves(moves,momentum_map,gamemap)

        send_frame(moves)

