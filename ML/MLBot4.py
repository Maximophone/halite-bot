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


if __name__ == '__main__':
    
    logging.debug("Start logging")

    with open(os.devnull, 'w') as sys.stderr:
        from keras.models import load_model
        model = load_model(MODEL)

    logging.debug("Keras imported")

    model.predict(np.random.randn(1,*(input_shape+(4,)))).shape
    
    logging.debug("Model loaded")

    my_id, gamemap = get_init()
    logging.debug("Got init")

    send_init(MODEL.split('/')[-1])
    logging.debug("Sent init")

    turn = 0
    while True:
        turn += 1
        gamemap.get_frame()

        logging.debug("Received new frame")

        frame = gamemap_to_frame(gamemap)

        iframe = np.empty(shape=frame.shape[:2]+(4,))

        iframe[:,:,0] = frame[:,:,0] == my_id
        iframe[:,:,1] = (frame[:,:,0] != my_id) & (frame[:,:,0] != 0)
        iframe[:,:,2] = frame[:,:,1]/20.
        iframe[:,:,3] = frame[:,:,2]/255.

        centroid = mlutils.get_centroid(iframe)

        wframe = mlutils.center_frame(iframe,centroid,wrap_size=input_shape)

        logging.debug("Computed wframe")

        moves_pred = model.predict(np.array([wframe]))

        logging.debug("Prediction made")

        # moves_pred[:,:,:,0]/=2.

        moves_mat = np.argmax(moves_pred,axis=3)[0]

        moves_mat_rebased = rebase_moves(moves_mat,centroid,input_shape,frame.shape)

        logging.debug("Rebased moves")

        moves_map = [
            Move(square,(moves_mat_rebased[square.y,square.x]-1)%5)
            for square in gamemap
            if square.owner==my_id
            ]

        logging.debug(moves_map)

        logging.debug("Sending moves")

        send_frame(moves_map)

