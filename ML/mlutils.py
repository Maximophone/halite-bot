import numpy as np
import math

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
    return int(round(L/math.pi/2.*math.atan2(mu_y,mu_x)%L))

def get_centroid(frame):
    X,Y = np.where(frame[:,:,0]==1)
    mX = get_centroid_1D(X,frame.shape[0])
    mY = get_centroid_1D(Y,frame.shape[1])
    return (mX,mY)