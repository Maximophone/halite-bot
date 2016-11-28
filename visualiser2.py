from hlt2 import *
import utils
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import numpy as np
import _pickle as pickle
import os

def draw_arrow(d,res=16):
    pattern = (0,0,0,1)
    if res!=16:
        raise Exception('Resolution must be 16')
    dot = np.zeros((16,16,4))
    arrow = np.zeros((16,16,4))
    arrow[7:9,2:14]=pattern
    arrow[6:10,12]=pattern
    arrow[5:11,11]=pattern
    arrow[4:12,10]=pattern
    if d == 0:
        return arrow.transpose((1,0,2))[::-1,:]
    elif d == 1:
        return arrow
    elif d == 2:
        return arrow.transpose((1,0,2))
    elif d == 3:
        return arrow[:,::-1]
    elif d == 4:
        return dot
    
arrows = {i:draw_arrow(i) for i in range(5)}

def grad_red(x):
    return (x,0,0,x)

def grad_black(x):
    return (1-x,1-x,1-x,x)

def int_col(x):
    d = {
        0:(0,0,0,1),
        1:(0,1,0,1),
        2:(0,0,1,1),
        3:(0,1,1,1),
        4:(0,0.5,1,1),
        5:(0,1,0.5,1),
        6:(0.2,0.7,1,1),
        7:(0.7,0.2,1,1),
    }
    return d[x]

def compute_im(data,quantity,res=16,function=None,style='heat',cmap=grad_red):
    if function is None:
        function = lambda x:x if x is not None else 0
    gamemap = data['gamemap']
    im = np.zeros((gamemap.height*res,gamemap.width*res,4))
    qmap = {}
    
    # print(quantity)
    # print(data.keys())
    quantity_map = data.get(quantity,{square:getattr(square,quantity,None) for square in gamemap})

    for square in gamemap:
        qmap[(square.y,square.x)] = float(function(quantity_map[square]))

    qmin,qmax = min(qmap.values()),max(qmap.values())
    
    for square in gamemap:
        x = square.x
        y = square.y
        if style == 'heat':
            im[y*res:y*res+res,x*res:x*res+res]=cmap((qmap[(y,x)]-qmin)/(qmax-qmin))
        elif style == 'colors':
            im[y*res:y*res+res,x*res:x*res+res]=cmap(qmap[(y,x)])
        elif style == 'squares':
            qnormed = (qmap[(y,x)]-qmin)/(qmax-qmin)
            square_size = qnormed*(res-2)+1
            start = int((res-square_size)/2)
            end = int((res-square_size)/2 + square_size)
            im[y*res+start:y*res+end,x*res+start:x*res+end]=(1,1,1,1)
        elif style == 'arrows':
            im[y*res:y*res+res,x*res:x*res+res]=arrows[int(qmap[(y,x)])]
    return im

def plot_im(im,alpha):
    plt.imshow(im,alpha=alpha)

def plot_quantity(gameMap,quantity,res=16,function=None,style='heat',alpha=1.,cmap=grad_red):
    if function is None:
        function = lambda x:x
    im = np.zeros((gameMap.height*res,gameMap.width*res,4))
    qmap = {}
    
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            qmap[(y,x)] = float(function(getattr(gameMap.getSite(loc),quantity)))

    qmin,qmax = min(qmap.values()),max(qmap.values())
    
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            if style == 'heat':
                im[y*res:y*res+res,x*res:x*res+res]=cmap((qmap[(y,x)]-qmin)/qmax)
            elif style == 'colors':
                im[y*res:y*res+res,x*res:x*res+res]=cmap(qmap[(y,x)])
            elif style == 'squares':
                qnormed = (qmap[(y,x)]-qmin)/(qmax-qmin)
                square_size = qnormed*(res-2)+1
                start = int((res-square_size)/2)
                end = int((res-square_size)/2 + square_size)
                im[y*res+start:y*res+end,x*res+start:x*res+end]=(1,1,1,1)
            elif style == 'arrows':
                im[y*res:y*res+res,x*res:x*res+res]=arrows[int(qmap[(y,x)])]
                
    plt.imshow(im,alpha=alpha)

def plot_ims(gameMap=None,state=None,cached_ims = None):
    ims = []
    if not cached_ims:
        ims.append(compute_im(gameMap,state['color'],style='colors',cmap=int_col) if state.get('color') else None)
        ims.append(compute_im(gameMap,state['heat'],style='heat',cmap=grad_red,function=lambda x: 0. if x is None else x) if state.get('heat') else None)
        ims.append(compute_im(gameMap,state['size'],style='squares') if state.get('size') else None)
        ims.append(compute_im(gameMap,state['arrows'],style='arrows',function=lambda x: 0. if x is None else x) if state.get('arrows') else None)
    else:
        ims = cached_ims
    alphas = [1.,1.,0.8,0.2]
    for im,alpha in zip(ims,alphas):
        if im is not None:
            plot_im(im,alpha=alpha)
    return ims

all_gameMaps = [f for f in os.listdir('dumps') if f.split('_')[0] == 'gameMap']
all_bots = set([x.split('_')[1] for x in all_gameMaps])
steps = {b:[int(gm.split('_')[2]) for gm in all_gameMaps if gm.split('_')[1] == b] for b in all_bots}

def get_gamemap(step,botname):
    return [x for x in all_gameMaps if int(x.split('_')[2])==step and x.split('_')[1]==botname][0]

step_to_gameMap  = {int(x.split('_')[2]):x for x in all_gameMaps}
# steps = [x for x in step_to_gameMap.keys()]
# steps.sort()
# steps = [str(x) for x in steps]

im_cache = {}