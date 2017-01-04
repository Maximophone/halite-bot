import numpy as np
import matplotlib.pyplot as plt

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
    if d == 1:
        return arrow.transpose((1,0,2))[::-1,:]
    elif d == 2:
        return arrow
    elif d == 3:
        return arrow.transpose((1,0,2))
    elif d == 4:
        return arrow[:,::-1]
    elif d == 0:
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

def compute_im(mat,res=16,function=None,style='heat',cmap=grad_red):
    h = mat.shape[0]
    w = mat.shape[1]
    if function is None:
        function = lambda x:x if x is not None else 0
    im = np.zeros((h*res,w*res,4))
    qmap = {}
    vf = np.vectorize(function)
    
    qmap = vf(mat).astype(float)
    qmin,qmax = qmap.min(),qmap.max()
    
    for y in range(h):
        for x in range(w):
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

def plot_ims(state=None):
    ims = []
    cmap_mapping = {
        'colors':int_col,
        'heat':grad_red
        
    }
    function_mapping = {
        'heat':lambda x:0. if x is None else x,
        'arrows': lambda x: 0. if x is None else x
    }
    alphas = [1.,1.,0.8,0.2]
    for k,v in state.items():
        ims.append(compute_im(v,style=k,cmap=cmap_mapping.get(k),function=function_mapping.get(k)))
    
    for im,alpha in zip(ims,alphas):
        if im is not None:
            plot_im(im,alpha=alpha)
    return ims