from hlt import *
from networking import *
import logging
import time

logging.basicConfig(filename='last_run.log',level=logging.DEBUG)
logging.debug('Hello')

alpha = 1.

def weightedChoice(choices,weights):
    interval = sum(weights)
    x = random.random()*interval
    cum_weights = 0
    for i,w in enumerate(weights):
        cum_weights += w
        if x <= cum_weights:
            return choices[i]

def attractiveness(site):
    return alpha*site.production/site.strength/12.+ (1-alpha)*(site.production-1.)/11.

def mapAttractiveness(myID,gameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            site = gameMap.getSite(Location(x,y))
            if site.owner == myID:
                site.attractiveness = 0
            site.attractiveness = attractiveness(site)

def mapSmoothedAttractiveness(myID,gameMap,kernel):
    WCARDINALS = zip(kernel,CARDINALS)
    sum_weights = sum(kernel) + 1
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            site = gameMap.getSite(loc)
            smoothedAttractiveness = site.attractiveness
            for w,d in WCARDINALS:
                smoothedAttractiveness+=w*gameMap.getSite(loc,d).attractiveness
            site.smoothedAttractiveness = smoothedAttractiveness/sum_weights

def getRegion(x,y,radius,height,width):
    locs = []
    for a in range(-radius,radius+1):
        rb = radius - abs(a)
        for b in range(-rb,rb+1):
            locs.append(Location(x+a,y+b).rectify(height,width))
    return locs

def findStart(myID,gameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            if gameMap.getSite(Location(x,y)).owner == myID:
                return Location(x,y) 

def findLocalMaxSmootherAttr(center,myID,gameMap,regionRadius=10):
    locs = getRegion(center.x,center.y,regionRadius,gameMap.height,gameMap.width)
    attrs = [gameMap.getSite(l).smoothedAttractiveness for l in locs]
    return locs[argsort(attrs,reverse=True)[0]]

def argsort(seq,reverse=False):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

def argmin(seq):
    min_val = None
    min_i = 0
    for i,x in enumerate(seq):
        if min_val is None or x < min_val:
            min_val = x
            min_i = i
    return min_i

def argmax(seq):
    max_val = None
    max_i = 0
    for i,x in enumerate(seq):
        if max_val is None or x > max_val:
            max_val = x
            max_i = i
    return max_i

epsilon = 0.000001

def findBestDirections(locA,locB,gameMap):
    dirs = [epsilon for _ in range(5)]
    locCenter = Location(gameMap.width/2,gameMap.height/2)
    locA_centered = Location(
        locA.x+(locCenter.x-locB.x),
        locA.y+(locCenter.y-locB.y),
        ).rectify(gameMap.height,gameMap.width)
    if locA_centered.x < locCenter.x:
        dirs[EAST] = 1
    elif locA_centered.x > locCenter.x:
        dirs[WEST] = 1
    if locA_centered.y < locCenter.y:
        dirs[SOUTH] = 1
    elif locA_centered.y > locCenter.y:
        dirs[NORTH] = 1
    return dirs

def mapGlobalDirections(target,gameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            gameMap.getSite(loc).globalDirs = findBestDirections(loc,target,gameMap)

def mapLocalDirections(gameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            site = gameMap.getSite(loc)
            dirs = [gameMap.getSite(loc,d).attractiveness for d in DIRECTIONS]
            site.localDirs = dirs

def mapFinalDirections(gameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            site = gameMap.getSite(loc)
            site.finalDirs = [a*b**2 for a,b in zip(site.globalDirs,site.localDirs)]

def mapMoves(gameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            max_d = 1
            max_attr = 0
            moves = [(gameMap.getSite(loc,d).attractiveness,d) for d in CARDINALS]
            moves.sort()
            moves.reverse()
            site = gameMap.getSite(loc)
            site.moves = [m[1] for m in moves]

def find_frontier(start,myID,gameMap):
    explorers = [start,start,start,start]
    for _ in range(1000):
        for d in range(4):
            explorers[d] = gameMap.getLocation(explorers[d],d+1)
            if gameMap.getSite(explorers[d]).owner != myID:
                return d+1

def find_centroid(myID,gameMap):
    #Finds approximate center of my territory
    x_center,y_center,n_x,n_y = 0,0,0.,0.
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            site = gameMap.getSite(Location(x,y))
            if site.owner == myID:
                x_center += x
                y_center += y
                n_x += 1
                n_y += 1
    return Location(round(x_center/n_x),round(y_center/n_y))

def create_move_map(centroid,gameMap):
    #Creates a map of movements to get away from the centroid
    moveMap = [[0 for _ in range(gameMap.width)] for _ in range(gameMap.height)] 


def leave_center(start,centroid,gameMap):
    max_d = 1
    max_dist = 0
    for d in CARDINALS:
        dist = gameMap.getDistance(gameMap.getLocation(start,d),centroid)
        if dist > max_dist:
            max_dist = dist
            max_d = d
    return max_d

def find_frontier(myID,gameMap):
    frontier = []
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            site = gameMap.getSite(loc)
            if site.owner == myID:
                continue
            for d in CARDINALS:
                if gameMap.getSite(loc,d).owner != myID:
                    continue
                else:
                    frontier.append(loc)
                    break
    return frontier
                
def dist_frontier(frontier,myID,gameMap,dist=-1):
    if not frontier:
        return
    if dist > 8:
        return
    new_frontier = []
    for loc in frontier:
        for c in CARDINALS:
            new_loc = gameMap.getLocation(loc,c)
            site = gameMap.getSite(new_loc)
            if site.owner!=myID or site.dist_frontier is not None:
                continue
            site.dist_frontier = dist+1
            new_frontier.append(new_loc)
    dist_frontier(new_frontier,myID,gameMap,dist=dist+1)

def custom_min(x):
    v_min = 999999
    a_min = None
    for v,a in x:
        if v<v_min:
            v_min = v
            a_min = a
    return a_min

def mapFrontierDirections(myID,gameMap,oldGameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            site = gameMap.getSite(loc)
            if site.owner == myID and site.dist_frontier is None:
                site.dist_frontier = oldGameMap.getSite(loc).dist_frontier

    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            site = gameMap.getSite(loc)
            if site.owner != myID or site.dist_frontier == 0:
                site.frontierDir = None
            else:
                fmoves = [(gameMap.getSite(loc,d).dist_frontier,d) for d in CARDINALS]
                random.shuffle(fmoves)  
                site.frontierDir = custom_min(fmoves)

def mapDistFrontier(myID,gameMap):
    frontier = find_frontier(myID,gameMap)
    dist_frontier(frontier,myID,gameMap)

def str_attrMap(gameMap):
    chars = []
    for y in range(gameMap.height):
        charsRow = []
        for x in range(gameMap.width):
            charsRow.append(str("{:.0f}".format(9*gameMap.getSite(Location(x,y)).attractiveness)))
        chars.append(charsRow)
    return '\n'.join([' '.join(charsRow) for charsRow in chars])

def str_moveMap(gameMap):
    chars = []
    for y in range(gameMap.height):
        charsRow = []
        for x in range(gameMap.width):
            charsRow.append(str(gameMap.getSite(Location(x,y)).moves[0]))
        chars.append(charsRow)
    return '\n'.join([''.join(charsRow) for charsRow in chars])

def reconstruct_path(came_from,node):
    pass

def cost(site):
    return site.strength/float(site.production)

def min_func(seq,function):


def a_star(start,end,gameMap):
    closed_set = set()
    open_set = set([start])
    came_from = {}
    g_score = {}
    g_score[start] = 0
    f_score = {}
    f_score[start] = gameMap.getDistance(start,end)

    while open_set:
        current = min(open_set,key=lambda x:f_score[x])

        if current == end:
            return reconstruct_path(came_from,current)
        
        open_set.remove(current)
        closed_set.add(current)

        for d in CARDINALS:
            neighbor = gameMap.getLocation(current,d)
            if neighbor in closed_set:
                continue
            site = gameMap.getSite(neighbor)
            tentative_g_score = g_score[current] + cost(site)

            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score[neighbor]:
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + gameMap.getDistance(neighbor,end)

    return False


if __name__ == "__main__":
    myID, gameMap = getInit()

    mapAttractiveness(myID,gameMap)
    mapSmoothedAttractiveness(myID,gameMap,kernel=[1.5,1.5,1.5,1.5])
    start = findStart(myID,gameMap)
    target = findLocalMaxSmootherAttr(start,myID,gameMap,regionRadius=9)
    startDirs = findBestDirections(start,target,gameMap)
    extendDirs = [startDirs[0]] + [startDirs[-1]] + startDirs[1:-1]
    mapGlobalDirections(target,gameMap)
    mapLocalDirections(gameMap)
    mapFinalDirections(gameMap)

    frontier = find_frontier(myID,gameMap)
    dist_frontier(frontier,myID,gameMap)
    # mapMoves(gameMap)

    logging.debug('\n'+str_attrMap(gameMap))
    logging.debug('\n START: ({},{})'.format(start.x,start.y))
    logging.debug('\n TARGET: ({},{})'.format(target.x,target.y))

    originGameMap = gameMap
    oldGameMap = gameMap

    moves_lookup = {(x,y):-1 for x in range(gameMap.width) for y in range(gameMap.height)}

    sendInit("MaximoBot_v0.2")

    target_reached = False
    turn = 0
    while True:
        dtleaving = 0
        moves = []
        gameMap = getFrame()
        # import cPickle as pickle
        # pickle.dump((myID,gameMap),open("test.p",'wb'))
        # raise Exception()
        t0 = time.time()
        frontier = find_frontier(myID,gameMap)
        t1 = time.time()
        dist_frontier(frontier,myID,gameMap)
        t2 = time.time()
        mapFrontierDirections(myID,gameMap,oldGameMap)
        t3 = time.time()
        logging.debug("TURN: {}".format(turn))
        logging.debug("dt1={:.5f}".format(t1-t0))
        logging.debug("dt2={:.5f}".format(t2-t1))
        logging.debug("dt3={:.5f}".format(t3-t2))
        for y in range(gameMap.height):
            for x in range(gameMap.width):
                site = gameMap.getSite(Location(x, y))

                originSite = originGameMap.getSite(Location(x, y))
                if site.owner == myID:
                    if not target_reached and x == target.x and y == target.y:
                        target_reached = True
                    moved = False

                    if not target_reached:
                        d = weightedChoice(DIRECTIONS,originSite.finalDirs)
                    else:
                        if site.frontierDir is not None:
                            #inside
                            d = site.frontierDir
                        else:
                            attr_dirs = [
                                (originGameMap.getSite(Location(x,y),d).attractiveness,d)
                                for d in CARDINALS
                                if gameMap.getSite(Location(x,y),d).owner != myID]
                            d = max(attr_dirs)[1]

                    siteMove = gameMap.getSite(Location(x, y),d)
                    # if siteMove.owner != myID:
                    if not moved and (siteMove.strength > site.strength or site.strength<=5*site.production):
                        moves.append(Move(Location(x, y), STILL))
                        moved = True
                    else:
                        moves.append(Move(Location(x, y), d))
                        moves_lookup[(x,y)] = d
                        moved = True
        t4 = time.time()
        sendFrame(moves)
        oldGameMap = gameMap
        logging.debug("dt4={:.5f}".format(t4-t3))
        turn += 1
