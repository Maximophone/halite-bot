from hlt import *
from networking import *
import logging
import time
import cPickle as pickle

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
    attr_locs = [(gameMap.getSite(l).smoothedAttractiveness,l) for l in locs]
    attr_locs.sort()
    attr_locs.reverse()
    return attr_locs[0][1]

# def findBestDirections(locA,locB,gameMap):
#     dirs = [0 for _ in range(5)]
#     if abs(locB.x-locA.x) <= gameMap.width-abs(locB.x-locA.x):
#         dirs[EAST]=1
#     else:
#         dirs[WEST]=1
#     if abs(locB.y-locA.y) <= gameMap.height-abs(locB.y-locA.y):
#         dirs[SOUTH]=1
#     else:
#         dirs[NORTH]=1
#     return dirs
epsilon = 0.05

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

# def mapFrontierDirections(gameMap):
#     for y in range(gameMap.height):
#         for x in range(gameMap.width):
#             loc = Location(x,y)
#             site = gameMap.getSite(loc)
#             if site.dist_frontier is None or site.dist_frontier <= 0:
#                 site.frontierDir = None
#             else:
#                 fmoves = [(gameMap.getSite(loc,d).dist_frontier,d) for d in CARDINALS]
#                 random.shuffle()
#                 site.frontierDir = min(fmoves)[1]

def custom_min(x):
    v_min = 999999
    a_min = None
    for v,a in x:
        if v<v_min:
            v_min = v
            a_min = a
    return a_min

def mapFrontierDirections(gameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            site = gameMap.getSite(loc)
            if site.dist_frontier is None or site.dist_frontier <= 0:
                site.frontierDir = None
            else:
                fmoves = [(gameMap.getSite(loc,d).dist_frontier,d) for d in CARDINALS]
                random.shuffle(fmoves)  
                site.frontierDir = custom_min(fmoves)

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

def mapDistFrontier(myID,gameMap):
    frontier = find_frontier(myID,gameMap)
    dist_frontier(frontier,myID,gameMap)

# def dist_frontier(loc,myID,gameMap):
#     site = gameMap.getSite(loc)
#     if site.owner != myID:
#         return -1 if site.owner == 0 else -3
#     if site.dist_frontier is not None and site.dist_frontier!=-10:
#         return site.dist_frontier
#     site.dist_frontier=-10
#     values = [
#             dist_frontier(gameMap.getLocation(loc,d),myID,gameMap) 
#             for d in CARDINALS
#             if gameMap.getSite(loc,d).dist_frontier != -10
#         ]
#     if len(values) == 0:
#         site.dist_frontier = None
#         return 999
#     dist = min(values) + 1
#     site.dist_frontier = dist
#     return dist

# def mapDistFrontier(myID,gameMap):
#     for y in range(gameMap.height):
#         for x in range(gameMap.width):
#             loc = Location(x,y)
#             if gameMap.getSite(loc).owner == myID and gameMap.getSite(loc).dist_frontier is None:
#                 dist_frontier(loc,myID,gameMap)


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

if __name__ == "__main__":
    myID, gameMap = getInit()
    with open('gameMap.p','wb') as f:
        pickle.dump((myID,gameMap),f)

    mapAttractiveness(myID,gameMap)
    mapSmoothedAttractiveness(myID,gameMap,kernel=[1.5,1.5,1.5,1.5])
    start = findStart(myID,gameMap)
    target = findLocalMaxSmootherAttr(start,myID,gameMap,regionRadius=9)
    startDirs = findBestDirections(start,target,gameMap)
    extendDirs = [startDirs[0]] + [startDirs[-1]] + startDirs[1:-1]
    mapGlobalDirections(target,gameMap)
    mapLocalDirections(gameMap)
    mapFinalDirections(gameMap)
    # mapMoves(gameMap)

    logging.debug('\n'+str_attrMap(gameMap))
    logging.debug('\n START: ({},{})'.format(start.x,start.y))
    logging.debug('\n TARGET: ({},{})'.format(target.x,target.y))
    # logging.debug('\n'+str_moveMap(gameMap))

    originGameMap = gameMap

    # prodMap = []
    # blurredProdMap = []
    # for y in range(gameMap.height):
    #     prodMapRow = []
    #     blurredProdMapRow = []
    #     for x in range(gameMap.width):
    #         site = gameMap.getSite(Location(x, y))
    #         blurredProd = site.production
    #         for d in CARDINALS:
    #             neighbour_site = gameMap.getSite(Location(x, y),d)
    #             blurredProd += neighbour_site.production
    #         prodMapRow.append(site.production)
    #         blurredProdMapRow.append(blurredProd/5.)
    #     prodMap.append(prodMapRow)
    #     blurredProdMap.append(blurredProdMapRow)

    moves_lookup = {(x,y):-1 for x in range(gameMap.width) for y in range(gameMap.height)}

    sendInit("MaximoBot_v0.2")

    target_reached = False
    # attack = False
    turn = 0
    while True:
        dtleaving = 0
        moves = []
        gameMap = getFrame()
        if turn == 250:
            with open('gameMap250.p','wb') as f:
                pickle.dump((myID,gameMap),f)
        if turn == 251:
            with open('gameMap251.p','wb') as f:
                pickle.dump((myID,gameMap),f)
        t0 = time.time()
        mapDistFrontier(myID,gameMap)
        mapFrontierDirections(gameMap)
        t1 = time.time()
        logging.debug("TURN: {}".format(turn))
        for y in range(gameMap.height):
            for x in range(gameMap.width):
                site = gameMap.getSite(Location(x, y))
                # if target_reached and not attack:
                #     #acquiring target
                #     if site.owner not in (myID,0):
                #         new_target = Location(x,y)
                #         attack = True
                #         dir1 = findBestDirections(target,new_target,gameMap)
                #         dir2 = findBestDirections(start,new_target,gameMap)
                #         # new_dirs = [a+b for a,b in zip(dir1,dir2)]
                #         new_dirs = dir2
                #         logging.debug('\n NEW TARGET: ({},{})'.format(new_target.x,new_target.y))
                #         logging.debug('\n NEW DIRS:'+str(new_dirs))

                originSite = originGameMap.getSite(Location(x, y))
                if site.owner == myID:
                    if not target_reached and x == target.x and y == target.y:
                        target_reached = True
                    moved = False
                    # if moves_lookup[(x,y)] != -1: 
                    #     if site.strength>=5*site.production:
                    #         moves.append(Move(Location(x, y), moves_lookup[(x,y)]))
                    #         moved = True
                    #     else:
                    #         moves.append(Move(Location(x, y), STILL))
                    #         moved = True
                    #     continue
                    if not target_reached:
                        d = weightedChoice(DIRECTIONS,originSite.finalDirs)
                    # elif not attack:
                    #     d = weightedChoice(DIRECTIONS,extendDirs)
                    else:
                        if site.frontierDir is not None:
                            #inside
                            d = site.frontierDir
                            # if site.strength>=5*site.production:
                            #     moves.append(Move(Location(x, y), d))
                            #     moved = True
                            # else:
                            #     moves.append(Move(Location(x, y), STILL))
                            #     moved = True

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
                    # for d in originSite.moves:
                    #     siteMove = gameMap.getSite(Location(x, y),d)
                    #     if siteMove.owner == myID:
                    #         continue
                    #     if siteMove.strength >= site.strength:
                    #         moves.append(Move(Location(x, y), STILL))
                    #         moved = True
                    #     else:
                    #         moves.append(Move(Location(x, y), d))
                    #         moves_lookup[(x,y)] = d
                    #         moved = True
                    # if not moved:
                    #     moves.append(Move(Location(x, y), NORTH if bool(int(random.random() * 2)) else WEST))
                    #     moved = True

                    
                    # if not moved:
                    #     # moves.append(Move(Location(x, y), NORTH if bool(int(random.random() * 2)) else WEST))
                    #     moves.append(Move(Location(x, y), STILL))
                    #     moved = True
                    # if not moved and site.strength<=15:
                    #     moves.append(Move(Location(x, y), STILL))
                    # elif not moved:
                    #     moves.append(Move(Location(x, y), NORTH if bool(int(random.random() * 2)) else WEST))
        t2 = time.time()
        sendFrame(moves)
        logging.debug("dt1={:.5f} dt2={:.5f}".format(t1-t0,t2-t1))
        turn += 1
