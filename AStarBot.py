from hlt import *
from networking import *
import logging
import time

from utils import Dumper

logging.basicConfig(filename='last_run.log',level=logging.DEBUG)
logging.debug('Hello')

alpha = 1.

class TimeTracker(object):
    def __init__(self):
        self.last_time = None
        self.dts = []
        self.names = []

    def track(self,name=None):
        t = time.time()
        if self.last_time is None:
            self.last_time = t
        else:
            self.dts.append(t-self.last_time)
            self.last_time = t
            self.names.append(name)

    def log(self,reset=True):
        for i,(dt,name) in enumerate(zip(self.dts,self.names)):
            logging.debug("dt{}({})={:.5f}".format(i,name,dt))
        if reset:
            self.last_time = None
            self.dts = []
            self.names = []

def getGameMapStats(myID,gameMap):
    total = float(gameMap.width*gameMap.height)
    stats = {
        'n_owners':[0]*10
    }
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            site = gameMap.getSite(Location(x,y))
            stats['n_owners'][site.owner] += 1
    return stats

def getMinDistance(loc,locs,gameMap):
    return min([gameMap.getDistance(l,loc) for l in locs])

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
    radius = int(radius)
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
                
def dist_frontier(frontier,myID,gameMap,dist=-1,penalty=1):
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
            if dist==-1 and gameMap.getSite(loc).strength>150:
                site.dist_frontier+=penalty
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

def directionDist(d1,d2):
    return abs((d2-d1))%4

def momentumChose(prev,moves):
    return min(moves,key=lambda x:directionDist(x,prev))

def mapFrontierDirections(myID,gameMap,oldGameMap,momentumMap):
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
                # fmoves = [(gameMap.getSite(loc,d).dist_frontier,d) for d in CARDINALS]
                # random.shuffle(fmoves)  
                momentum = momentumMap[loc]
                site.frontierDir = min(
                    CARDINALS,
                    key=lambda d:(
                        gameMap.getSite(loc,d).dist_frontier if gameMap.getSite(loc,d).dist_frontier is not None else 0,
                        directionDist(d,momentum)))
                # site.frontierDir = custom_min(fmoves)

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

def reconstruct_path(came_from,current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path

def cost(site):
    return site.strength/float(site.production+1)

def invert_direction(d):
    return ((d-1) + 2)%4 + 1

def is_frontier(loc,player_id):
    site = gameMap.getSite(loc)
    if site.owner == player_id:
        return False
    for d in CARDINALS:
        if gameMap.getSite(loc,d).owner != player_id:
            continue
        else:
            return True
    return False

def frontier_tracking(prev_frontier,myID,gameMap):
    checked = set()
    new_frontier = []
    for loc in prev_frontier:
        for d in DIRECTIONS:
            neighbor = gameMap.getLocation(loc,d)
            if neighbor not in checked:
                checked.add(neighbor)
                if is_frontier(neighbor,myID):
                    new_frontier.append(neighbor)
    return new_frontier

def bounding_box_tracking(prev_bounding_box,frontier,myID,gameMap):
    new_bounding_box = {
        "max_x":prev_bounding_box["max_x"]-1,
        "min_x":prev_bounding_box["min_x"]+1,
        "max_y":prev_bounding_box["max_y"]-1,
        "min_y":prev_bounding_box["min_y"]+1,
    }
    for loc in frontier:
        if loc.x>new_bounding_box["max_x"]:
            new_bounding_box["max_x"] = loc.x
        if loc.x==prev:
            pass


def get_bounding_box(myID,gameMap):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            pass

def smooth_frontier(frontier,gameMap):
    width = gameMap.width
    height = gameMap.height
    lf = len(frontier)
    new_frontier = []
    for i,loc in enumerate(frontier):
        new_loc = Location(
            round(
                0.5*loc.x
                + 0.2*(frontier[(i+1)%lf].x+frontier[(i-1)%lf].x)
                + 0.05*(frontier[(i+2)%lf].x+frontier[(i-2)%lf].x)
                )%width,
            round(
                0.5*loc.y
                + 0.2*(frontier[(i+1)%lf].y+frontier[(i-1)%lf].y)
                + 0.05*(frontier[(i+2)%lf].y+frontier[(i-2)%lf].y)
                )%height,
            )
        new_frontier.append(new_loc)
    return new_frontier

def order_frontier(frontier,gameMap):
    not_visited_set = set(frontier)
    visited_set = set()
    current = frontier[0]
    new_frontier = [current]
    player_id = current.owner
    while len(visited_set) < len(frontier):
        not_visited_set.remove(current)
        visited_set.add(current)
        outsiders = None
        neighbors = [] 
        for hd in HALFCARDINALS:
            loc = gameMap.getLocation(current,hd)
            # if gameMap.getSite(loc).owner != 

#TODO: keep track of previous moves to add momentum and prevent reverting

def a_star(start,end,gameMap):
    closed_set = set()
    open_set = set([start])
    came_from = {}
    directions_dict = {}
    g_score = {}
    g_score[start] = 0
    f_score = {}
    f_score[start] = gameMap.getDistance(start,end)

    while open_set:
        current = min(open_set,key=lambda x:f_score[x])

        if current == end:
            return {k:invert_direction(v) for k,v in directions_dict.items()}, reconstruct_path(came_from,current)
        
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

            directions_dict[neighbor] = d
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + gameMap.getDistance(neighbor,end)

    return False

momentumMap = {}

def shouldMove(site,siteMove,myID):
    if site.strength <= 5*site.production:
        return False
    if siteMove.strength > site.strength and siteMove.owner != myID:
        return False
    if siteMove.strength > site.strength and siteMove.strength + site.strength > 255 and siteMove.owner == myID:
        return False
    return True

if __name__ == "__main__":
    myID, gameMap = getInit()
    gameMapStats = getGameMapStats(myID,gameMap)

    mapAttractiveness(myID,gameMap)
    mapSmoothedAttractiveness(myID,gameMap,kernel=[1.5,1.5,1.5,1.5])

    start = findStart(myID,gameMap)
    others_start = [findStart(player_id,gameMap) for player_id,n in enumerate(gameMapStats['n_owners']) if player_id not in (0,myID) and n>0]
    logging.debug(others_start)
    min_distance_to_others = getMinDistance(start,others_start,gameMap)
    logging.debug(min_distance_to_others)
    momentumMap[start] = STILL
    inner_frontier = [start]
    target = findLocalMaxSmootherAttr(start,myID,gameMap,regionRadius=min_distance_to_others/2)

    directions_dict,path = a_star(target,start,gameMap)


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

    sendInit("AStarBot")

    early_stop = False
    dumps = False
    map_dumper = Dumper('gameMap','astarbot',on=dumps)
    frontier_dumper = Dumper('frontier','astarbot',on=dumps)

    target_reached = False
    turn = 0
    time_tracker = TimeTracker()
    while True:
        dtleaving = 0
        moves = []
        gameMap = getFrame()
        logging.debug("TURN: {}".format(turn))
        # import cPickle as pickle
        # raise Exception()
        time_tracker.track()
        if not early_stop:
            # frontier = find_frontier(myID,gameMap)
            frontier = frontier_tracking(frontier,myID,gameMap)
        time_tracker.track("Find Frontier")
        if not early_stop:
            dist_frontier(frontier,myID,gameMap)
        time_tracker.track("Map Frontier Distance")
        if not early_stop:
            mapFrontierDirections(myID,gameMap,oldGameMap,momentumMap)
        time_tracker.track("Map Frontier Directions")
        # inner_frontier = frontier_tracking(inner_frontier,myID,gameMap)
        # if dumps: pickle.dump(inner_frontier,open("dumps/frontier{}.p".format(turn),'wb'))
        time_tracker.track("Track Frontier")
        # if dumps: pickle.dump((myID,gameMap),open("dumps/gameMap{}.p".format(turn),'wb'))
        map_dumper.dump((myID,gameMap),turn)
        for y in range(gameMap.height):
            for x in range(gameMap.width):
                loc = Location(x,y)
                site = gameMap.getSite(loc)

                originSite = originGameMap.getSite(loc)
                if site.owner == myID:
                    if not target_reached and x == target.x and y == target.y:
                        target_reached = True
                    moved = False

                    if not target_reached:
                        d = directions_dict[loc]
                        # d = weightedChoice(DIRECTIONS,originSite.finalDirs)
                    else:
                        if site.frontierDir is not None:
                            #inside
                            d = site.frontierDir
                        else:
                            attr_dirs = [
                                (originGameMap.getSite(loc,d).attractiveness,d)
                                for d in CARDINALS
                                if gameMap.getSite(loc,d).owner != myID]
                            d = max(attr_dirs)[1]

                    siteMove = gameMap.getSite(loc,d)
                    # if siteMove.owner != myID:
                    if not moved and shouldMove(site,siteMove,myID):
                        moves.append(Move(loc, d))
                        momentumMap[gameMap.getLocation(loc,d)] = d
                        moves_lookup[(x,y)] = d
                        moved = True
                    else:
                        moves.append(Move(loc, STILL))
                        momentumMap[loc] = STILL
                        moved = True

                    # if not moved and ((siteMove.strength > site.strength and siteMove.owner ) or site.strength<=5*site.production):
                    #     moves.append(Move(loc, STILL))
                    #     momentumMap[loc] = STILL
                    #     moved = True

                    # else:
                    #     moves.append(Move(loc, d))
                    #     momentumMap[gameMap.getLocation(loc,d)] = d
                    #     moves_lookup[(x,y)] = d
                    #     moved = True
        time_tracker.track("Main Loop")
        sendFrame(moves)
        oldGameMap = gameMap
        time_tracker.log()
        turn += 1
