from hlt import *
from networking import *
import logging
import utils
import random

random.seed(0)

logging.basicConfig(filename='last_run.log',level=logging.DEBUG)
logging.debug('Hello')


def attractiveness_OLD(site):
    return (255.-site.strength)/255 + site.production/30.

def attractiveness(site):
    # if site.strength>180:
    #     return 0
    return site.production/float(site.strength) if site.strength else site.production

def attractiveness_start(site):
    return site.production/float(site.strength)

def is_frontier(loc,player_id,locsites):
    site = locsites[(loc,0)]
    if site.owner == player_id:
        return False
    for d in CARDINALS:
        if locsites[(loc,d)].owner != player_id:
            continue
        else:
            return True
    return False

def frontier_tracking(prev_frontier,myID,locsmap_d,locsites):
    checked = set()
    new_frontier = []
    for loc in prev_frontier:
        for d in DIRECTIONS:
            neighbor = locsmap_d[(loc,d)]
            if neighbor not in checked:
                checked.add(neighbor)
                if is_frontier(neighbor,myID,locsites):
                    new_frontier.append(neighbor)
                    locsites[(neighbor,0)].is_frontier = True
    return new_frontier

def map_attractiveness_OLD(myID,locslist,locsites,attr):
    for x,y,loc in locslist:
            site = locsites[(loc,0)]
            if site.owner == myID:
                site.attractiveness = -999
            else:
                site.attractiveness = attr(site)
            site.potential_attr = site.attractiveness

def map_attractiveness(myID,locslist,locsites,attr):
    attrmap = {}
    attrmods = {}
    for x,y,loc in locslist:
            site = locsites[(loc,0)]
            if site.owner == myID:
                # site.attractiveness = -999
                attrmods[loc] = -1.
                attrmap[loc] = 0.
            else:
                # site.attractiveness = attr(site)
                attrmods[loc] = 0.
                attrmap[loc] = attr(site)
            # site.potential_attr = site.attractiveness
    max_attr = max(attrmap.values())
    for x,y,loc in locslist:
        locsites[(loc,0)].attractiveness = attrmap[loc]/max_attr + attrmods[loc]
        locsites[(loc,0)].potential_attr = attrmap[loc]/max_attr + attrmods[loc]

def map_potential_attr(frontier,myID,locsmap_d,locsites,visited_set=None,decay=0.1,nesting_level=0,max_nesting=6):
    if not frontier:
        return
    if visited_set is None:
        visited_set = frontier
    new_frontier = set()
    for loc in frontier:
        site = locsites[(loc,0)]
        for d in CARDINALS:
            new_loc = locsmap_d[(loc,d)]
            new_site = locsites[(new_loc,0)]
            if new_site.owner != myID or new_loc in visited_set:
                continue
            new_site.potential_attr = max(site.potential_attr-decay,new_site.potential_attr)
            new_frontier.add(new_loc)
    visited_set.update(new_frontier)
    map_potential_attr(new_frontier,myID,locsmap_d,locsites,visited_set=visited_set,decay=decay,nesting_level=nesting_level+1,max_nesting=max_nesting)

def find_inner(myID,locsites):
    inner = []
    for x,y,loc in locslist:
        if locsites[(loc,0)].owner == myID:
            inner.append(loc)
    return inner

def map_potential(inner,frontier,locsites,decay):
    inner_set = set(inner)
    frontier_set = set(frontier)
    total_set = inner_set.union(frontier_set)
    visited = set()
    sorted_nodes = [(f,locsites[(f,0)].potential_attr) for f in frontier]
    sorted_nodes.sort(key=lambda x:x[1])
    potentials = {loc:None for loc in inner}
    for loc in frontier:
        potentials[loc] = locsites[(loc,0)].potential_attr
    while len(sorted_nodes)>0:
        current,value = sorted_nodes.pop()
        for d in CARDINALS:
            new_loc = locsmap_d[(current,d)]
            if not new_loc in inner_set or new_loc in visited:
                continue
            potentials[new_loc] = potentials[current] - decay
            sorted_nodes.append((new_loc,potentials[new_loc]))
            sorted_nodes.sort(key=lambda x:x[1])
            visited.add(new_loc)
    for loc in inner:
        locsites[(loc,0)].potential_attr = potentials[loc]
    # for loc in inner:
    #     locsites[(loc,0)].direction = max([d for d in CARDINALS], key= lambda x:locsites[(loc,x)].potential_attr)

def attr_direction(loc,d,locsites,momentumMap,momentumTerm=0.5):
    potential = locsites[(loc,d)].potential_attr
    inv_d = utils.invert_direction(d)
    # if d == momentumMap[loc]:
    #     momentum = momentumTerm
    if inv_d == momentumMap[loc]:
        momentum = -momentumTerm
    else:
        momentum = 0.
    return potential + momentum

def map_directions_OLD(myID,locslist,locsites,momentumMap,momentumTerm=0.5):
    for x,y,loc in locslist:
        site = locsites[(loc,0)]
        if site.owner!=myID:
            site.direction=None
            # site.potential_direction=None
            continue
        site.direction = max(CARDINALS,key=lambda d:attr_direction(loc,d,locsites,momentumMap,momentumTerm=momentumTerm))

def map_directions(myID,locslist,locsmap_d,locsites,momentumMap,directions_dict=None,momentumTerm=0.5):
    moves_list = []
    map_old = {loc:(
        locsites[(loc,0)].strength,
        locsites[(loc,0)].production,
        locsites[(loc,0)].owner) for x,y,loc in locslist}
    map_uptodate = {loc:(
        locsites[(loc,0)].strength if locsites[(loc,0)].owner!=myID else 0.,
        locsites[(loc,0)].production,
        locsites[(loc,0)].owner) for x,y,loc in locslist}
    for x,y,loc in locslist:
        site = locsites[(loc,0)]
        if site.owner!=myID:
            site.direction=None
            site.wanted_direction = None
            # site.potential_direction=None
            continue
        # logging.debug(loc)
        if directions_dict is None or directions_dict.get(loc) is None:
            potential_directions = sorted(CARDINALS,reverse=True,key=lambda d:attr_direction(loc,d,locsites,momentumMap,momentumTerm=momentumTerm))[:1]
        else:
            # logging.debug("using directions dict")
            potential_directions = [directions_dict.get(loc,STILL)] # NEED TO FIX THAT
            # except KeyError:
            #     potential_directions = sorted(CARDINALS,reverse=True,key=lambda d:attr_direction(loc,d,locsites,momentumMap,momentumTerm=momentumTerm))[:1]
            # logging.debug(potential_directions)
        moved = False
        site.wanted_direction = potential_directions[0]
        for d in potential_directions:
            new_loc = locsmap_d[(loc,d)]
            if shouldMove(loc,new_loc,map_old,map_uptodate,myID):
                # logging.debug("should move")
                moved = True
                site.direction = d
                new_strength = map_old[loc][0] + map_uptodate[new_loc][0] if map_uptodate[new_loc][2]==myID else map_old[loc][0]-map_uptodate[new_loc][0]
                map_uptodate[new_loc] = (new_strength,map_uptodate[new_loc][1],myID)
                map_uptodate[loc] = (0,map_old[loc][1],myID)
                break
        if not moved:
            # logging.debug("should not move")
            site.direction = STILL
        if utils.invert_direction(site.direction) == momentumMap[loc] and site.direction!=0:
            logging.debug("MOMENTUM REVERSION")
            logging.debug('map')
            logging.debug(momentumMap)
            logging.debug('potential dirs')
            logging.debug(potential_directions)
            logging.debug('momentum')
            logging.debug(momentumMap[loc])
            logging.debug('chosen')
            logging.debug(site.direction)

        moves_list.append((loc,site.direction))
    return moves_list
        #process_move(loc,site.direction,myID,locsmap_d,locsites,momentumMap,moves)
        # logging.debug(site.direction)


def shouldMove(loc,new_loc,map_old,map_uptodate,myID):
    if map_old[loc][0] <= 5*map_old[loc][1]: 
    #if strength <= 5*production
        # logging.debug("won't move because too small")
        return False
    if map_uptodate[new_loc][0] > map_old[loc][0] and map_uptodate[new_loc][2] != myID: 
    #if foreign tile of superior strength
        # logging.debug("wont move because would lose")
        return False
    if map_uptodate[new_loc][0] > map_old[loc][0] and map_uptodate[new_loc][0] + map_old[loc][0] > 255 and map_uptodate[new_loc][2] == myID:
    #if friendly tile and sum>255
        # logging.debug("wont move because would exceed 255")
        return False
    return True

def shouldMove_OLD(site,siteMove,myID):
    if site.strength <= 5*site.production:
        return False
    if siteMove.strength > site.strength and siteMove.owner != myID:
        return False
    if siteMove.strength > site.strength and siteMove.strength + site.strength > 255 and siteMove.owner == myID:
        return False
    return True

def process_movelist(movelist,moves,momentumMap,locsmap_d):
    for loc,d in movelist:
        moves.append(Move(loc, d))
        if d == STILL:
            momentumMap[loc] = STILL
        else:
            momentumMap[locsmap_d[(loc,d)]] = d

def process_move(loc,d,myID,locsmap_d,locsites,momentumMap,moves):
    moves.append(Move(loc, d))
    if d == STILL:
        momentumMap[loc] = STILL
    else:
        momentumMap[locsmap_d[(loc,d)]] = d

def process_move_OLD(loc,d,myID,locsmap_d,locsites,momentumMap,moves):
    site = locsites[(loc,0)]
    siteMove = locsites[(loc,d)]
    if shouldMove(site,siteMove,myID):
        moves.append(Move(loc, d))
        momentumMap[locsmap_d[(loc,d)]] = d
    else:
        moves.append(Move(loc, STILL))
        momentumMap[loc] = STILL

# available_dir = {
#     1:(1,2,4),
#     2:(1,2,3),
#     3:(2,3,4),
#     4:(1,3,4),
#     0:(1,2,3,4)
# }

# def test_frontier_tile(loc,myID,locsites,enemy_attr,decay,depth,n_tries):
#     final_average = 0
#     multipliers = [(1-decay)**i for i in range(depth)]
#     for _ in range(n_tries):
#         visited = set()
#         local_average = 0
#         last_d = 0
#         for m in multipliers:
#             for 



def adjust_frontier_potential(frontier,myID,locsites,turn,enemy_attr=1.):
    enemy_detected = {loc:False for loc in frontier}
    for loc in frontier:
        site = locsites[(loc,0)]
        for d in CARDINALS:
            newsite = locsites[(loc,d)]
            if newsite.owner not in (0,myID):
                enemy_detected[loc] = True
                site.potential_attr += enemy_attr
    # if sum(enemy_detected.values())>0:
    #     for loc in frontier:
    #         if not enemy_detected[loc]:
    #             locsites[(loc,0)].potential_attr = 0.
    #         else:
    #             logging.debug("enemy detected")
    #             locsites[(loc,0)].enemy_detected = True

def cost(site):
    return site.strength/float(site.production+1)

target_reached = False

momentumMap = {}

if __name__ == "__main__":
    myID, gameMap = getInit()

    logging.debug("Init received")

    locslist = [(x,y,Location(x,y)) for x in range(gameMap.width) for y in range(gameMap.height)]
    locsmap = {(x,y):Location(x,y) for x in range(gameMap.width) for y in range(gameMap.height)}
    locsmap_d = {(loc,d):gameMap.getLocation(loc,d) for x,y,loc in locslist for d in DIRECTIONS}
    locsites = {(loc,d):gameMap.getSite(loc,d) for x,y,loc in locslist for d in DIRECTIONS}

    gameMapStats = utils.getGameMapStats(myID,gameMap)
    logging.debug("Computed map stats")

    frontier = utils.find_frontier(myID,gameMap)
    logging.debug("Found frontier")

    start = utils.findStart(myID,gameMap)
    momentumMap[start] = STILL
    others_start = [utils.findStart(player_id,gameMap) for player_id,n in enumerate(gameMapStats['n_owners']) if player_id not in (0,myID) and n>0]
    logging.debug("Found Starting Positions")
    logging.debug(start)
    logging.debug(others_start)

    min_distance_to_others = utils.getMinDistance(start,others_start,gameMap)
    logging.debug("Found Minimum Distance")
    logging.debug(min_distance_to_others)

    map_attractiveness(myID,locslist,locsites,attractiveness_start)
    logging.debug("Mapped Attractiveness")

    utils.mapSmoothedAttractiveness(myID,gameMap,kernel=[1.5,1.5,1.5,1.5])
    logging.debug("Mapped Smoothed Attractiveness")

    target = utils.findLocalMaxSmootherAttr(start,myID,gameMap,regionRadius=min_distance_to_others/2)
    logging.debug("Found Target")
    logging.debug(target)

    directions_dict,path = utils.a_star(target,start,gameMap,cost)
    logging.debug("Found Path")

    sendInit("ForwardFrontierBot")

    logging.debug("Init sent")
    
    decay = 0.1
    momentumTerm = 1000.
    enemy_attr = 0.1 #0.5 works well too

    turn = 0
    time_tracker = utils.TimeTracker(logging)
    game_dumper = utils.Dumper('gameMap','smartfrontier2',on=False)

    while True:
        # enemy_attr = 0.5 if turn>= 100 else -0.5
        moves = []
        gameMap = getFrame()
        logging.debug("TURN: {}".format(turn))

        time_tracker.track()

        locsites = {(loc,d):gameMap.getSite(loc,d) for x,y,loc in locslist for d in DIRECTIONS}
        time_tracker.track("Building access dictionaries")

        frontier = set(frontier_tracking(frontier,myID,locsmap_d,locsites))
        time_tracker.track("Tracking frontier")

        inner = set(find_inner(myID,locsites))
        time_tracker.track("Finding inner")

        map_attractiveness(myID,locslist,locsites,attractiveness)
        time_tracker.track("Map Attr")

        adjust_frontier_potential(frontier,myID,locsites,turn,enemy_attr=enemy_attr)
        time_tracker.track("Adjusting Frontier Potential Attr")

        map_potential(inner,frontier,locsites,decay=decay)
        time_tracker.track("Map Potential Attr")

        if target_reached:
            movelist = map_directions(myID,locslist,locsmap_d,locsites,momentumMap,momentumTerm=momentumTerm)
        else:
            movelist = map_directions(myID,locslist,locsmap_d,locsites,momentumMap,directions_dict,momentumTerm=momentumTerm)
        time_tracker.track("Map Directions")

        process_movelist(movelist,moves,momentumMap,locsmap_d)

        game_dumper.dump((myID,gameMap),turn)

        if locsites[(target,0)].owner==myID:
            target_reached = True

        # for y in range(gameMap.height):
        #     for x in range(gameMap.width):

        #         loc = locsmap[(x,y)]
        #         site = locsites[(loc,0)]

        #         if site.owner == myID:
        #             if not target_reached and target == loc:
        #                 target_reached = True

        #             # if not target_reached:
        #             #     chosen_d = directions_dict[loc]
        #             # else:
        #             #     chosen_d = site.direction

        #             process_move(loc,site.direction,myID,locsmap_d,locsites,momentumMap,moves)

        time_tracker.track("Processing other moves")
        sendFrame(moves)
        time_tracker.log()
        turn += 1
