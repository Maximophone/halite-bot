from hlt import *
from networking import *
import logging
import utils

logging.basicConfig(filename='last_run.log',level=logging.CRITICAL)
logging.debug('Hello')


def attractiveness(site):
    return (255.-site.strength)/255 + site.production/30.

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

def map_attractiveness(myID,gameMap,attr):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = Location(x,y)
            site = gameMap.getSite(loc)
            if site.owner == myID:
                site.attractiveness = -999
            else:
                site.attractiveness = attr(site)
            site.potential_attr = site.attractiveness

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

def map_directions(myID,locsmap,locsites,momentumMap,momentumTerm=0.5):
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            loc = locsmap[(x,y)]
            site = locsites[(loc,0)]
            if site.owner!=myID:
                site.direction=None
                # site.potential_direction=None
                continue
            site.direction = max(CARDINALS,key=lambda d:attr_direction(loc,d,locsites,momentumMap,momentumTerm=momentumTerm))
                
def shouldMove(site,siteMove,myID):
    if site.strength <= 5*site.production:
        return False
    if siteMove.strength > site.strength and siteMove.owner != myID:
        return False
    if siteMove.strength > site.strength and siteMove.strength + site.strength > 255 and siteMove.owner == myID:
        return False
    return True

def process_move(loc,d,myID,locsmap_d,locsites,momentumMap,moves):
    site = locsites[(loc,0)]
    siteMove = locsites[(loc,d)]
    if shouldMove(site,siteMove,myID):
        moves.append(Move(loc, d))
        momentumMap[locsmap_d[(loc,d)]] = d
    else:
        moves.append(Move(loc, STILL))
        momentumMap[loc] = STILL

def adjust_frontier_potential(frontier,myID,gameMap,turn,enemy_attr=1.):
    for loc in frontier:
        site = gameMap.getSite(loc)
        for d in CARDINALS:
            newsite = gameMap.getSite(loc,d)
            if newsite.owner not in (0,myID):
                site.potential_attr += enemy_attr

def cost(site):
    return site.strength/float(site.production+1)

target_reached = False

momentumMap = {}

if __name__ == "__main__":
    myID, gameMap = getInit()

    logging.debug("Init received")

    locsmap = {(x,y):Location(x,y) for x in range(gameMap.width) for y in range(gameMap.height)}
    locsmap_d = {(loc,d):gameMap.getLocation(loc,d) for loc in locsmap.values() for d in DIRECTIONS}

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

    map_attractiveness(myID,gameMap,attractiveness_start)
    logging.debug("Mapped Attractiveness")

    utils.mapSmoothedAttractiveness(myID,gameMap,kernel=[1.5,1.5,1.5,1.5])
    logging.debug("Mapped Smoothed Attractiveness")

    target = utils.findLocalMaxSmootherAttr(start,myID,gameMap,regionRadius=min_distance_to_others/2)
    logging.debug("Found Target")
    logging.debug(target)

    directions_dict,path = utils.a_star(target,start,gameMap,cost)
    logging.debug("Found Path")

    sendInit("SmartFrontierBot2")

    logging.debug("Init sent")
    
    decay = 0.5
    momentumTerm = 20.
    enemy_attr = 1. #0.5 works well too

    turn = 0
    time_tracker = utils.TimeTracker(logging)
    game_dumper = utils.Dumper('gameMap','smartfrontier',on=False)

    while True:
        moves = []
        gameMap = getFrame()
        logging.debug("TURN: {}".format(turn))

        time_tracker.track()

        locsites = {(loc,d):gameMap.getSite(loc,d) for loc in locsmap.values() for d in DIRECTIONS}
        time_tracker.track("Building access dictionaries")

        frontier = set(frontier_tracking(frontier,myID,locsmap_d,locsites))
        time_tracker.track("Tracking frontier")

        map_attractiveness(myID,gameMap,attractiveness)
        time_tracker.track("Map Attr")

        adjust_frontier_potential(frontier,myID,gameMap,turn,enemy_attr=enemy_attr)
        time_tracker.track("Adjusting Frontier Potential Attr")

        map_potential_attr(frontier,myID,locsmap_d,locsites,decay=decay)
        time_tracker.track("Map Potential Attr")

        map_directions(myID,locsmap,locsites,momentumMap,momentumTerm=momentumTerm)
        time_tracker.track("Map Directions")

        game_dumper.dump((myID,gameMap),turn)

        for y in range(gameMap.height):
            for x in range(gameMap.width):

                loc = locsmap[(x,y)]
                site = locsites[(loc,0)]

                if site.owner == myID:
                    if not target_reached and target == loc:
                        target_reached = True

                    if not target_reached:
                        chosen_d = directions_dict[loc]
                    else:
                        chosen_d = site.direction
                    process_move(loc,chosen_d,myID,locsmap_d,locsites,momentumMap,moves)

        time_tracker.track("Processing other moves")
        sendFrame(moves)
        time_tracker.log()
        turn += 1
