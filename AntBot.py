from hlt import *
from networking import *
import logging
from utils import *

# import cPickle as pickle

logging.basicConfig(filename='last_run.log',level=logging.DEBUG)
logging.debug('Hello')


def attractiveness(site):
    return site.production/float(site.strength+1)

attr_ennemy = 15

def direct_attractiveness(site,myID):
    if site.owner==myID:
        return 0
    elif site.owner!=0:
        return attr_ennemy/float(site.strength+1)
    else:
        return site.production/float(site.strength+1)

def cost(site):
    return site.strength/float(site.production+1)

def shouldMove(site,siteMove,myID):
    if site.strength <= 5*site.production:
        return False
    if siteMove.strength > site.strength and siteMove.owner != myID:
        return False
    if siteMove.strength > site.strength and siteMove.strength + site.strength > 255 and siteMove.owner == myID:
        return False
    return True

def process_move(loc,d,myID,gameMap,momentumMap,moves):
    site = gameMap.getSite(loc)
    siteMove = gameMap.getSite(loc,d)
    if shouldMove(site,siteMove,myID):
        moves.append(Move(loc, d))
        momentumMap[gameMap.getLocation(loc,d)] = d
    else:
        moves.append(Move(loc, STILL))
        momentumMap[loc] = STILL

momentumMap = {}
suggested_attractiveness = {}

if __name__ == "__main__":
    myID, gameMap = getInit()

    logging.debug("Init received")

    frontier = find_frontier(myID,gameMap)

    logging.debug("Found frontier")

    sendInit("AntBot")

    logging.debug("Init sent")

    dumps = True

    decay = 0.1

    turn = 0
    time_tracker = TimeTracker()
    while True:
        moves = []
        gameMap = getFrame()
        logging.debug("TURN: {}".format(turn))

        time_tracker.track()
        frontier = frontier_tracking(frontier,myID,gameMap)
        frontier_set = set(frontier)

        time_tracker.track("Tracking frontier")

        for loc in frontier:
            chosen_d = max(
                [d for d in CARDINALS if gameMap.getSite(loc,d).owner!=myID],
                key = lambda d:direct_attractiveness(gameMap.getSite(loc,d),myID))
            chosen_loc = gameMap.getLocation(loc,chosen_d)
            suggested_attractiveness[chosen_loc] = \
                direct_attractiveness(gameMap.getSite(chosen_loc,chosen_d),myID)
            suggested_attractiveness[loc] = suggested_attractiveness[chosen_loc]*(1-decay)

            process_move(loc,chosen_d,myID,gameMap,momentumMap,moves)

        time_tracker.track("Processing frontier moves")

        for y in range(gameMap.height):
            for x in range(gameMap.width):

                loc = Location(x,y)
                site = gameMap.getSite(loc)

                if site.owner == myID and loc not in frontier_set:

                    chosen_d = max(
                        [d for d in CARDINALS],
                        key = lambda d:suggested_attractiveness[gameMap.getLocation(loc,d)]
                        )
                    chosen_loc = gameMap.getLocation(loc,chosen_d)
                    if loc not in suggested_attractiveness:
                        suggested_attractiveness[loc] = 0
                    suggested_attractiveness[loc] += suggested_attractiveness[chosen_loc]*(1-decay)

                    process_move(loc,chosen_d,myID,gameMap,momentumMap,moves)

        time_tracker.track("Processing other moves")
        sendFrame(moves)
        time_tracker.log(logging)
        turn += 1
