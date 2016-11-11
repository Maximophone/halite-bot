from hlt import *
from networking import *
import logging
import time

logging.basicConfig(filename='last_run.log',level=logging.DEBUG)
logging.debug('Hello')

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

myID, gameMap = getInit()

prodMap = []
blurredProdMap = []
for y in range(gameMap.height):
    prodMapRow = []
    blurredProdMapRow = []
    for x in range(gameMap.width):
        site = gameMap.getSite(Location(x, y))
        blurredProd = site.production
        for d in CARDINALS:
            neighbour_site = gameMap.getSite(Location(x, y),d)
            blurredProd += neighbour_site.production
        prodMapRow.append(site.production)
        blurredProdMapRow.append(blurredProd/5.)
    prodMap.append(prodMapRow)
    blurredProdMap.append(blurredProdMapRow)

sendInit("MaximoBot_v0.1")

turn = 0
while True:
    dtleaving = 0
    moves = []
    gameMap = getFrame()
    t0 = time.time()
    centroid = find_centroid(myID,gameMap)
    t1 = time.time()
    logging.debug("TURN: {}".format(turn))
    logging.debug("CENTROID: {},{}".format(centroid.x,centroid.y))
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            site = gameMap.getSite(Location(x, y))
            if site.owner == myID:
                moved = False
                # logging.debug("Turn {}".format(turn))
                # If we are surrounded by our territory,
                tleaving0 = time.time()
                if all(gameMap.getSite(Location(x, y),d).owner==myID for d in CARDINALS) and site.strength > site.production*5:
                    #Go towards the closest frontier
                    # d = leave_center(Location(x,y),centroid,gameMap)
                    # moves.append(Move(Location(x, y), d))
                    moves.append(Move(Location(x, y), NORTH if bool(int(random.random() * 2)) else WEST))
                    moved = True
                tleaving1 = time.time()
                dtleaving += (tleaving1-tleaving0)
                # logging.debug("Checked frontier")
                for d in CARDINALS:
                    neighbour_site = gameMap.getSite(Location(x, y),d)
                    if neighbour_site.owner != myID and neighbour_site.strength < site.strength:
                        moves.append(Move(Location(x, y), d))
                        moved = True
                        break
                # if not moved and site.strength < site.production*5:
                #     moves.append(Move(Location(x, y), STILL))
                #     moved = True
                if not moved:
                    # moves.append(Move(Location(x, y), NORTH if bool(int(random.random() * 2)) else WEST))
                    moves.append(Move(Location(x, y), STILL))
                    moved = True
                # if not moved and site.strength<=15:
                #     moves.append(Move(Location(x, y), STILL))
                # elif not moved:
                #     moves.append(Move(Location(x, y), NORTH if bool(int(random.random() * 2)) else WEST))
    t2 = time.time()
    sendFrame(moves)
    logging.debug("dt1={:.5f} dt2={:.5f} dtleaving={:.5f}".format(t1-t0,t2-t1,dtleaving))
    turn += 1
