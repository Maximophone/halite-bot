from hlt import *
from networking import *

myID, gameMap = getInit()
sendInit("BasicBot")

turn = 0
while True:
    moves = []
    gameMap = getFrame()
    for y in range(gameMap.height):
        for x in range(gameMap.width):
            site = gameMap.getSite(Location(x, y))
            if site.owner == myID:
                moved = False
                for d in CARDINALS:
                    neighbour_site = gameMap.getSite(Location(x, y),d)
                    # attractivity = neighbour_site.production/float(neighbour_site.strength)
                    if neighbour_site.owner != myID and neighbour_site.strength < site.strength:
                        moves.append(Move(Location(x, y), d))
                        moved = True
                        break
                if not moved and site.strength < site.production*5:
                    moves.append(Move(Location(x, y), STILL))
                    moved = True
                if not moved:
                    moves.append(Move(Location(x, y), NORTH if bool(int(random.random() * 2)) else WEST))
                    moved = True
                # if not moved and site.strength<=15:
                #     moves.append(Move(Location(x, y), STILL))
                # elif not moved:
                #     moves.append(Move(Location(x, y), NORTH if bool(int(random.random() * 2)) else WEST))
    sendFrame(moves)
    turn += 1
