import random
import math
import copy

STILL = 0
NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4

NORTHEAST = 1.5
SOUTHEAST = 2.5
SOUTHWEST = 3.5
NORTHWEST = 4.5

DIRECTIONS = [a for a in range(0, 5)]
CARDINALS = [a for a in range(1, 5)]
HALFCARDINALS = [x/2. for x in range(2,10)]

ATTACK = 0
STOP_ATTACK = 1

class Location:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def rectify(self,h,w):
        self.x %= h
        self.y %= h
        return self

    def __hash__(self):
        return hash((self.x,self.y))

    def __eq__(self,other):
        return self.x==other.x and self.y==other.y

    def __ne__(self,other):
        return not self==other

    def __str__(self):
        return "loc({},{})".format(self.x,self.y)

    def __repr__(self):
        return str(self)

class Site:
    def __init__(self, owner=0, strength=0, production=0):
        self.owner = owner
        self.strength = strength
        self.production = production
        self.dist_frontier = None
        self.is_frontier = None
        self.is_inner_frontier = None
        self.enemy_detected = False

    def __str__(self):
        return "site(owner={},strength={},prod={})".format(self.owner,self.strength,self.production)

    def __repr__(self):
        return str(self)

class Move:
    def __init__(self, loc=0, direction=0):
        self.loc = loc
        self.direction = direction

_locscache = {}

class GameMap:
    def __init__(self, width = 0, height = 0, numberOfPlayers = 0):
        self.width = width
        self.height = height
        self.contents = []
        self._locsmap = None

        for y in range(0, self.height):
            row = []
            for x in range(0, self.width):
                row.append(Site(0, 0, 0))
            self.contents.append(row)

    def inBounds(self, l):
        return l.x >= 0 and l.x < self.width and l.y >= 0 and l.y < self.height

    def getDistance(self, l1, l2):
        dx = abs(l1.x - l2.x)
        dy = abs(l1.y - l2.y)
        if dx > self.width / 2:
            dx = self.width - dx
        if dy > self.height / 2:
            dy = self.height - dy
        return dx + dy

    def getAngle(self, l1, l2):
        dx = l2.x - l1.x
        dy = l2.y - l1.y

        if dx > self.width - dx:
            dx -= self.width
        elif -dx > self.width + dx:
            dx += self.width

        if dy > self.height - dy:
            dy -= self.height
        elif -dy > self.height + dy:
            dy += self.height
        return math.atan2(dy, dx)

    def getLocation(self, loc, direction):
        if (loc,direction) in _locscache:
            return _locscache[(loc,direction)]
        l = copy.deepcopy(loc)
        if direction != STILL:
            if direction == NORTH:
                if l.y == 0:
                    l.y = self.height - 1
                else:
                    l.y -= 1
            elif direction == EAST:
                if l.x == self.width - 1:
                    l.x = 0
                else:
                    l.x += 1
            elif direction == SOUTH:
                if l.y == self.height - 1:
                    l.y = 0
                else:
                    l.y += 1
            elif direction == WEST:
                if l.x == 0:
                    l.x = self.width - 1
                else:
                    l.x -= 1
            elif direction == NORTHEAST:
                if l.x == self.width - 1:
                    l.x = 0
                else:
                    l.x += 1
                if l.y == 0:
                    l.y = self.height - 1
                else:
                    l.y -= 1
            elif direction == NORTHWEST:
                if l.x == 0:
                    l.x = self.width - 1
                else:
                    l.x -= 1
                if l.y == 0:
                    l.y = self.height - 1
                else:
                    l.y -= 1
            elif direction == SOUTHEAST:
                if l.x == self.width - 1:
                    l.x = 0
                else:
                    l.x += 1
                if l.y == self.height - 1:
                    l.y = 0
                else:
                    l.y += 1
            elif direction == SOUTHWEST:
                if l.x == 0:
                    l.x = self.width - 1
                else:
                    l.x -= 1
                if l.y == self.height - 1:
                    l.y = 0
                else:
                    l.y += 1
        _locscache[(loc,direction)] = l
        return l

    def getSite(self, l, direction = STILL):
        l = self.getLocation(l, direction)
        return self.contents[l.y][l.x]
