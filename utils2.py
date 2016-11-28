import time
from hlt2 import *

class Dumper(object):

    def __init__(self,name,botname,on=False):
        self.folder = "dumps"
        self.name = name
        self.botname = botname
        self.on = on

    def dump(self,obj,turn):
        if self.on:
            import _pickle
            filename = '{}/{}_{}_{}'.format(self.folder,self.name,self.botname,turn)
            with open(filename,'wb') as f:
                _pickle.dump(obj,f)

class TimeTracker(object):
    def __init__(self,logging):
        self.last_time = None
        self.dts = []
        self.names = []
        self.i=0
        self.logging = logging

    def track(self,name=None):
        t = time.time()
        if self.last_time is None:
            self.last_time = t
        else:
            self.dts.append(t-self.last_time)
            self.logging.debug("dt{}({})={:.5f}".format(self.i,name,t-self.last_time))
            self.last_time = t
            self.names.append(name)
        self.i+=1

    def log(self,reset=True):
        self.logging.debug("Total={:.5f}".format(sum(self.dts)))
        if reset:
            self.last_time = None
            self.dts = []
            self.names = []
            self.i = 0

def get_gamemap_stats(my_id,gamemap):
    stats = {
        'n_owners':[0]*(gamemap.starting_player_count+1),
        'strength_owners':[0]*(gamemap.starting_player_count+1),
        'production_owners':[0]*(gamemap.starting_player_count+1),
    }
    for square in gamemap:
        stats['n_owners'][square.owner] += 1
        stats['strength_owners'][square.owner] += square.strength
        stats['production_owners'][square.owner] += square.production 
    return stats

def find_frontier(my_id,gamemap):
    frontier = []
    for square in gamemap:
        if square.owner == my_id:
            continue
        for neighbor in gamemap.neighbors(square):
            if neighbor.owner != my_id:
                continue
            else:
                frontier.append(square)
                break
    return frontier

def find_start(my_id,gamemap):
    for square in gamemap:
        if square.owner == my_id:
            return square

def get_min_distance(square,squares,gamemap):
    return min([gamemap.get_distance(s,square) for s in squares])

def map_attractiveness(my_id,gamemap,attr):
    attrmap = {}
    for square in gamemap:
        attrmap[square] = attr(square,my_id)
    min_attr, max_attr = min(attrmap.values()), max(attrmap.values())
    for square in gamemap:
        attrmap[square] = (attrmap[square]-min_attr)/float((max_attr - min_attr)) if max_attr - min_attr else 0.
    return attrmap

def smooth_map(map_to_smooth,gamemap,kernel):
    smoothed_map = {}
    kernel_sum = float(sum(kernel))
    for square in gamemap:
        smoothed_map[square] = 0.
        for neighbor in gamemap.neighbors(square,n=len(kernel)-1):
            distance = gamemap.get_distance(square,neighbor)
            smoothed_map[square] += kernel[distance]*map_to_smooth[neighbor]/kernel_sum
    return smoothed_map

def find_local_max(center,radius,map,gamemap):
    squares = gamemap.neighbors(center,n=radius)
    return max(squares,key=lambda sq:map[sq])

def reconstruct_path(came_from,current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path

def a_star(start,end,gamemap,cost):
    closed_set = set()
    open_set = set([start])
    came_from = {}
    directions_dict = {}
    g_score = {}
    g_score[start] = 0
    f_score = {}
    f_score[start] = gamemap.get_distance(start,end)

    while open_set:
        current = min(open_set,key=lambda x:f_score[x])

        if current == end:
            return {k:opp_cardinal[v] for k,v in directions_dict.items()}, reconstruct_path(came_from,current)
        
        open_set.remove(current)
        closed_set.add(current)

        for d,neighbor in enumerate(gamemap.neighbors(current)):
            if neighbor in closed_set:
                continue
            tentative_g_score = g_score[current] + cost(neighbor)

            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score[neighbor]:
                continue

            directions_dict[neighbor] = d
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + gamemap.get_distance(neighbor,end)

    return False

def is_frontier(square,player_id,gamemap):
    if square.owner == player_id:
        return False
    for neighbor in gamemap.neighbors(square):
        if neighbor.owner != player_id:
            continue
        else:
            return True
    return False

def track_frontier(prev_frontier,my_id,gamemap):
    checked = set()
    frontier_map = {square:False for square in gamemap}
    new_frontier = set()
    for square in prev_frontier:
        for neighbor in gamemap.neighbors(square,include_self=True):
            if neighbor not in checked:
                checked.add(neighbor)
                if is_frontier(neighbor,my_id,gamemap):
                    new_frontier.add(neighbor)
                    frontier_map[neighbor] = True
    return new_frontier,frontier_map

def find_inner(my_id,gamemap):
    return set(square for square in gamemap if square.owner == my_id)
