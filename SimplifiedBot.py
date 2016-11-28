from hlt2 import *
import utils2
import logging
import random
import math

random.seed(0)
logging.basicConfig(filename='last_run.log',level=logging.DEBUG)

def attractiveness(square,my_id):
    if square.owner == my_id:
        return 0.
    if square.owner != 0:
        return square.production
    return ((255.-square.strength)/255 + square.production/30.)**2

def attractiveness_start(square,my_id):
    return square.production/float(square.strength)

def cost(square):
    return square.strength/float(square.production+1)

def get_smart_decay(my_id,stats):
    return 1.
    my_territory = stats['n_owners'][my_id]
    decay = 1/math.sqrt(my_territory)
    return decay

def get_smart_enemy_attr(my_id,enemy_id,stats):
    enemy_strength = stats['strength_owners'][enemy_id]
    my_strength = stats['strength_owners'][my_id]
    return (enemy_strength/(my_strength+1.))

def adjust_frontier_potential(frontier,my_id,attr_map,gamemap,stats,enemy_attr=1.,exploration_factor=1.,radius=5,enemy_attr_far=2.):
    adjusted_attr = {k:v for k,v in attr_map.items()}
    for square in frontier:
        average_attr = 0.
        region = [neighbor for neighbor in gamemap.neighbors(square,n=radius) if neighbor.owner != my_id]
        for neighbor in region:
            average_attr += attr_map[neighbor]/float(len(region))
        adjusted_attr[square] = average_attr
    for square in frontier:
        for neighbor in gamemap.neighbors(square):
            if neighbor.owner not in (0,my_id):
                adjusted_attr[neighbor] += enemy_attr*get_smart_enemy_attr(my_id,neighbor.owner,stats)
    return adjusted_attr


def map_potential(inner,frontier,attr_map,gamemap,decay):
    new_attr_map = {k:v for k,v in attr_map.items()}
    inner_set = set(inner)
    frontier_set = set(frontier)
    total_set = inner_set.union(frontier_set)
    visited = set()
    sorted_nodes = [(f,new_attr_map[f]) for f in frontier]
    sorted_nodes.sort(key=lambda x:x[1])
    while len(sorted_nodes)>0:
        current,value = sorted_nodes.pop()
        for neighbor in gamemap.neighbors(current):
            if not neighbor in inner_set or neighbor in visited:
                continue
            new_attr_map[neighbor] = new_attr_map[current] - decay
            sorted_nodes.append((neighbor,new_attr_map[neighbor]))
            sorted_nodes.sort(key=lambda x:x[1])
            visited.add(neighbor)
    return new_attr_map

def shouldMove(square,new_square,map_uptodate,my_id):
    if square.strength <= 3*square.production: 
    #if strength <= 5*production
        return False
    if map_uptodate[new_square].strength > square.strength and map_uptodate[new_square].owner != my_id: 
    #if foreign tile of superior strength
        return False
    if map_uptodate[new_square].strength + square.strength > 255 and map_uptodate[new_square].owner == my_id:
    #if friendly tile and sum>255
        return False
    return True

def map_directions(my_id,attr_map,gamemap,momentum_map,directions_dict=None):
    directions_map = {}
    wanted_directions_map = {}
    moves_list = []
    map_uptodate = {square:Square(
        square.x, 
        square.y, 
        square.owner, 
        square.strength if square.owner!=my_id else 0., 
        square.production) for square in gamemap}
    for square in gamemap:
        if square.owner!=my_id:
            directions_map[square] = None
            wanted_directions_map[square] = None
            continue
        if directions_dict is None or directions_dict.get(square) is None:
            candidate_directions = [(d,n) for d,n in enumerate(gamemap.neighbors(square)) if d != opp_cardinal[momentum_map[(square.x,square.y)]]]
            # candidate_directions = [(d,n) for d,n in enumerate(gamemap.neighbors(square))]
            potential_direction,new_square = max(candidate_directions,key=lambda dsq:attr_map[dsq[1]])
        else:
            potential_direction = directions_dict.get(square,STILL)
            new_square = gamemap.get_target(square,potential_direction)
        wanted_directions_map[square] = potential_direction
        if shouldMove(square,new_square,map_uptodate,my_id):
            directions_map[square] = potential_direction
            new_strength = square.strength + map_uptodate[new_square].strength if map_uptodate[new_square].owner==my_id else square.strength-map_uptodate[new_square].strength
            #updating mapping
            map_uptodate[new_square]._replace(strength=new_strength)
            map_uptodate[new_square]._replace(owner = my_id)
            map_uptodate[square]._replace(strength = 0)
        else:
            directions_map[square] = STILL

        moves_list.append((square,directions_map[square]))
    return moves_list,directions_map,wanted_directions_map

def process_directions(moves,directions_map,momentum_map,gamemap):
    sorted_squares = sorted(gamemap,key=lambda sq:sq.strength)
    for square in sorted_squares:
        if directions_map[square] is None:
            continue
        d = directions_map[square]
        moves.append(Move(square,d))
        if d == STILL:
            momentum_map[(square.x,square.y)] = STILL
        else:
            new_square = gamemap.get_target(square,d)
            momentum_map[(new_square.x,new_square.y)] = d


if __name__ == "__main__":

    target_reached = False
    momentum_map = {}

    my_id,gamemap = get_init()

    gamemap_stats = utils2.get_gamemap_stats(my_id,gamemap)
    logging.debug("Computed map stats")

    frontier = utils2.find_frontier(my_id,gamemap)
    logging.debug("Found frontier")

    start = utils2.find_start(my_id,gamemap)
    momentum_map[(start.x,start.y)] = STILL
    others_start = [utils2.find_start(player_id,gamemap) for player_id in range(gamemap.starting_player_count+1) if player_id not in (0,my_id)]
    logging.debug("Found Starting Positions")
    logging.debug(start)
    logging.debug(others_start)

    min_distance_to_others = utils2.get_min_distance(start,others_start,gamemap)
    logging.debug("Found Minimum Distance")
    logging.debug(min_distance_to_others)

    attr_map = utils2.map_attractiveness(my_id,gamemap,attractiveness_start)
    logging.debug("Mapped Attractiveness")

    smoothed_attr_map = utils2.smooth_map(attr_map,gamemap,kernel=[1.,1.5])
    logging.debug("Mapped Smoothed Attractiveness")

    target = utils2.find_local_max(start,int(min_distance_to_others/2),smoothed_attr_map,gamemap)
    logging.debug("Found Target")
    logging.debug(target)

    directions_dict,path = utils2.a_star(target,start,gamemap,cost)
    logging.debug("Found Path")

    send_init("SimplifiedBot")

    logging.debug("Init sent")
    
    decay_factor = 0.05
    momentumTerm = 1000.
    enemy_attr = 0.2 #0.5 works well too
    enemy_attr_far = 0.01
    radius = 6

    turn = 0
    time_tracker = utils2.TimeTracker(logging)
    game_dumper = utils2.Dumper('gameMap','simplified',on=True)

    while True:

        moves = []
        gamemap.get_frame()
        logging.debug("TURN: {}".format(turn))

        time_tracker.track()

        gamemap_stats = utils2.get_gamemap_stats(my_id,gamemap)
        time_tracker.track("Computing map stats")

        decay = get_smart_decay(my_id,gamemap_stats)*decay_factor
        logging.debug("Decay: {:.2f}".format(decay))

        frontier,frontier_map = utils2.track_frontier(frontier,my_id,gamemap)
        time_tracker.track("Tracking frontier")

        inner = utils2.find_inner(my_id,gamemap)
        time_tracker.track("Finding inner")

        attr_map = utils2.map_attractiveness(my_id,gamemap,attractiveness)
        time_tracker.track("Mapping Attractiveness")

        adjusted_attr_map = adjust_frontier_potential(
            frontier,
            my_id,
            attr_map,
            gamemap,
            gamemap_stats,
            enemy_attr=enemy_attr,
            enemy_attr_far=enemy_attr_far,
            radius=radius)
        time_tracker.track("Adjusting Frontier Potential Attr")

        final_attr_map = map_potential(inner,frontier,adjusted_attr_map,gamemap,decay=decay)
        time_tracker.track("Map Potential Attr")

        movelist,directions_map,wanted_directions_map = map_directions(
            my_id,
            final_attr_map,
            gamemap,
            momentum_map,
            directions_dict if not target_reached else None)
        time_tracker.track("Map Directions")

        #process_movelist(movelist,moves,momentum_map,gamemap)
        process_directions(moves,directions_map,momentum_map,gamemap)

        game_state_dict = {
            'my_id':my_id,
            'gamemap':gamemap,
            'attr_map':attr_map,
            'adjusted_attr_map':adjusted_attr_map,
            'final_attr_map':final_attr_map,
            'directions_map':directions_map,
            'wanted_directions_map':wanted_directions_map,
            'momentum_map':momentum_map,
            'smoothed_attr_map':smoothed_attr_map,
            'inner':inner,
            'frontier':frontier,
            'frontier_map':frontier_map,
            'target_reached':target_reached,
            'directions_dict':directions_dict,
            'target':target,
            'start':start
        }

        game_dumper.dump(game_state_dict,turn)

        if gamemap.get_target(target,4).owner!=0:
            target_reached = True

        send_frame(moves)
        time_tracker.log()
        turn += 1








