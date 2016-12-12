from hlt2 import *
import utils2
import logging
import random
import math

random.seed(0)
logging.basicConfig(filename='routingbot2.log',level=logging.DEBUG)

alpha = 0.5
def attractiveness(square,my_id):
    if square.owner == my_id or square.strength>200:
        return 0.
    return ((255.-square.strength)/255 + square.production/30.)**2

def attractiveness_start(square,my_id):
    return square.production/float(square.strength)

def cost(square):
    return square.strength/float(square.production+1)

def get_min_distance_enemy(my_id,gamemap,n_samples = 200, prev_min_dist = 1000):
    my_set = []
    enemy_set = []
    for square in gamemap:
        if square.owner == my_id:
            my_set.append(square)
        elif square.owner != 0:
            enemy_set.append(square)
    min_dist = 1000
    for _ in range(n_samples):
        sq0 = random.choice(my_set)
        sq1 = random.choice(enemy_set)
        min_dist = min(min_dist,gamemap.get_distance(sq0,sq1))
    return min(prev_min_dist,min_dist)

def get_smart_decay_OLD(my_id,stats):
    return 1.
    my_territory = stats['n_owners'][my_id]
    decay = 1/math.sqrt(my_territory)
    return decay

def get_stats_strength(gamemap):
    mean_strength = sum([square.strength for square in gamemap])/float(gamemap.width*gamemap.height)
    var_strength = sum([(square.strength - mean_strength)**2 for square in gamemap])/float(gamemap.width*gamemap.height)
    return mean_strength, var_strength

def get_initial_decay(gamemap,max_decay=0.1,min_decay=0.05):
    pass

def get_decay_factor(my_id,stats,gamemap,min_dist):
    # if min_dist<=4:
    #     return 1.
    min_factor = 0.1
    max_factor = 1.
    diff = max_factor - min_factor
    return diff/(1+math.exp((min_dist-16)/4))+min_factor

def get_smart_enemy_attr(my_id,enemy_id,stats):
    enemy_strength = stats['strength_owners'][enemy_id]
    my_strength = stats['strength_owners'][my_id]
    return (enemy_strength/(my_strength+1.))

def adjust_frontier_potential(frontier,my_id,attr_map,gamemap,stats,enemy_attr=1.,exploration_factor=1.,radius=5,enemy_attr_far=2.):
    adjusted_attr = {k:v for k,v in attr_map.items()}
    for square in frontier:
        sum_attr = 0
        region = [neighbor for neighbor in gamemap.neighbors(square,n=radius) if neighbor.owner != my_id]
        len_region = 0.
        for neighbor in region:
            if neighbor in frontier:
                continue
            len_region += 1.
            attr = attr_map[neighbor]
            if neighbor.owner not in (0,my_id):
                attr += enemy_attr_far*get_smart_enemy_attr(my_id,neighbor.owner,stats) - 0.0
            sum_attr += attr
        average_attr = sum_attr/len_region if len_region else 0
        adjusted_attr[square] += exploration_factor*average_attr
    for square in frontier:
        for neighbor in gamemap.neighbors(square):
            if neighbor.owner not in (0,my_id):
                adjusted_attr[neighbor] += enemy_attr*get_smart_enemy_attr(my_id,neighbor.owner,stats)
    return adjusted_attr

def adjust_frontier_potential_VAR(frontier,my_id,attr_map,gamemap,stats,enemy_attr=1.,radius=5,enemy_attr_far=2.,dist_decay=.9):
    adjusted_attr = {k:v for k,v in attr_map.items()}
    for square in frontier:
        sum_attr = 0
        region = [neighbor for neighbor in gamemap.neighbors(square,n=radius, include_self=True) if neighbor.owner != my_id]
        sum_weights = 0.
        for neighbor in region:
            if neighbor in frontier:
                continue
            attr = attr_map[neighbor]
            if neighbor.owner not in (0,my_id):
                attr += enemy_attr_far*get_smart_enemy_attr(my_id,neighbor.owner,stats) - 0.0
            dist = gamemap.get_distance(neighbor,square)
            weight = dist_decay**dist
            sum_weights += weight
            sum_attr += weight*attr
        average_attr = sum_attr/sum_weights if sum_weights else 0
        adjusted_attr[square] += average_attr
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
    # potentials = {square:None for square in inner}
    # for square in frontier:
    #     potentials[square] = attr_map[square]
    while len(sorted_nodes)>0:
        current,value = sorted_nodes.pop()
        for neighbor in gamemap.neighbors(current):
            if not neighbor in inner_set or neighbor in visited:
                continue
            # potentials[neighbor] = potentials[current] - decay
            new_attr_map[neighbor] = new_attr_map[current] - decay
            sorted_nodes.append((neighbor,new_attr_map[neighbor]))
            sorted_nodes.sort(key=lambda x:x[1])
            visited.add(neighbor)
    # for square in inner:
    #     new_attr_map[square] = potentials[square]
    return new_attr_map

def shouldMove_OLD(square,new_square,map_uptodate,my_id):
    if square.strength <= 3*square.production: 
    #if strength <= 5*production
        return False
    if map_uptodate[new_square].strength > square.strength and map_uptodate[new_square].owner != my_id: 
    #if foreign tile of superior strength
        return False
    if map_uptodate[new_square].strength + square.strength > 255 and map_uptodate[new_square].owner == my_id:
        logging.debug('prevented move')
    #if friendly tile and sum>255
        return False
    return True

def map_directions_OLD(my_id,attr_map,gamemap,momentum_map,directions_dict=None):
    directions_map = {}
    wanted_directions_map = {}
    moves_list = []
    map_uptodate = {square:Square(
        square.x, 
        square.y, 
        square.owner, 
        square.strength if square.owner!=my_id else 0., 
        square.production) for square in gamemap}
    sorted_squares = sorted(gamemap,key=lambda sq:sq.strength, reverse=True)
    for square in sorted_squares:
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
            logging.debug(new_strength)
            logging.debug(map_uptodate[new_square])
            #updating mapping
            map_uptodate[new_square] = map_uptodate[new_square]._replace(strength=new_strength, owner=my_id)
            # map_uptodate[new_square]._replace(owner = my_id)
            logging.debug(map_uptodate[new_square])
            logging.debug('')
            # map_uptodate[square]._replace(strength = 0)
        else:
            directions_map[square] = STILL
            map_uptodate[square] = map_uptodate[square]._replace(strength=square.strength)

        moves_list.append((square,directions_map[square]))
    return moves_list,directions_map,wanted_directions_map

def should_move(square,new_square,my_id):
    if square.strength <= 3*square.production: 
    #if strength <= 5*production
        return False
    if new_square.strength > square.strength and new_square.owner != my_id: 
    #if foreign tile of superior strength
        return False
    return True

def resolve_directions(directions_map,my_id,gamemap):
    arrivals_map = {square:[] for square in gamemap}
    for sq,d in directions_map.items():
        if d is None:
            continue
        new_sq = gamemap.get_target(sq,d)
        arrivals_map[new_sq].append(sq)
    conflicts = True
    # j = 0
    while conflicts:
        conflicts = False
        for new_square,origins in arrivals_map.items():
            if len(origins)>1:
                #There is a potential conflict
                if sum([square.strength for square in origins])>300:
                    # j+=1
                    #conflict
                    # logging.debug("CONFLICT")
                    # logging.debug(new_square)
                    conflicts = True
                    origins.sort(key=lambda x:x.strength)
                    # logging.debug(origins)
                    total = sum([square.strength for square in origins])
                    # logging.debug(total)
                    # logging.debug("entering loop")
                    # i = 0
                    while total>300:
                        # i+=1
                        # if i>100:
                            # break
                        square = origins.pop(0)
                        # logging.debug(square)
                        if directions_map[square] == STILL:
                            # logging.debug('still...')
                            origins.append(square)
                            # logging.debug(origins)
                            # logging.debug('continuing')
                            continue
                        arrivals_map[square].append(square)
                        total -= square.strength
                        directions_map[square] = STILL
                        # logging.debug(total)
        # if j>1000:
            # logging.debug("breaking outer loop")
            # break
    return directions_map

def map_directions(my_id,attr_map,gamemap,momentum_map,directions_dict=None):
    directions_map = {}
    wanted_directions_map = {}
    moves_list = []
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
        if should_move(square,new_square,my_id):
            directions_map[square] = potential_direction
        else:
            directions_map[square] = STILL

    directions_map = resolve_directions(directions_map,my_id,gamemap)
    return directions_map,wanted_directions_map

def process_movelist(movelist,moves,momentum_map,gamemap):
    for square,d in movelist:
        moves.append(Move(square, d))
        if d == STILL:
            momentum_map[(square.x,square.y)] = STILL
        else:
            new_square = gamemap.get_target(square,d)
            momentum_map[(new_square.x,new_square.y)] = d

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
    mean_strength,var_strength = get_stats_strength(gamemap)
    logging.debug("Computed map stats")
    logging.debug("Mean strength: {:.0f}".format(mean_strength))
    logging.debug("STD strength: {:.2f}".format(math.sqrt(var_strength)))

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

    send_init("RoutingBot2")

    logging.debug("Init sent")
    
    start_decay = 0.1
    enemy_attr = 0.1 #0.5 works well too
    enemy_attr_far = 0.01
    radius = 6

    turn = 0
    time_tracker = utils2.TimeTracker(logging)
    game_dumper = utils2.Dumper('gameMap','refactored',on=False)

    while True:

        moves = []
        gamemap.get_frame()
        logging.debug("TURN: {}".format(turn))

        time_tracker.track()

        gamemap_stats = utils2.get_gamemap_stats(my_id,gamemap)
        time_tracker.track("Computing map stats")

        min_distance_to_others = get_min_distance_enemy(my_id,gamemap,prev_min_dist=min_distance_to_others)
        logging.debug("Min Distance: {}".format(min_distance_to_others))

        decay = get_decay_factor(my_id,gamemap_stats,gamemap,min_distance_to_others)*start_decay
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

        directions_map,wanted_directions_map = map_directions(
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








