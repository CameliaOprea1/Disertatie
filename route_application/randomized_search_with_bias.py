import random
from collections import deque
import networkx as nx
import itertools

# Define main road types
main_road_types = ['trunk', 'trunk_link', 'primary', 'primary_link']

class Node:
    __slots__ = ['node', 'distance', 'parent', 'osmid', 'G']

    def __init__(self, graph, osmid, distance=0, parent=None):
        self.node = graph[osmid]
        self.distance = distance
        self.parent = parent
        self.osmid = osmid
        self.G = graph

    def expand(self, main_road_bias=1):
        children = [Node(graph=self.G, osmid=child, distance=self.node[child][0]['length'], parent=self) 
                    for child in self.node]
        
        # Categorize children into main roads and other roads
        main_road_children = [child for child in children if any(self.G[self.osmid][child.osmid][key]['highway'] in main_road_types for key in self.G[self.osmid][child.osmid])]
        other_children = [child for child in children if child not in main_road_children]
        biased_children = main_road_children + other_children
        return biased_children


    def path(self):
        node = self
        path = []
        while node:
            path.append(node.osmid)
            node = node.parent
        return path[::-1]

    def __eq__(self, other):
        try:
            return self.osmid == other.osmid
        except:
            return self.osmid == other

    def __hash__(self):
        return hash(self.osmid)

def randomized_search_biased(G, source, destination, main_road_bias=0.7):
    origin = Node(graph=G, osmid=source)
    destination = Node(graph=G, osmid=destination)
    
    route = []  # the route to be yielded
    frontier = deque([origin])
    explored = set()
    
    while frontier:
        node = random.choice(frontier)  # here is the randomization part
        frontier.remove(node)
        explored.add(node.osmid)

        for child in node.expand(main_road_bias=main_road_bias):
            if child.osmid not in explored and child not in frontier:
                if child == destination:
                    route = child.path()
                    return route
                frontier.append(child)

    raise Exception("destination and source are not on same component")

def cost(G, route):
    weight = 0
    for u, v in zip(route, route[1:]):
        weight += G[u][v][0]['length']   
    return round(weight, 4)

def cost_tour(G, route):
    weight = 0
    route = list(route)
    for u, v in zip(route, route[1:] + [route[0]]):
        weight += G[u][v]['weight']
    return weight

def one_way_route(G, route):
    def isvalid(G, route):
        for u, v in zip(route, route[1:]):
            try:
                G[u][v]
            except:
                return False
        return True

    while True:
        if isvalid(G, route): break
        i = 0
        j = 1
        found = False
        while not found and j < len(route) - 1:          
            try:
                u, v = route[i], route[j]
                G[u][v]
                i += 1
                j += 1
            except:
                node_before = route[i]
                node_failing = route[i:j+1]
                node_after = route[j+1]
                output = shortest_path_with_failed_nodes(G, route, node_before, node_after, node_failing)
                while type(output) is not list and i > 1 and j < len(route) - 1:
                    i -= 1
                    j += 1
                    node_before = route[i]
                    node_failing = route[i:j+1]
                    node_after = route[j+1]
                    output = shortest_path_with_failed_nodes(G, route, node_before, node_after, node_failing)
                route[i:j+2] = output
                found = True
                i += 1
                j += 1
    return route

def random_tour(iterable, r=None, number_of_perms=50):
    for i in range(number_of_perms):
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        yield list(random.sample(pool, r))

def probability(p):
    return p > random.uniform(0.0, 1.0)

def flatten(list2d):
    return list(itertools.chain(*list2d))


def shortest_path_with_failed_nodes(G, route ,i,j, failed : list):
    source = route[i-1]
    target = route[j+1]
    origin = Node(graph = G, osmid = source)