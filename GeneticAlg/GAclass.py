import osmnx as ox
import networkx as nx
import geopandas as gpd
from collections import deque
from deap import base, creator, tools, algorithms
ox.settings.log_console=True
ox.settings.use_cache=True
import random
from numpy.random import choice
from smart_mobility_utilities.common import randomized_search
import math



class GeneticAlgorithm:
    def __init__(self, graph, successors_dict, source, destination, population_size=50,crossover_rate=0.8):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.population_size = population_size
        #self.vicinity_matrix = vicinity_matrix.todense()
        self.successors_dict = successors_dict
        self.crossover_rate = crossover_rate
        self.distance_cache = {}
        # Initialize parameters

    def find_common_elements(self, list1, list2):
        # Exclude source and destination from the lists
        list1_without_ends = [node for node in list1 if node != self.source and node != self.destination]
        list2_without_ends = [node for node in list2 if node != self.source and node != self.destination]
        
        # Find common elements
        return set(list1_without_ends) & set(list2_without_ends)

    def generate_chromosome(self, source=None, destination=None):
        if source is None:
            source = self.source
        if destination is None:
            destination = self.destination
    
        chromosome = deque([source])  # Use a deque for chromosome
        visited = {source: None}  # Dictionary to store visited nodes and their parent nodes
        route = []
    
        while chromosome:
            random.shuffle(chromosome)  # Shuffle the deque to introduce randomness
            current_node = chromosome.popleft()  # Choose the first node after shuffling
            if current_node == destination:
                # Reconstruct route from destination to source
                while current_node is not None:
                    route.append(current_node)
                    current_node = visited[current_node]  # Get parent node
                route.reverse()  # Reverse to get the correct order (from source to destination)
                return route
            visited[current_node] = None  # Mark current node as visited
    
            for neighbor in self.new_gene(current_node):
                if neighbor not in visited:
                    chromosome.append(neighbor)
                    visited[neighbor] = current_node  # Assign current node as parent for the neighbor
        return route

    def new_gene(self, current_node):
        #available_nodes = [neighbor for neighbor in self.successors_dict[current_node] if neighbor not in chromosome]
        #if available_nodes:
        #    return random.choice(available_nodes)
        #else:
        available_nodes = [neighbor for neighbor in self.successors_dict[current_node] ]
        return available_nodes
        
    def crossover(self, parent1, parent2):
        #if random.random() > self.crossover_rate:
        #    print('NO CROSSOVER')
        #    return parent1, parent2  # No crossover
        
        common_elements = list(self.find_common_elements(parent1, parent2))

        if not common_elements:
            # If no common elements found, return parents
            #print('NO COMMON NODES')
            return parent1, parent2

        crossover_point_index = random.choice(common_elements)  # Ensure crossover point does not include source or destination

        #print(crossover_point_index)
        

        # Swap segments between parents to create children
        child1 = parent1[:parent1.index(crossover_point_index)] + parent2[parent2.index(crossover_point_index):]
        child2 = parent2[:parent2.index(crossover_point_index)] + parent1[parent1.index(crossover_point_index):]

        # Debug prints to check segment swapping
        #print('Parent1:', parent1)
        #print('Parent2:', parent2)
        #print('Child 1:', child1)
        #print('Child 2:', child2)

        #edge cases:
        #cand rutele coincid pana intr-un punct nu are sens sa luam un nod care are aceeasi vecini in ambiik parinti==> facem swap de parinti, deci nu e crossover
        #cand ruta rezultata contine de mai multe ori acelasi nod, trebuie rectificata
        return child1, child2
    
    def mutation(self, individual):
        if len(individual) < 3:
            #print('Individual is too short to perform mutation')
            return individual
        mutation_point_index = random.randint(1, len(individual) - 2)
        #print('Mutation Index:', mutation_point_index)

        # Build a new valid path from the mutation point
        new_chromosome = randomized_search(self.graph,individual[mutation_point_index],self.destination)
        individual[mutation_point_index:] = new_chromosome

        return individual
    
    def fitnessEvaluation(self, individual):
        total_distance = self.calculate_total_distance(individual)
        return (total_distance,)  # Comma is necessary for DEAP compatibility
    '''

    def calculate_total_distance(self, individual):
        total_distance = 0
        for i in range(len(individual) - 1):
            current_node = individual[i]
            next_node = individual[i + 1]
            edge_distance = self.calculate_distance(current_node, next_node)
            total_distance += edge_distance
        return total_distance

    def calculate_distance(self, node1, node2):
        # Calculate distance between two nodes using OSMnx
        try:
            #route = nx.shortest_path(self.graph, node1, node2, weight='length')
            #distance = sum(ox.utils_graph.get_route_edge_attributes(self.graph, route, 'length'))
            #mai optim:
            distance = nx.shortest_path_length(self.graph, node1, node2, weight='length')
            return distance
        except nx.NetworkXNoPath:
            # Handle case when there's no path between the nodes
            return math.inf
    '''
    def calculate_total_distance(self, individual):
        total_distance = 0
        for i in range(len(individual) - 1):
            current_node = individual[i]
            next_node = individual[i + 1]
            edge_distance = self.get_cached_distance(current_node, next_node)
            if edge_distance is None:
                edge_distance = self.calculate_distance(current_node, next_node)
                self.cache_distance(current_node, next_node, edge_distance)
            total_distance += edge_distance
        return total_distance

    def get_cached_distance(self, node1, node2):
        
        key = (node1, node2)
        return self.distance_cache.get(key)

    def cache_distance(self, node1, node2, distance):
        key = (node1, node2)
        self.distance_cache[key] = distance

    def calculate_distance(self, node1, node2):
        try:
            distance = nx.shortest_path_length(self.graph, node1, node2, weight='length')
            return distance
        except nx.NetworkXNoPath:
            return math.inf




