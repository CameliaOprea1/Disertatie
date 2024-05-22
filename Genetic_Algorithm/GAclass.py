import osmnx as ox
from collections import deque

ox.settings.log_console=True
ox.settings.use_cache=True
import random
from smart_mobility_utilities.common import randomized_search # type: ignore
import pickle
import sys
import os

# Add the relevant directories to the system path
script_dir = os.path.dirname(__file__)
google_apis_path = os.path.join(script_dir, '..', 'Google_APIS_Request')
datasets_admin_path = os.path.join(script_dir, '..', 'Datasets_Administration_Creation')
osmnx_graph_path = os.path.join(script_dir, '..', 'Osmnx_graph_maniupulation')

sys.path.append(google_apis_path)
sys.path.append(datasets_admin_path)
sys.path.append(osmnx_graph_path)

from get_processed_api_data import predict_traffic_congestion
from datasets_creation import load_my_graph

'''
    In this script I define all that is needed for the genetic algorithm, i.e. genetic operators and define the fitness function
'''

##############################################################################################################################################
G, G_projected =  load_my_graph()
nodes, edges = ox.graph_to_gdfs(G)
nodes_cartesian, edge_cartesian = ox.graph_to_gdfs(G_projected)
nodes_geographical_df = nodes.to_crs(epsg=4326)  # Assuming EPSG 4326 is WGS84

model_pre_trained = pickle.load(open('C:\\Users\\Camelia\\Desktop\\refactored_App\\Machine_learning_modelling\\model_XGB_1.pkl','rb'))
url_openWeather = f"https://api.openweathermap.org/data/2.5/weather?lat={45.64861}&lon={25.60613}&appid=3f116979dae9f6daf340d900c87de197&units=metric"
##################################################################################################################################################
class GeneticAlgorithm:
   

    def __init__(self, graph, successors_dict, source, destination, population_size=50,crossover_rate=0.8,X_test={}):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.population_size = population_size
        #self.vicinity_matrix = vicinity_matrix.todense()
        self.successors_dict = successors_dict
        self.crossover_rate = crossover_rate
        self.distance_cache = {}
        self.distance_cache_for_individuals = {}
        self.X_test = X_test
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
            print('NO COMMON NODES')
            return parent1, parent2

        crossover_point_index = random.choice(common_elements)  # Ensure crossover point does not include source or destination

        #print(crossover_point_index)
        #cand rutele coincid pana intr-un punct nu are sens sa luam un nod care are aceeasi vecini in ambiik parinti==> facem swap de parinti, deci nu e crossover
        if (parent1[:parent1.index(crossover_point_index)] == parent2[parent2.index(crossover_point_index):]) and (parent2[:parent2.index(crossover_point_index)] == parent1[:parent1.index(crossover_point_index)]):
            print('Swapping produces same offsprings')
        else:
            # Swap segments between parents to create children
            child1 = parent1[:parent1.index(crossover_point_index)] + parent2[parent2.index(crossover_point_index):]
            child2 = parent2[:parent2.index(crossover_point_index)] + parent1[parent1.index(crossover_point_index):]
        child1, child2 = parent1,parent2

        #if (child1 == parent1 or child1==parent2 or child2==parent1 or child2==parent2):
        #    print('Children are equal with parents')
        #else:
        #    print('NEW OffspringS')
        #edge cases:
        
        #cand ruta rezultata contine de mai multe ori acelasi nod trebuie rectificata
        return child1, child2
    
    

        
    def mutation(self, individual):
        if len(individual) < 3:
            # If the individual is too short to perform mutation, return it as is
            return individual
        
        # Select node1 and ensure it's different from node2
        node1 = random.choice(individual[1:-1])  # Exclude the start and end nodes
        node2 = node1
        while node2 == node1:
            node2 = random.choice(individual[1:-1])  # Exclude the start and end nodes
    
        # Ensure node1 comes before node2 in the path
        if individual.index(node1) > individual.index(node2):
            node1, node2 = node2, node1
    
        # Perform mutation by replacing the subpath between node1 and node2
        # with a new valid subpath
        new_subpath = randomized_search(self.graph, node1, node2)
        
        # Find the indices of node1 and node2 in the individual
        index1 = individual.index(node1)
        index2 = individual.index(node2)
        
        # Ensure index1 is less than index2
        if index1 > index2:
            index1, index2 = index2, index1
    
        # Replace the subpath between index1 and index2 with the new subpath
        individual[index1:index2 + 1] = new_subpath  # Adjust the replacement indices
    
        return individual
        

    def fitnessEvaluation_google(self, individual):
        # Generate a unique identifier for the individual
        individual_id = tuple(individual)
    
        # Check if individual's fitness has already been evaluated
        if individual_id in self.distance_cache_for_individuals:
            print('I did nbot send another api...')
            # If the individual's data is already cached, return it
            return self.distance_cache_for_individuals[individual_id]
    
        # If not cached, fetch the data from Google API or elsewhere
        congestion_index, distance_value, duration_in_traffic_value = predict_traffic_congestion(nodes_geographical_df, individual, model_pre_trained)
    
        # Cache the data for future use
        self.distance_cache_for_individuals[individual_id] = (congestion_index, distance_value.tolist()[0], duration_in_traffic_value.tolist()[0])
    
        # Return the fetched data
        return (congestion_index, distance_value.tolist()[0], duration_in_traffic_value.tolist()[0])


    '''

        def mutation(self, individual):
        if len(individual) < 3:
            #print('Individual is too short to perform mutation')
            return individual
        mutation_point_index = random.randint(1, len(individual) - 2)
        #print('Mutation Index:', mutation_point_index)
        
        # Build a new valid path from the mutation point
        new_chromosome = randomized_search(self.graph,individual[mutation_point_index],self.destination)
        #if individual != new_chromosome:
        #    print('Mutation produced anotehr ind')
        #else:
        #    print('Mutation did nothing....')
        individual[mutation_point_index:] = new_chromosome
        

        return individual

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
    ###############################################################################
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
            shortest_path = ox.shortest_path(self.graph, node1, node2, weight='length')
            total_distance = sum(ox.utils_graph.get_route_edge_attributes(self.graph, shortest_path, attribute="length"))

            return total_distance
        except nx.NetworkXNoPath:
            return math.inf

        
    def fitnessEvaluation(self, individual):
        total_distance = self.calculate_total_distance(individual)
        return (total_distance,)  # Comma is necessary for DEAP compatibility
    '''







