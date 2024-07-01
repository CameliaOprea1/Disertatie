from .GAclass import GeneticAlgorithm
from .GAMOclass import GAMO
from .datasets_creation import load_my_graph
from .GenerateSubgraphClass import get_subgraph
import osmnx as ox
import networkx as nx
import pickle
import asyncio
import nest_asyncio
import math

# Load the graph only once to avoid redundant operations
G, G_projected= load_my_graph()
# Convert graphs to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)
nodes_cartesian, edge_cartesian = ox.graph_to_gdfs(G_projected)
print(len(G.nodes))
# Specify the filename
filename = "C:/Users/Camelia/Desktop/my_django_project/route_application/successors_dictionary.pkl"

# Load the dictionary from the file
with open(filename, 'rb') as f:
    successors_dict = pickle.load(f)

# Run the algorithm asynchronously
async def run_gamo(origin, destination, training_period, road_type,result_type):
    if result_type == 'fastest':
        print('aici')
        gd = GAMO(successors_dict, origin, destination, training_period=training_period,popsize=5, road_type=road_type)
        best, fitnessOfBest = await gd.run(logs_path="", ngen=5)
    elif result_type == 'optimal':
        gd = GAMO(successors_dict, origin, destination, training_period=training_period,popsize=12, road_type=road_type)
        best, fitnessOfBest = await gd.run(logs_path="", ngen=8)
    elif  result_type == 'extensive':
        gd = GAMO(successors_dict, origin, destination, training_period=training_period,popsize=25, road_type=road_type)
        best, fitnessOfBest = await gd.run(logs_path="", ngen=10)

    return best, fitnessOfBest

def find_optimal_route(origin, destination, training_period, road_type, result_type):
    route = ox.shortest_path(G_projected, origin, destination, weight="length")

    # Get the actual route as a GeoDataFrame of edges
    route_gdf = ox.routing.route_to_gdf(G_projected, route)
    origin_geograph = (nodes.loc[origin].y,nodes.loc[origin].x)
    destination_geograph = (nodes.loc[destination].y,nodes.loc[destination].x)

    print(route_gdf['length'].sum())
    print(origin_geograph, destination_geograph)

    ###ga = GeneticAlgorithm(G, successors_dict, source=origin, destination=destination)
    ###gd = GAMO(G, successors_dict, origin, destination,popsize=20)
    ###best, fitnessOfBest = gd.run(ngen=8,logs_path="")

    best, fitnessOfBest = asyncio.run(run_gamo(origin, destination,training_period, road_type, result_type))

    route_GA =best[0]
    route_gdf_GA = ox.routing.route_to_gdf(G_projected, route_GA) #neaparat G_projected, lucrez in sist cartezian!!
    route_gdf_geographical_GA = route_gdf_GA.to_crs(epsg=4326)
    #path = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route_GA]

    congestion_mapping = {0: 'Low Congestion', 1: 'Moderate Congestion', 2: 'High Congestion'}
    congestion_level = congestion_mapping[fitnessOfBest[0]]

    optimal_route = {
        'start_lat': origin_geograph[0],
        'start_lng': origin_geograph[1],
        'end_lat': destination_geograph[0],
        'end_lng': destination_geograph[1],
        'path': route_gdf_geographical_GA,
        'distance': fitnessOfBest[1] / 1000,  # Distance in km
        'duration': fitnessOfBest[2],  # Exact duration in seconds
        'congestion_level': congestion_level  # Example data
    }

    #optimal_route = {
    #    'start_lat': origin_geograph[0],
    #    'start_lng': origin_geograph[1],
    #    'end_lat': destination_geograph[0],
    #    'end_lng': destination_geograph[1],
    #    'path': route_gdf.to_crs(epsg=4326), # Assuming EPSG 4326 is WGS84,
    #    'distance': 1,  # Distance in km
    #    'duration': 1,  # Estimated duration in minutes
    #    'congestion_level': 'Low Congestion'  # Example data
    #}

    return optimal_route