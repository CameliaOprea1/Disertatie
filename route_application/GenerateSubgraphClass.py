import osmnx as ox
import networkx as nx
import random
import pyproj
import pandas as pd
from shapely.geometry import Point
import pickle
from .datasets_creation import load_my_graph

#####################################################
G, G_projected =  load_my_graph()
nodes, edges = ox.graph_to_gdfs(G)
nodes_cartesian, edge_cartesian = ox.graph_to_gdfs(G_projected)
# Define the projection systems
source_crs = 'epsg:32635' # Coordinate system of the file
#print(G.graph["crs"])
target_crs = 'epsg:4326' # Global lat-lon coordinate system
# Load the coordinates DataFrame
df_coordinates = pd.read_csv('C:\\Users\\Camelia\\Desktop\\app\\Disertatie\\GeneticAlg\\AugmentedPoints.csv')

# Add custom node weights based on page rank
node_weights_dict = dict(zip(df_coordinates['Node_ID'], df_coordinates['influence_factor_page_rank_minMax']))
nx.set_node_attributes(G, node_weights_dict, 'weight')

# Specify the filename

filename = "C:/Users/Camelia/Desktop/my_django_project/route_application/successors_dictionary.pkl"

# Load the dictionary from the file
with open(filename, 'rb') as f:
    successors_dict = pickle.load(f)
#####################################################


def get_subgraph(return_fields=None):

    if return_fields is None:
        return_fields = ['origin', 'origin_geogr', 'destination', 'destination_geogr', 'radius', 'center_point', 'route_gdf', 'route', 'successors_dict', 'subset_of_nodes', 'nodes_geographical', 'route_gdf_geographical', 'center_point_geographical']
    

    rand_points = random.choices(list(G.nodes), weights=df_coordinates['influence_factor_page_rank_minMax'], k=2)
    origin = rand_points[0]
    destination = rand_points[1]

    #origin_geograph = Point(cartesian_to_geographical(*G.nodes[origin]['x'], source_crs, target_crs))
    origin_geograph = (nodes.loc[origin].y,nodes.loc[origin].x)
    destination_geograph = (nodes.loc[destination].y,nodes.loc[destination].x)
    #destination_geograph = Point(cartesian_to_geographical(*G.nodes[destination]['x'], source_crs, target_crs))

    route = ox.shortest_path(G_projected, origin, destination, weight="length")

    # Get the actual route as a GeoDataFrame of edges
    route_gdf = ox.routing.route_to_gdf(G_projected, route)

    # Calculate route length
    route_length = route_gdf['length'].sum()

    # Calculate the bounding box of the route
    minx, miny, maxx, maxy = route_gdf.total_bounds
    #route_bounding_box = (minx, miny, maxx, maxy)

    # Calculate the radius of the circle to contain the entire route
    radius = max(ox.distance.euclidean(miny, minx, maxy, maxx) / 2, route_length / 2)

    # Calculate the center point of the bounding box
    center_point = ((maxy + miny) / 2, (maxx + minx) / 2)
    # Convert center point to geographical coordinates
    center_point_geographical = Point(cartesian_to_geographical(center_point[1], center_point[0],source_crs,target_crs))
    # Convert the radius to geographical units
    center_point_cartesian = Point(center_point[1], center_point[0])

    # Convert route geometries to geographical coordinates
    route_gdf_geographical = route_gdf.to_crs(epsg=4326)  # Assuming EPSG 4326 is WGS84
    # Convert node geometries to geographical coordinates
    nodes_geographical = nodes.to_crs(epsg=4326)  # Assuming EPSG 4326 is WGS84
    inside_nodes = get_points_inside_circle2(nodes_cartesian,radius, center_point_cartesian)
    # Filter the nodes GeoDataFrame based on the subset of node IDs
    subset_of_nodes = nodes[nodes.index.isin(inside_nodes.index)]
    info_dict ={}

    for field in return_fields:
        if field == 'origin':
            info_dict['origin'] = origin
        elif field == 'origin_geogr':
            info_dict['origin_geogr'] = origin_geograph
        elif field == 'destination':
            info_dict['destination'] = destination
        elif field == 'destination_geogr':
            info_dict['destination_geogr'] = destination_geograph
        elif field == 'radius':
            info_dict['radius'] = radius
        elif field == 'center_point':
            info_dict['center_point'] = center_point
        elif field == 'route_gdf':
            info_dict['route_gdf'] = route_gdf
        elif field == 'route':
            info_dict['route'] = route
        elif field == 'subset_of_nodes':
            info_dict['subset_of_nodes'] = subset_of_nodes
        elif field == 'nodes_geographical':
            info_dict['nodes_geographical'] = nodes_geographical
        elif field == 'route_gdf_geographical':
            info_dict['route_gdf_geographical'] = route_gdf_geographical
        elif field == 'center_point_geographical':
            info_dict['center_point_geographical'] = center_point_geographical

    return info_dict




# Function to convert Cartesian coordinates to geographical coordinates
def cartesian_to_geographical(x, y, source_crs=source_crs,target_crs=target_crs):
    polar_to_latlon = pyproj.Transformer.from_crs(source_crs, target_crs)
    lat, lon = polar_to_latlon.transform(x, y)
    return lat,lon

def geographical_to_cartesian(x, y, source_crs=target_crs,target_crs=source_crs):
    polar_to_latlon = pyproj.Transformer.from_crs(source_crs, target_crs)
    lat_cartezian, lon_cartezian = polar_to_latlon.transform(x, y)
    return lat_cartezian, lon_cartezian

def get_points_inside_circle2(nodes_geographical, radius, center_point):
    # Create a circular buffer around the center point
    buffer_geometry = center_point.buffer(radius)
    # Use spatial indexing to efficiently find points inside the buffer
    inside_circle_gdf = nodes_geographical[nodes_geographical.geometry.intersects(buffer_geometry)]
    inside_circle_gdf = inside_circle_gdf.to_crs(epsg=4326)
    geometry_df = inside_circle_gdf[['geometry']].copy()
    return geometry_df

    

