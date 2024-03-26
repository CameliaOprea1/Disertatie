import osmnx as ox
import networkx as nx
import geopandas as gpd
import random
import pandas as pd
import pickle
import os
import pyproj
from shapely.geometry import Point, LineString


city_name = "Brasov, Romania"
# Check if the graph has already been saved
graph_file_geographical = "graph_geographical_coords.pkl"

if os.path.exists(graph_file_geographical):
    with open(graph_file_geographical, "rb") as f:
        # Load the graph from file
        G = pickle.load(f)
else:
    # Create the graph and save it
    G = ox.graph_from_place(city_name, network_type="drive", simplify=True)

# Load the coordinates DataFrame
df_coordinates = pd.read_csv('AugmentedPoints.csv')

# Add custom node weights based on page rank
node_weights_dict = dict(zip(df_coordinates['Node_ID'], df_coordinates['influence_factor_page_rank_minMax']))
nx.set_node_attributes(G, node_weights_dict, 'weight')

# Check if the projected graph has already been saved
projected_graph_file = "graph_cartesian_coords.pkl"
if os.path.exists(projected_graph_file):
    with open(projected_graph_file, "rb") as f:
        # Load the projected graph from file
        G_projected = pickle.load(f)
else:
    # Project the graph and save it
    G_projected = ox.project_graph(G)

# Convert graphs to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)
nodes_cartesian, edge_cartesian = ox.graph_to_gdfs(G_projected)

# Define the projection systems
source_crs = 'epsg:32635' # Coordinate system of the file
#print(G.graph["crs"])
target_crs = 'epsg:4326' # Global lat-lon coordinate system

def get_subgraph():
    rand_points = random.choices(list(G.nodes), weights=df_coordinates['influence_factor_page_rank_minMax'], k=2)
    origin = rand_points[0]
    destination = rand_points[1]

    route = ox.shortest_path(G_projected, origin, destination, weight="length")

    # Get the actual route as a GeoDataFrame of edges
    route_gdf = ox.utils_graph.route_to_gdf(G_projected, route)

    # Calculate route length
    route_length = route_gdf['length'].sum()

    # Calculate the bounding box of the route
    minx, miny, maxx, maxy = route_gdf.total_bounds
    #route_bounding_box = (minx, miny, maxx, maxy)

    # Calculate the radius of the circle to contain the entire route
    radius = max(ox.distance.euclidean(miny, minx, maxy, maxx) / 2, route_length / 2)

    # Calculate the center point of the bounding box
    center_point = ((maxy + miny) / 2, (maxx + minx) / 2)

    print(f'Radius={radius}, Route Length={route_length}')
    # Convert center point to geographical coordinates
    center_point_geographical = Point(cartesian_to_geographical(center_point[1], center_point[0],source_crs,target_crs))
    # Convert the radius to geographical units
    center_point_cartesian = Point(center_point[1], center_point[0])
    #radius_geographical = center_point_cartesian.buffer(radius)
    # Convert route geometries to geographical coordinates
    #route_gdf_geographical = route_gdf.to_crs(epsg=4326)  # Assuming EPSG 4326 is WGS84
    # Convert node geometries to geographical coordinates
    nodes_geographical = nodes.to_crs(epsg=4326)  # Assuming EPSG 4326 is WGS84
    inside_nodes = get_points_inside_circle2(nodes_cartesian,radius, center_point_cartesian)
    successors_dict = {node: list(G.successors(node)) for node in G.nodes()}
    # Filter the nodes GeoDataFrame based on the subset of node IDs
    subset_of_nodes = nodes[nodes.index.isin(inside_nodes.index)]
    
    #ox.plot_graph(subgraph, bgcolor='k', node_color='r', node_size=30)



    info_dict ={}
    info_dict['origin'] = origin
    info_dict['destination'] = destination
    info_dict['radius'] = radius
    info_dict['center_point'] = center_point
    info_dict['route_gdf'] = route_gdf
    info_dict['route'] = route
    info_dict['successors_dict'] = successors_dict
    info_dict['subset_of_nodes'] = subset_of_nodes
    info_dict['nodes_geographical'] = nodes_geographical

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