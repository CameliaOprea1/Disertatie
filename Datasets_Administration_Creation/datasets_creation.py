import pandas as pd
import json
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import random
import pandas as pd
import pickle
import os
import pyproj
from shapely.geometry import Point, LineString
from datetime import date, timedelta,datetime
import time
import csv
from geopy.distance import geodesic
from polyline import decode as decode_polyline
import ast

def load_my_graph():
    city_name = "Brasov, Romania"
    # Check if the graph has already been saved
    graph_file_geographical = "C:\\Users\\Camelia\\Desktop\\app\\Disertatie\\GeneticAlg\\graph_geographical_coords.pkl"

    if os.path.exists(graph_file_geographical):
        with open(graph_file_geographical, "rb") as f:
            # Load the graph from file
            G = pickle.load(f)
    else:
        # Create the graph and save it
        G = ox.graph_from_place(city_name, network_type="drive", simplify=True)

    # Load the coordinates DataFrame
    df_coordinates = pd.read_csv('C:\\Users\\Camelia\\Desktop\\app\\Disertatie\\GeneticAlg\\AugmentedPoints.csv')

    # Add custom node weights based on page rank
    node_weights_dict = dict(zip(df_coordinates['Node_ID'], df_coordinates['influence_factor_page_rank_minMax']))
    nx.set_node_attributes(G, node_weights_dict, 'weight')

    # Check if the projected graph has already been saved
    projected_graph_file = "C:\\Users\\Camelia\\Desktop\\app\\Disertatie\\GeneticAlg\\graph_cartesian_coords.pkl"
    if os.path.exists(projected_graph_file):
        with open(projected_graph_file, "rb") as f:
            # Load the projected graph from file
            G_projected = pickle.load(f)
    else:
        # Project the graph and save it
        G_projected = ox.project_graph(G)

    # Convert graphs to GeoDataFrames
    #nodes, edges = ox.graph_to_gdfs(G)
    #nodes_cartesian, edge_cartesian = ox.graph_to_gdfs(G_projected)
    return G, G_projected



def extract_weather(df_merged):
    # Define lists to store weather information
    weather_descriptions = []
    main_weather_conditions = []
    cloud_cover_percentages = []

    # Loop through each row of the DataFrame
    for index, row in df_merged.iterrows():
        parsed_data = ast.literal_eval(row['weather'])

        # Access specific information for each row
        weather_description = parsed_data['weather'][0]['description']
        main_weather_condition = parsed_data['weather'][0]['main']
        cloud_cover_percentage = parsed_data['clouds']['all']
        
        # Append the information to the respective lists
        weather_descriptions.append(weather_description)
        main_weather_conditions.append(main_weather_condition)
        cloud_cover_percentages.append(cloud_cover_percentage)

    # Create a DataFrame from the extracted weather information
    weather_df = pd.DataFrame({
        'weather_description': weather_descriptions,
        'main_weather_condition': main_weather_conditions,
        'cloud_cover_percentage': cloud_cover_percentages
    })

    return weather_df


def plotTrafficincident(df):
    all_affected_incidents = []

    for traffic_incidents, route_steps_json in zip(df['traffic_incidents'], df['list_of_steps_json']):
        if pd.isna(traffic_incidents):
            all_affected_incidents.append([])
            continue

        data_dict = ast.literal_eval(traffic_incidents)
        route_steps_json = ast.literal_eval(route_steps_json)

        # Handle missing keys in data_dict
        try:
            incident_locations = [Point(poi['p']['y'], poi['p']['x']) for poi in data_dict['tm']['poi']]
        except KeyError:
            all_affected_incidents.append([])
            continue

        entry_affected_incidents = []
        for step in route_steps_json:
            polyline_str = step['polyline']['points']
            decoded_points = decode_polyline(polyline_str)
            
            # Check if the decoded points contain at least two elements
            if len(decoded_points) < 2:
                continue
            
            route_line = LineString(decoded_points)

            for incident in incident_locations:
                closest_distance = min(geodesic((point[1], point[0]), (incident.y, incident.x)).kilometers
                                       for point in route_line.coords)
                if closest_distance < 0.5:
                    entry_affected_incidents.append((incident.x, incident.y, closest_distance))
                    break

        all_affected_incidents.append(entry_affected_incidents)

    return all_affected_incidents


def store_incidents_as_dict(affected_incidents):
    stored_incidents = {}
    for i, incident in enumerate(affected_incidents):
        stored_incidents[i] = tuple(incident)
    return stored_incidents



def preprocess_dict(dictionary):
    # Create lists to store row indices and tuples
    indices = []
    tuples = []

    # Iterate over the dictionary
    for key, value in dictionary.items():
        indices.append(key)
        tuples.append(value)

    # Create a DataFrame from the lists
    df = pd.DataFrame({'tuples': tuples}, index=indices)
    return df


def append_to_csv(df1, df2_filename):


    df2=pd.read_csv(df2_filename) #old data
    #df1=new data
    columns_to_drop = [col for col in df2.columns if col.startswith('Unnamed:')]
    df2.drop(columns=columns_to_drop, inplace=True)

    """
    Concatenate two DataFrames and write the result to a CSV file.

    Args:
    df1 (pandas.DataFrame): First DataFrame.
    df2 (pandas.DataFrame): Second DataFrame.
    csv_filename (str): Path to the CSV file.

    Returns:
    None
    """
    df2 = df2[df1.columns]
    # Check if both DataFrames have the same columns
    if not df1.columns.equals(df2.columns):
        print("Error: The DataFrames do not have the same columns.")
        print(df1.columns)
        print(df2.columns)
        return None
    # Concatenate the two DataFrames
    concatenated_df = pd.concat([df2, df1], ignore_index=True)
    

    return concatenated_df

def load_today_saved_google_data(today_date_add_osmid):
    #Store the strat and end address as OSMID, and store the csv
    today_date_str = today_date_add_osmid.strftime("%Y-%m-%d")
    name_day = 'route_data_' + today_date_str
    df_today= pd.read_csv('C:\\Users\\Camelia\\Desktop\\app\\Disertatie\\GeneticAlg\\'+name_day+'.csv')
    G, G_projected= load_my_graph()

    df_today['origin_osmid'] = df_today.apply(lambda row: ox.nearest_nodes(G, row['start_location_lng'], row['start_location_lat']), axis=1)
    df_today['destination_osmid'] = df_today.apply(lambda row: ox.nearest_nodes(G, row['end_location_lng'], row['end_location_lat']), axis=1)
    # Identify columns to drop
    cols_to_drop = [col for col in df_today.columns if col.startswith('Unnamed')]
    # Drop the identified columns
    df_today.drop(columns=cols_to_drop, inplace=True)

    # Save the data
    output_directory = 'C:\\Users\\Camelia\\Desktop\\app\\Disertatie\\GeneticAlg'
    filename = name_day + 'osmid.csv'
    file_path = os.path.join(output_directory, filename)

    # Save DataFrame to CSV
    df_today.to_csv(file_path, index=False)

    extracted_weather_df = extract_weather(df_today)
    affecting_incidents = plotTrafficincident(df_today)
    stored_incidents_dict = store_incidents_as_dict(affecting_incidents)
    df_stored_incidents = preprocess_dict(stored_incidents_dict)

    #all_data = append_to_csv(df_today, 'MergedUntil_'+today_date.strftime("%Y-%m-%d")+'.csv')
    #all_weather_data = append_to_csv(extracted_weather_df, 'WeatherDataUntil_'+today_date.strftime("%Y-%m-%d")+'.csv')
    #all_incidents_data = append_to_csv(df_stored_incidents, 'IncidentDataUntil_'+today_date.strftime("%Y-%m-%d")+'.csv')

    return df_today, extracted_weather_df, df_stored_incidents




