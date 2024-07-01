import pandas as pd
from datetime import datetime,timedelta
import time
import json
import os
import csv
import requests
import numpy as np
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import Point, LineString
import ast
from geopy.distance import geodesic
import aiohttp
import nest_asyncio
import asyncio

# Constants

CSV_FILENAME_PREFIX = "live_data_for_genetic_alg_"
DIST_MATRIX_FILENAME_PREFIX = "distance_matrix_data_"
###GOOGLE API
YOUR_API_KEY = 'AIzaSyBfF-hqY2s2EnowovqFi8QuvMCcRSQtDbQ'
#'AIzaSyARYh2obFP1JK6PEcDT7q0ASA_e5k-kgB0'
#####WEATHER
url_openWeather = f"https://api.openweathermap.org/data/2.5/weather?lat={45.64861}&lon={25.60613}&appid=3f116979dae9f6daf340d900c87de197&units=metric"
###TRAFFIC INCIDENTS
apiKeytraffic ='XhCMZTMAH9SFF0HGGj5Qd99dSAWvUIvt'
# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"{CSV_FILENAME_PREFIX}{current_date}.csv"
dist_matrix_filename = f"{DIST_MATRIX_FILENAME_PREFIX}{current_date}.csv"
coordinates_filename = f"coordinates_{current_date}.csv"
ml_dataset_filename = "RawData_live_responses.csv"
batch_size = 10  # Define the batch size
label_encoder = LabelEncoder()
holiday_dates = {
    'Ziua Muncii': '2024-05-01',
    'Vinerea Mare': '2024-05-03',
    'Paștele_1': '2024-05-04',
    'Paștele_2': '2024-05-05',
    'Paștele_3': '2024-05-06',
    'Ziua Copilului': '2024-06-01'
}

# Convert holiday dates to datetime objects
for holiday, date in holiday_dates.items():
    holiday_dates[holiday] = pd.to_datetime(date).date()

nest_asyncio.apply()

# Helper functions
def get_geograph_coordinates_from_osmid(nodes_geographical_df, osmid_node):
    ''' from osmid --> (lat, lng) '''
    return nodes_geographical_df.loc[osmid_node].y, nodes_geographical_df.loc[osmid_node].x

def transform_to_coordinates(nodes_df, chromosome):
    ''' from [osmid_1,..., omsid_n] --> [(lat_1, lng_1),...,(lat_n, lng_n)] '''
    return [get_geograph_coordinates_from_osmid(nodes_df, node) for node in chromosome]

def format_coordinates_for_api(coordinates):
    ''' from [(lat_1, lng_1),...,(lat_n, lng_n)] --> 'lat_1,lng_1|...|lat_n, lng_n' '''
    return "|".join([f"{lat},{lng}" for lat, lng in coordinates])

async def fetch_snapped_points(session, coordinates, api_key):
    ''' Get the correct geographical coordinates from lat_1, lng_1),...,(lat_n, lng_n) provided by OSMN module '''
    path = format_coordinates_for_api(coordinates)
    url = f"https://roads.googleapis.com/v1/snapToRoads?path={path}&key={api_key}"
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            print("Request to Google Roads API failed with status code:", response.status)
            return []

async def fetch_distance_matrix(session, origins, destinations, api_key):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "departure_time": "now",
        "origins": "|".join([f"{origin[0]},{origin[1]}" for origin in origins]),
        "destinations": "|".join([f"{dest[0]},{dest[1]}" for dest in destinations]),
        "key": api_key,
    }
    async with session.get(url, params=params) as response:
        return await response.json()

async def fetch_weather(session, url_open_weather):
    async with session.get(url_open_weather) as response:
        return await response.json()

async def fetch_traffic_incidents(session, bounding_box, api_key_traffic):
    url = f"https://api.tomtom.com/traffic/services/4/incidentDetails/s3/{bounding_box}/22/-1/json?key={api_key_traffic}&projection=EPSG4326&originalPosition=true"
    async with session.get(url) as response:
        return await response.json()


def create_batches(coordinates, batch_size):
    batches = []
    for i in range(0, len(coordinates)-1, batch_size):
        origins_batch = coordinates[i:i+batch_size]
        destinations_batch = coordinates[i+1:i+batch_size+1]
        batches.append((origins_batch, destinations_batch))
    return batches

    
def calculate_bounding_box(coordinates):

    ''' Disatnce matrix API does not have the bounds of a route which is needed for the live traffic incidents
    '''
    # Initialize min and max values
    min_lat = coordinates[0][0]
    max_lat = coordinates[0][0]
    min_lon = coordinates[0][1]
    max_lon = coordinates[0][1]

    # Iterate through all coordinates
    for coord in coordinates:
        lat, lon = coord
        # Update minimum and maximum latitude
        if lat < min_lat:
            min_lat = lat
        elif lat > max_lat:
            max_lat = lat
        
        # Update minimum and maximum longitude
        if lon < min_lon:
            min_lon = lon
        elif lon > max_lon:
            max_lon = lon

    bounding_data = [['northeast', max_lat, max_lon], ['southwest', min_lat, min_lon]]
    return bounding_data

def format_bbox(bounding_box):
    ne_lat = bounding_box[0][1]
    ne_lng = bounding_box[0][2]
    sw_lat = bounding_box[1][1]
    sw_lng = bounding_box[1][2]
    bbox_string = f"{ne_lat},{ne_lng},{sw_lat},{sw_lng}"
    return bbox_string

def process_bounds(bounds_dict):
    """
    Process the bounds data contained in a dictionary retrieved from directions API.

    Args:
    bounds_dict (dict): A dictionary containing bounds data.

    Returns:
    list: A list containing the formatted bounds data.
    """
    # Extract latitude and longitude coordinates for northeast and southwest points
    northeast_lat = bounds_dict['northeast']['lat']
    northeast_lng = bounds_dict['northeast']['lng']
    southwest_lat = bounds_dict['southwest']['lat']
    southwest_lng = bounds_dict['southwest']['lng']

    # Create a list containing the formatted bounds data
    bounds_data = [['northeast', northeast_lat, northeast_lng], ['southwest', southwest_lat, southwest_lng]]

    return bounds_data


def extract_weather(df_merged, row_title='weather_json'):
    weather_descriptions, main_weather_conditions, cloud_cover_percentages = [], [], []
    for _, row in df_merged.iterrows():
        parsed_data = json.loads(row[row_title])
        weather_description = parsed_data['weather'][0]['description']
        main_weather_condition = parsed_data['weather'][0]['main']
        cloud_cover_percentage = parsed_data['clouds']['all']
        weather_descriptions.append(weather_description)
        main_weather_conditions.append(main_weather_condition)
        cloud_cover_percentages.append(cloud_cover_percentage)
    weather_df = pd.DataFrame({
        'weather_description': weather_descriptions,
        'main_weather_condition': main_weather_conditions,
        'cloud_cover_percentage': cloud_cover_percentages
    })
    return weather_df


def get_metrics_from_response(data1):# Extract distances and durations
    rows = data1["rows"]
    distances = []
    durations = []
    durations_in_traffic = []
    for row in rows:
            elements = row["elements"]
            distances_row = [element["distance"]["value"] for element in elements]
            durations_row = [element["duration"]["value"] for element in elements]
            durations_in_traffic_row = [element["duration_in_traffic"]["value"] for element in elements]
            distances.append(distances_row)
            durations.append(durations_row)
            durations_in_traffic.append(durations_in_traffic_row)
    return distances, durations, durations_in_traffic


def preprocess_distance_matrices(df, row_name='matrices_json'):
    ''' 
    In this method it is calculated the total time in traffic, average traffic, and distance for my list of snapped coordinates that forms the path
    '''
    distance_matrices = json.loads(df[row_name].iloc[0])

    total_distance = 0
    total_time_in_traffic = 0
    total_average_time = 0

    for index, distance_matrix in enumerate(distance_matrices, start=1):
        try:
            distances, durations, durations_in_traffic = get_metrics_from_response(distance_matrix)
            
            # Determine the minimum dimension of the matrix
            min_dimension = min(len(distances), len(distances[0]))
            
            diagonal_sum = sum(distances[i][i] for i in range(min_dimension))
            diagonal_sum_time_in_traffic = sum(durations_in_traffic[i][i] for i in range(min_dimension))
            diagonal_sum_time = sum(durations[i][i] for i in range(min_dimension))
            
            print("Sum of diagonal elements for distances{}: {}".format(index, diagonal_sum))
            
            total_distance += diagonal_sum
            total_time_in_traffic += diagonal_sum_time_in_traffic
            total_average_time += diagonal_sum_time
        
        except Exception as e:
            print("Error processing distance matrix {}: {}".format(index, str(e)))
    
    print("Total distance: ", total_distance)
    print("Total time in traffic: ", total_time_in_traffic / 60)
    print("Total average time: ", total_average_time / 60)
    
    return total_distance, total_time_in_traffic, total_average_time

def get_traffic_incidents_categorical(df, coordinates):
    ''' Method for retrieving the relevant information from traffic incidnts API
    '''
    traffic_incidents = df['traffic_json'].iloc[0]  # Assuming there's only one row

    if pd.isna(traffic_incidents):
        return 0  # Return an empty tuple if there are no affected incidents

    data_dict = ast.literal_eval(traffic_incidents)
    
    # Handle missing keys in data_dict
    try:
        incident_locations = [Point(poi['p']['y'], poi['p']['x']) for poi in data_dict['tm']['poi']]
    except KeyError:
        return 0 # Return an empty tuple if there are no affected incidents

    route_line = LineString(coordinates)
    affected_incidents = []

    for incident in incident_locations:
        closest_distance = min(geodesic((point[1], point[0]), (incident.y, incident.x)).kilometers
                               for point in route_line.coords)
        if closest_distance < 0.5:
            affected_incidents.append((incident.x, incident.y, closest_distance))

    if affected_incidents:
        return 1  # Return 1 if there are affected incidents
    else:
        return 0  # Return 0 if there are no affected incidents


# Function to check if a given timestamp is a holiday
def is_holiday(timestamp):
    for date in holiday_dates.values():
        if timestamp.date() == date:
            return 1
    return 0

def categorize_hour(hour):
    if 7 <= hour < 9:
        return 0
    elif 9 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    else:
        return 3
    
async def get_distance_matrices_for_batches(session, batches, api_key):
    tasks = []
    for batch in batches:
        origins, destinations = batch
        task = fetch_distance_matrix(session, origins, destinations, api_key)
        tasks.append(task)
        
    distance_matrices = await asyncio.gather(*tasks)
    return distance_matrices

async def process_route_information(chromosome, nodes_geographical_df, session, api_key = YOUR_API_KEY, weather_url = url_openWeather, traffic_key =  apiKeytraffic):
    # Transform chromosome into coordinates
    coordinates = transform_to_coordinates(nodes_geographical_df, chromosome)

    # Fetch snapped points asynchronously
    snapped_points_tasks = [fetch_snapped_points(session,coordinates,api_key)]
    snapped_points_results = await asyncio.gather(*snapped_points_tasks)
    snapped_coordinates =  [(item['location']['latitude'], item['location']['longitude']) for item in snapped_points_results[0]["snappedPoints"]]


    # Divide snapped points into batches for distance matrix calculation
    batches = create_batches(snapped_coordinates, batch_size)

    # Fetch distance matrices for each batch asynchronously
    
    distance_matrix_results = await get_distance_matrices_for_batches(session, batches, api_key)

     # Fetch weather and traffic information
    weather_task = fetch_weather(session, weather_url)
    bounding_box = calculate_bounding_box(coordinates)
    traffic_task = fetch_traffic_incidents(session, format_bbox(bounding_box), traffic_key)

    # Wait for all tasks to complete
    weather_result = await weather_task
    traffic_result = await traffic_task

    # Serialize each response as JSON
    #snapped_points_json = json.dumps(snapped_points_results)
    distance_matrix_json = json.dumps(distance_matrix_results)
    weather_json = json.dumps(weather_result)
    traffic_json = json.dumps(traffic_result)

    dict_response = {
        "snapped_points": snapped_points_results,
        "matrices_json": distance_matrix_json,
        "weather_json": weather_json,
        "traffic_json": traffic_json
    }

    return dict_response

async def get_google_dist_matrix_response_and_preprocess(nodes_geographical_df,chromosome,api_key, weather_url, traffic_key, session):

    ''' 
    Wrapping all the methods empoyed for retrieving and preprocess Distance matrix API, weather API, and traffic incidents API
    '''
    async with aiohttp.ClientSession() as session:
        route_info_dict = await process_route_information(chromosome, nodes_geographical_df, session, api_key, weather_url, traffic_key)
        coordinates = [(item['location']['latitude'], item['location']['longitude']) for item in route_info_dict["snapped_points"][0]["snappedPoints"]]

        # Step 3: Process response
        df = pd.DataFrame(route_info_dict)

        return df, coordinates


async def predict_traffic_congestion(nodes_geographical_df,chromosome, model_pre_trained,session,api_key = YOUR_API_KEY, weather_url = url_openWeather, traffic_key = apiKeytraffic ):
    async with aiohttp.ClientSession() as session:
        df, coordinates = await get_google_dist_matrix_response_and_preprocess(nodes_geographical_df,chromosome,api_key, weather_url, traffic_key, session)
    
        # Step 4: Preprocess distance matrices
        df['fetched_coordinates'] = [coordinates]
        df['distance_value'], df['duration_in_traffic_numeric'], df['duration_value'] = preprocess_distance_matrices(df, row_name='matrices_json')

        # Step 5: Compute average speed
        df['intraffic_speed'] = 3.6 * (df['distance_value'] / df['duration_in_traffic_numeric'])  # in km/h
        df['average_speed'] = 3.6 * (df['distance_value'] / df['duration_value'])  # in km/h

        mode = 'a' if os.path.exists(ml_dataset_filename) else 'w'
        with open(ml_dataset_filename, mode=mode, encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                if mode == 'w':
                    # Write headers only if it's a new file
                    writer.writerow(["chromosome", "initial_coordinates", "snapped_coordinates", "distance_matrix","weather","traffic"])
                writer.writerow([chromosome, coordinates, df['fetched_coordinates'].iloc[0], df['matrices_json'].iloc[0], df['weather_json'].iloc[0], df['traffic_json'].iloc[0]])


        # Step 6: Extract features for machine learning
        df_ml = pd.DataFrame()
        df['timestamp']  = datetime.now()
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df_ml['weather_data'] = extract_weather(df, row_title='weather_json')['weather_description']
        df_ml['day_of_week'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.day_name()
        df_ml['holiday_data'] = pd.to_datetime(df['timestamp'], errors='coerce').apply(is_holiday)
        df_ml['traffic_incidents'] = get_traffic_incidents_categorical(df, coordinates)
        df_ml['intraffic_speed'] = df['intraffic_speed'].values
        df_ml['average_speed'] = df['average_speed'].values
        df_ml['distance_value'] = df['distance_value'].values
        df_ml['time_of_day'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour.apply(categorize_hour)
        df_ml['hour'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour


        day_mapping = {
            'Monday': 0,
            'Tuesday': 1,
            'Wednesday': 2,
            'Thursday': 3,
            'Friday': 4,
            'Saturday': 5,
            'Sunday': 6
        }
        weather_mapping={
        'scattered clouds': 0,
        'broken clouds': 1,
        'overcast clouds': 2,
        'light rain': 3,
        'moderate rain' : 4,
        'heavy intensity rain': 5,
        'few clouds': 6,
        'clear sky':7

        }
        # Step 7: Encode categorical features
        df_ml['day_of_week'] = df_ml['day_of_week'].map(day_mapping)
        df_ml['weather_data'] = df_ml['weather_data'].map(weather_mapping)

        # Define file name
        second_filename = 'Prepocessed_ml_live_data.csv'
        second_file_exists = os.path.isfile(second_filename)
        selected_columns = ["weather_data", "day_of_week", "holiday_data", "traffic_incidents","intraffic_speed","average_speed","distance_value","time_of_day","hour"]
        # Write data to second CSV
        if not second_file_exists:
            df_ml[selected_columns].to_csv(second_filename, mode='a', header=True, index=False)
        else:
            df_ml[selected_columns].to_csv(second_filename, mode='a', header=False, index=False)

        # Step 8: Make predictions
        prediction = model_pre_trained.predict(df_ml)

        return prediction.tolist()[0], df['distance_value'].values, df['duration_in_traffic_numeric'].values