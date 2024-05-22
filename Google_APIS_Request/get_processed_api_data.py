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
from polyline import decode as decode_polyline
import ast
from geopy.distance import geodesic


# Constants

CSV_FILENAME_PREFIX = "live_data_for_genetic_alg_"
###GOOGLE API
YOUR_API_KEY = 'AIzaSyBlraUkOF9TgoWLZKqj14SbXHyt18_ecC0'
#'AIzaSyARYh2obFP1JK6PEcDT7q0ASA_e5k-kgB0'
#####WEATHER
url_openWeather = f"https://api.openweathermap.org/data/2.5/weather?lat={45.64861}&lon={25.60613}&appid=3f116979dae9f6daf340d900c87de197&units=metric"
###TRAFFIC INCIDENTS
apiKeytraffic ='XhCMZTMAH9SFF0HGGj5Qd99dSAWvUIvt'
# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"{CSV_FILENAME_PREFIX}{current_date}.csv"
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

def get_geograph_coordonates_from_osmid(nodes_geographical_df,osmid_node):
    ''' 
    from osmid --> (lat, lng) 
    '''
    return nodes_geographical_df.loc[osmid_node].y, nodes_geographical_df.loc[osmid_node].x

def transform_to_coordinates(nodes_df, chromosome):
    ''' 
    from [osmid_1,..., omsid_n] --> [(lat_1, lng_1),...,(lat_n, lng_n)] 
    '''
    return [get_geograph_coordonates_from_osmid(nodes_df, node) for node in chromosome]

def format_coordinates_for_api(coordinates):
    ''' 
    from [(lat_1, lng_1),...,(lat_n, lng_n)] --> 'lat_1,lng_1|...|lat_n, lng_n'
    '''
    return "|".join([f"{lat},{lng}" for lat, lng in coordinates])

def fetch_snapped_points(coordinates, api_key):
    '''
    Get the correct geograph coordinates from lat_1, lng_1),...,(lat_n, lng_n) provided by OSMN module
    '''
    path = format_coordinates_for_api(coordinates)
    url = f"https://roads.googleapis.com/v1/snapToRoads?path={path}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["snappedPoints"]
    else:
        print("Request to Google Roads API failed with status code:", response.status_code)
        return []
    

def get_distance_matrix(origins, destinations, api_key):
    # Google Distance Matrix API URL, matrix of (len(origins), len(destinations))
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    # Parameters for the request
    params = {
        "departure_time": "now",
        "origins": "|".join([f"{origin[0]},{origin[1]}" for origin in origins]),
        "destinations": "|".join([f"{dest[0]},{dest[1]}" for dest in destinations]),
        "key": api_key,
    }

    # Make the request
    response = requests.get(url, params=params)
    data = response.json()
    return data

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

def get_distance_matrices_for_batches(batches, api_key):
    ''' There are cases in which len(coordinates) > 10(max nr of destinations/origins from a request)
    '''
    distance_matrices = []
    for batch in batches:
        origins, destinations = batch
        distance_matrix = get_distance_matrix(origins, destinations, api_key)
        distance_matrices.append(distance_matrix)
    return distance_matrices


def create_batches(coordinates, batch_size):
    batches = []
    for i in range(0, len(coordinates)-1, batch_size):
        origins_batch = coordinates[i:i+batch_size]
        destinations_batch = coordinates[i+1:i+batch_size+1]
        batches.append((origins_batch, destinations_batch))
    return batches


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

def process_legs(route_info):
    """
    Create a DataFrame containing information about a route retrieved from Directions API.

    Args:
    route_info (dict): Information about the route.

    Returns:
    pd.DataFrame: DataFrame containing the route information.
    """
    data = {
        'distance_text': route_info['distance']['text'],
        'distance_value': route_info['distance']['value'],
        'duration_text': route_info['duration']['text'],
        'duration_in_traffic': route_info['duration_in_traffic']['text'],
        'duration_value': route_info['duration']['value'],
        'start_address': route_info['start_address'],
        'start_location_lat': route_info['start_location']['lat'],
        'start_location_lng': route_info['start_location']['lng'],
        'end_address': route_info['end_address'],
        'end_location_lat': route_info['end_location']['lat'],
        'end_location_lng': route_info['end_location']['lng']
    }

    return pd.Series(data)


def extract_weather(df_merged, row_title='weather_json'):
    # Define lists to store weather information
    weather_descriptions = []
    main_weather_conditions = []
    cloud_cover_percentages = []

    # Loop through each row of the DataFrame
    for index, row in df_merged.iterrows():
        parsed_data = ast.literal_eval(row[row_title])

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


################################### These function are used for the live response

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

def process_response(coordinates,url_openWeather,apiKeytraffic,api_key = YOUR_API_KEY):
    ''' Method for constructing the needed datyaframe with the columns containing: google API, weather API, live traffic incidents API
    '''
    #origins=coordinates[:-1] #need to test it later on
    #destinations=coordinates[1:]
    response_weather = requests.get(url_openWeather).json()
    bounding_box = calculate_bounding_box(coordinates)
    bounding_box = format_bbox(bounding_box)
    urltraffic = f"https://api.tomtom.com/traffic/services/4/incidentDetails/s3/{bounding_box}/22/-1/json?key={apiKeytraffic}&projection=EPSG4326&originalPosition=true"
    response_traffic=  requests.get(urltraffic).json()
    batches = create_batches(coordinates, batch_size)
    distance_matrices = get_distance_matrices_for_batches(batches, api_key)

    matrices_json = json.dumps(distance_matrices)
    matrices_json = {'matrices_json': matrices_json}
    matrices_json_df = pd.DataFrame(matrices_json, index=[0])

    #distance_matrices_df = pd.DataFrame(distance_matrices)
    weather_json = json.dumps(response_weather)
    # Construct DataFrame for weather
    weather_data = {'weather_json': weather_json}
    weather_df = pd.DataFrame(weather_data, index=[0])


    response_traffic_json = json.dumps(response_traffic)
        # Construct DataFrame for weather
    response_traffic = {'traffic_json': response_traffic_json}
    traffic_incidents_df = pd.DataFrame(response_traffic, index=[0])

    # Combine all DataFrames horizontally
    new_entry = pd.concat([matrices_json_df, weather_df, traffic_incidents_df], axis=1)

    # Add timestamp column
    new_entry['timestamp'] = datetime.now()

    return new_entry

def preprocess_distance_matrices(df, row_name='matrices_json'):
    ''' In this method it is calculated the total time in traffic, average traffic, and distance for my list of snapped coordinates that forms the path
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

def get_google_dist_matrix_response_and_preprocess(nodes_geographical_df,chr):

    ''' Wrapping all the methods empoyed for retrieving and preprocess Distance matrix API, weather API, and traffic incidents API
    '''
    # Step 1: Transform coordinates
    coordinates = transform_to_coordinates(nodes_geographical_df, chr)
    
    # Step 2: Fetch snapped points
    snapped_points = fetch_snapped_points(coordinates, YOUR_API_KEY)
    coordinates = [(item['location']['latitude'], item['location']['longitude']) for item in snapped_points]
    
    # Step 3: Process response
    df = process_response(coordinates, url_openWeather, apiKeytraffic)

    return df, coordinates


def predict_traffic_congestion(nodes_geographical_df,chr, model_pre_trained):
    
    df, coordinates = get_google_dist_matrix_response_and_preprocess(nodes_geographical_df,chr)
    
    # Step 4: Preprocess distance matrices
    df['fetched_coordinates'] = [coordinates]
    df['distance_value'], df['duration_in_traffic_numeric'], df['duration_value'] = preprocess_distance_matrices(df, row_name='matrices_json')
    
    # Step 5: Compute average speed
    df['intraffic_speed'] = 3.6 * (df['distance_value'] / df['duration_in_traffic_numeric'])  # in km/h
    df['average_speed'] = 3.6 * (df['distance_value'] / df['duration_value'])  # in km/h
    
    df.to_csv('RawData_live_response'+str(coordinates[0])+'_'+str(coordinates[-1]) +'.csv')

    # Step 6: Extract features for machine learning
    df_ml = pd.DataFrame()
    df_ml['weather_data'] = extract_weather(df, row_title='weather_json')['weather_description']
    df_ml['day_of_week'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.day_name()
    df_ml['holiday_data'] = pd.to_datetime(df['timestamp'], errors='coerce').apply(is_holiday)
    df_ml['traffic_incidents'] = get_traffic_incidents_categorical(df, coordinates)
    df_ml['intraffic_speed'] = df['intraffic_speed'].values
    df_ml['average_speed'] = df['average_speed'].values
    df_ml['distance_value'] = df['distance_value'].values
    df_ml['time_of_day'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour.apply(categorize_hour)
    df_ml['hour'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour

    # Step 7: Encode categorical features
    df_ml['day_of_week'] = label_encoder.fit_transform(df_ml['day_of_week'])
    df_ml['weather_data'] = label_encoder.fit_transform(df_ml['weather_data'])
    df_ml.to_csv('RawData_ml_response'+str(coordinates[0])+'_'+str(coordinates[-1]) +'.csv')
    # Step 8: Make predictions
    prediction = model_pre_trained.predict(df_ml)
    
    return prediction.tolist()[0], df['distance_value'].values, df['duration_in_traffic_numeric'].values