import osmnx as ox
from collections import deque
import pandas as pd

ox.settings.log_console=True
ox.settings.use_cache=True
import random
from smart_mobility_utilities.common import randomized_search

import pickle
import sys
import os
import aiohttp
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from datetime import date

# Add the relevant directories to the system path
script_dir = os.path.dirname(__file__)
google_apis_path = os.path.join(script_dir, '..', 'Google_APIS_Request')
datasets_admin_path = os.path.join(script_dir, '..', 'Datasets_Administration_Creation')
osmnx_graph_path = os.path.join(script_dir, '..', 'Osmnx_graph_maniupulation')

sys.path.append(google_apis_path)
sys.path.append(datasets_admin_path)
sys.path.append(osmnx_graph_path)

from .get_processed_api_data import predict_traffic_congestion # Import the async script
from .datasets_creation import load_my_graph
from .randomized_search_with_bias import randomized_search_biased

'''
    In this script I define all that is needed for the genetic algorithm, i.e. genetic operators and define the fitness function
'''

##############################################################################################################################################
G, G_projected =  load_my_graph()
nodes, edges = ox.graph_to_gdfs(G)
nodes_cartesian, edge_cartesian = ox.graph_to_gdfs(G_projected)
nodes_geographical_df = nodes.to_crs(epsg=4326)  # Assuming EPSG 4326 is WGS84
today_date = date.today()

#model_pre_trained = pickle.load(open('C:\\Users\\Camelia\\Desktop\\refactored_App\\Machine_learning_modelling\\model_XGB_1.pkl','rb'))
url_openWeather = f"https://api.openweathermap.org/data/2.5/weather?lat={45.64861}&lon={25.60613}&appid=3f116979dae9f6daf340d900c87de197&units=metric"
##################################################################################################################################################


def train_model(training_period):
    df_ml = pd.read_csv('C:\\Users\\Camelia\\Desktop\\app\\Disertatie\\ML\\Last_dataset\\MLDataset.csv')

    # Convert 'timestamp' to datetime
    df_ml['timestamp'] = pd.to_datetime(df_ml['timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S', exact=False)

    # Get the current date
    current_date = df_ml['timestamp'].max()

    # Filter data based on the chosen training period
    if training_period == 'whole':
        filtered_df = df_ml
    elif training_period == 'last_2_weeks':
        start_date = current_date - timedelta(weeks=2)
        filtered_df = df_ml[df_ml['timestamp'] >= start_date]
    elif training_period == 'last_week':
        start_date = current_date - timedelta(weeks=1)
        filtered_df = df_ml[df_ml['timestamp'] >= start_date]
    else:
        raise ValueError("Invalid training period. Choose from 'whole', 'last_2_weeks', or 'last_week'.")

    # Create a copy of the filtered DataFrame
    filtered_df = filtered_df.copy()

    congestion_mapping = {
        'Low Congestion': 0,
        'Moderate Congestion': 1,
        'High Congestion': 2
    }
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
        'moderate rain': 4,
        'heavy intensity rain': 5,
        'few clouds': 6,
        'clear sky': 7
    }
    
    # Apply mappings
    filtered_df['congestion_level'] = filtered_df['congestion_level'].map(congestion_mapping)
    filtered_df['day_of_week'] = filtered_df['day_of_week'].map(day_mapping)
    filtered_df['weather_data'] = filtered_df['weather_data'].map(weather_mapping)

    # Prepare features and target variable
    X = filtered_df.drop(['congestion_level', 'timestamp', 'origin_osmid', 'destination_osmid', 'Unnamed: 0'], axis=1)
    y = filtered_df['congestion_level']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train: {len(X_train)}, X_test: {len(X_test)}")
    class_labels = ['Low Congestion', 'Moderate Congestion', 'High Congestion']

    # Initialize XGBoost classifier
    xgb_classifier = xgb.XGBClassifier()

    # Train the classifier
    xgb_classifier.fit(X_train, y_train)

    # Predictions on the testing set
    y_pred = xgb_classifier.predict(X_test)

    # Print confusion matrix and classification report
    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_labels, columns=class_labels))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    return xgb_classifier

class GeneticAlgorithm:
   

    def __init__(self,successors_dict, source, destination, population_size=50,crossover_rate=0.8,training_period = 'whole',road_type='all_roads', X_test={}):
        self.G =  G
        self.source = source
        self.destination = destination
        self.population_size = population_size
        self.successors_dict = successors_dict
        self.crossover_rate = crossover_rate
        self.distance_cache = {}
        self.distance_cache_for_individuals = {}
        self.X_test = X_test
        self.model = train_model(training_period)
        self.road_type = road_type
        print(road_type)
        # Initialize parameters

    def find_common_elements(self, list1, list2):
        # Exclude source and destination from the lists
        list1_without_ends = [node for node in list1 if node != self.source and node != self.destination]
        list2_without_ends = [node for node in list2 if node != self.source and node != self.destination]
        
        # Find common elements
        return set(list1_without_ends) & set(list2_without_ends)




        
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
        if len(individual) < 4:
            # If the individual is too short to perform mutation, return it as is
            return individual

        # Select node1 and node2 ensuring they are different and node1 comes before node2
        node1, node2 = random.sample(individual[1:-1], 2)  # Exclude the start and end nodes

        if individual.index(node1) > individual.index(node2):
            node1, node2 = node2, node1

        # Perform mutation by replacing the subpath between node1 and node2 with a new valid subpath
        new_subpath = randomized_search(self.G, node1, node2)
        # Find the indices of node1 and node2 in the individual
        index1 = individual.index(node1)
        index2 = individual.index(node2)

        # Ensure index1 is less than index2
        if index1 > index2:
            index1, index2 = index2, index1

        # Replace the subpath between index1 and index2 with the new subpath
        individual = individual[:index1] + new_subpath + individual[index2 + 1:]  # Adjust the replacement indices
        return individual
    
    async def main(self, individual, session):
        result = await predict_traffic_congestion(nodes_geographical_df, individual, self.model, session)
        print('I am in gaclass:', result[0], result[1].tolist()[0], result[2].tolist()[0])
        print(result[0], result[1].tolist()[0], result[2].tolist()[0])
        return result
    
    async def fitnessEvaluation_google(self, individual):
        async with aiohttp.ClientSession() as session:
        # Generate a unique identifier for the individual
            individual_id = tuple(individual)

            # Check if individual's fitness has already been evaluated
            if individual_id in self.distance_cache_for_individuals:
                print('I did not send another api...')
                # If the individual's data is already cached, return it
                return self.distance_cache_for_individuals[individual_id]

            # If not cached, fetch the data from Google API or elsewhere
            congestion_index, distance_value, duration_in_traffic_value = await self.main(individual, session)

            # Cache the data for future use
            self.distance_cache_for_individuals[individual_id] = (congestion_index, distance_value.tolist()[0], duration_in_traffic_value.tolist()[0])

            # Return the fetched data
            return (congestion_index, distance_value.tolist()[0], duration_in_traffic_value.tolist()[0])







