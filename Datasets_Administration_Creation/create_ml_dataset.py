import pandas as pd
import numpy as np
import pandas as pd
from datetime import  timedelta
from datetime import date

today_date = date.today()
formatted_date = today_date.strftime("%Y-%m-%d")
prior_day = today_date - timedelta(days=1)
#root_path_to_csvs = 'C:\\Users\\Camelia\\Desktop\\app\\Disertatie\\GeneticAlg\\'
    # Function to check if a given timestamp is a holiday
holiday_dates = {
    'Ziua Muncii': '2024-05-01',
    'Vinerea Mare': '2024-05-03',
    'Paștele_1': '2024-05-04',
    'Paștele_2': '2024-05-05',
    'Paștele_3': '2024-05-06',
    'Ziua Copilului': '2024-06-01'}

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

def create_ml_dataset(df_google, df_weather, df_traffic_incidents):
    df_google['timestamp']=pd.to_datetime(df_google['timestamp'],errors='coerce')
    df_google['day_of_week'] = df_google['timestamp'].dt.day_name()
    # Convert duration values to numeric
    df_google['duration_in_traffic_numeric'] = df_google['duration_in_traffic'].str.extract('(\d+)').astype(int)
    df_google['duration_text_numeric'] = df_google['duration_text'].str.extract('(\d+)').astype(int)
    df_google['hour'] = df_google['timestamp'].dt.hour
    df_google['congestion_ratio'] = ((df_google['duration_in_traffic_numeric'] - df_google['duration_text_numeric']) / df_google['duration_text_numeric'])
    df_google['normalized_congestion']  = df_google['congestion_ratio'] + abs(df_google['congestion_ratio'].min())
    # Adding a new column to the DataFrame for average speed
    df_google['intraffic_speed'] = 3.6*(df_google['distance_value'] / (df_google['duration_in_traffic_numeric']*60)) #in km/h
    df_google['average_speed'] = 3.6*(df_google['distance_value'] / (df_google['duration_value'])) #in km/h
    # Calculate quartiles and median
    Q1 = np.percentile(df_google['normalized_congestion'], 25)
    Q3 = np.percentile(df_google['normalized_congestion'], 75)
    #median = df_scaled[scoretype].median()
    IQR= Q3-Q1
    median = df_google['normalized_congestion'].median()
    congestion_bins = [df_google['normalized_congestion'].min()-1e-6, median, Q3+1.5*IQR, df_google['normalized_congestion'].max()]  # You can adjust these bins based on your requirements
    congestion_labels = ['Low Congestion', 'Moderate Congestion', 'High Congestion']
    # Create categorical variables
    df_google['congestion_level'] = pd.cut(df_google['normalized_congestion'], bins=congestion_bins, labels=congestion_labels)
    categorical_traffic_incidents = []
    for tup in df_traffic_incidents['tuples']:
        if tup != '()' :
            categorical_traffic_incidents.append(1)
        else:
            categorical_traffic_incidents.append(0)
    
    # Convert holiday dates to datetime objects
    for holiday, date in holiday_dates.items():
        holiday_dates[holiday] = pd.to_datetime(date).date()
        categorical_holiday_data = pd.to_datetime(df_google['timestamp'],errors='coerce').apply(is_holiday)

    df_ml = pd.DataFrame()
    df_ml['timestamp'] = df_google['timestamp']
    df_ml['origin_osmid'] = df_google['origin_osmid'].values
    df_ml['destination_osmid'] = df_google['destination_osmid'].values

    df_ml['weather_data'] = df_weather['weather_description']
    
    df_ml['day_of_week'] = df_google['day_of_week'].values
    df_ml['holiday_data'] =  categorical_holiday_data
    df_ml['traffic_incidents'] = categorical_traffic_incidents

    df_ml['intraffic_speed'] = df_google['intraffic_speed'].values
    df_ml['average_speed'] = df_google['average_speed'].values


    df_ml['distance_value'] = df_google['distance_value'].values
    df_ml['congestion_level']  = df_google['congestion_level']

    # Apply the function to create a new categorical variable based on hour
    df_ml['time_of_day'] = df_google['hour'].apply(categorize_hour)
    df_ml['hour'] = df_google['timestamp'].dt.hour

    return df_ml



