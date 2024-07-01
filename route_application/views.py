from django.shortcuts import render
from django.http import JsonResponse
import folium
import json
import requests
import osmnx as ox
from .genetic_algorithm import find_optimal_route
from .datasets_creation import load_my_graph
import logging
# Set up logging
logger = logging.getLogger(__name__)
G, _= load_my_graph()
GOOGLE_MAPS_API_KEY = 'AIzaSyD3UT6TRUTowvf7mLn4772x8qMyHaBEYUc'
def home(request):
    # Create a default Folium map
    m = folium.Map(location=[45.64371895092807, 25.591639697176465], zoom_start=13)

    map_html = m._repr_html_()
    map_html = map_html.replace('zoomControl: true', 'zoomControl: false')

    return render(request, 'home.html', {'map_html': map_html})

def get_bounding_box(geometry):
    """Calculate the bounding box for a list of geometries."""
    coords = [coord for geom in geometry for coord in geom.coords]
    min_lat = min(coord[1] for coord in coords)
    max_lat = max(coord[1] for coord in coords)
    min_lng = min(coord[0] for coord in coords)
    max_lng = max(coord[0] for coord in coords)
    return (min_lat, max_lat, min_lng, max_lng)


def get_coordinates(address, api_key):
    """
    Get geographical coordinates (latitude and longitude) for a given address using Google Maps Geocoding API.
    """
    geocode_url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
    response = requests.get(geocode_url)
    if response.status_code != 200:
        logger.error(f"Failed to geocode address: {address}, Status code: {response.status_code}")
        raise ValueError(f"Failed to geocode address: {address}")

    results = response.json().get('results')
    if not results:
        logger.error(f"Could not geocode address: {address}. Response: {response.json()}")
        raise ValueError(f"Could not geocode address: {address}")

    location = results[0].get('geometry').get('location')
    return location['lat'], location['lng']

def get_osm_node(G, lat, lng):
    """
    Get the nearest OSMnx node ID for given geographical coordinates.
    """
    node = ox.distance.nearest_nodes(G, lng, lat)
    return node

def get_route(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            #start = data.get('start')
            #end = data.get('end')
            start_address = data.get('start')
            print(start_address)
            end_address = data.get('end')
            training_period = data.get('training_period')
            road_type  = data.get('road_type')
            result_type=data.get('result_type')
            
            if not start_address or not end_address:
                return JsonResponse({'error': 'Start and end node IDs are required'}, status=400)

            try:
                api_key = GOOGLE_MAPS_API_KEY
                start_lat, start_lng = get_coordinates(start_address, api_key)
                end_lat, end_lng = get_coordinates(end_address, api_key)

                start_node = get_osm_node(G, start_lat, start_lng)
                end_node = get_osm_node(G, end_lat, end_lng)
            except ValueError as e:
                return JsonResponse({'error': str(e)}, status=400)

            # Call your genetic algorithm to get the route
            route = find_optimal_route(start_node, end_node, training_period, road_type,result_type)
            bbox = get_bounding_box(route['path']['geometry'])

            # Create a map centered at the midpoint of start and end
            midpoint_lat = (bbox[0] + bbox[1]) / 2
            midpoint_lng = (bbox[2] + bbox[3]) / 2
            mymap = folium.Map(location=[midpoint_lat, midpoint_lng], zoom_start=13)
            # Fit the map to the bounds of the bounding box
            mymap.fit_bounds([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])
            
            for geom in route['path']['geometry']:
                coords = [(coord[1], coord[0]) for coord in geom.coords]  # Use (latitude, longitude) order
                folium.PolyLine(locations=coords, color='#4CBB17', weight = 5).add_to(mymap)
            start_coords = (route['path']['geometry'].iloc[0].coords[0][1], route['path']['geometry'].iloc[0].coords[0][0])
            end_coords = (route['path']['geometry'].iloc[-1].coords[-1][1], route['path']['geometry'].iloc[-1].coords[-1][0])

            folium.Marker(
                location=start_coords,
                popup=f"<strong>Start:</strong> {start_coords}",
                tooltip="Start Location",
                icon=folium.Icon(icon='map-pin', color="green", prefix='fa')
            ).add_to(mymap)

            folium.Marker(
                location=end_coords,
                popup=f"<strong>End:</strong> {end_coords}",
                tooltip="End Location",
                icon=folium.Icon(color='red', icon='start')
            ).add_to(mymap)


            # Save the map as an HTML file
            map_html = mymap._repr_html_()
            map_html = map_html.replace('zoomControl: true', 'zoomControl: false')

            return JsonResponse({
                'map_html': map_html,
                'distance': route['distance'],
                'duration': route['duration'],
                'congestion_level': route['congestion_level']
            })
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            print(f"Error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)