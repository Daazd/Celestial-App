import requests
import json

def find_stargazing_locations(lat, lon, radius=10000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["natural"="peak"]({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1});
      way["natural"="peak"]({lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1});
    );
    out center;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    locations = []
    for element in data['elements']:
        if element['type'] == 'node':
            locations.append({
                'name': element.get('tags', {}).get('name', 'Unknown Peak'),
                'lat': element['lat'],
                'lon': element['lon']
            })
        elif element['type'] == 'way':
            locations.append({
                'name': element.get('tags', {}).get('name', 'Unknown Peak'),
                'lat': element['center']['lat'],
                'lon': element['center']['lon']
            })
    
    return locations