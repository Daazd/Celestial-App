import requests

def get_studies_about(topic, limit=10, offset=0):
    """Fetch studies from the OSDR API based on the given topic."""
    
    base_url = "https://api.osdr.org/v1/studies"
    
    params = {
        'query': topic,
        'limit': limit,   
        'offset': offset   
    }
    
    try:
        response = requests.get(base_url, params=params)
        
        response.raise_for_status() 
        
        studies = response.json()

        if 'results' in studies and studies['results']:
            return studies['results'] 
        else:
            return []  

    except requests.RequestException as e:
        print(f"An error occurred while fetching studies: {e}")
        return None 

