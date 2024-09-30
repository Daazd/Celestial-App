import requests

def get_studies_about(topic, limit=10, offset=0):
    """Fetch studies from the OSDR API based on the given topic."""
    
    # Define the base URL for the OSDR API
    base_url = "https://api.osdr.org/v1/studies"
    
    # Set up query parameters
    params = {
        'query': topic,
        'limit': limit,   # Limit the number of results
        'offset': offset   # Used for pagination
    }
    
    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        
        # Check the response status
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        # Parse the JSON response
        studies = response.json()

        # Check if there are any studies found
        if 'results' in studies and studies['results']:
            return studies['results']  # Return the list of studies found
        else:
            return []  # Return an empty list if no studies are found

    except requests.RequestException as e:
        print(f"An error occurred while fetching studies: {e}")
        return None  # Optionally, return None or an empty list depending on your preference

