import requests
import re
import nltk
from textblob import TextBlob
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import urllib3
from urllib.parse import quote

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_model(file_path):
    loaded = joblib.load(file_path)
    return loaded['pipeline'], loaded['label_encoder']

nlp = spacy.load("en_core_web_sm")

def custom_tokenizer(text):
    return text.split()

def clean_query_for_search(user_input):
    """Clean user input to focus on keywords relevant for study search."""
    stop_words = ['show', 'me', 'studies', 'research', 'paper', 'on', 'about', 'the', 'for']
    words = user_input.lower().split()
    keywords = [word for word in words if word not in stop_words]
    
    return ' '.join(keywords)

def search_studies_by_query(user_input):
    """Search studies in the OSDR API based on cleaned-up keywords or study ID."""
    
    base_url = 'https://osdr.nasa.gov/osdr/data/search'
    
    # Clean up the input to get relevant keywords
    study_id = extract_study_id(user_input)
    if study_id.isdigit():  # If we extracted a number, search by study ID
        query_term = f'OSD-{study_id}'
    else:
        # If no number, use the main keywords for the search
        query_term = clean_query_for_search(user_input)
    
    params = {
        'term': query_term,
        'page': 0,
        'size': 25
    }
    print(f"Sending API request to: {base_url} with params: {params}")
    
    try:
        response = requests.get(base_url, params=params, verify=False, timeout=10)
        response.raise_for_status()
        
        studies_data = response.json()
        #print(f"Received response: {studies_data}")  # Log the response for debugging
        
        # Extract relevant studies if present
        return extract_relevant_study_files(studies_data, query_term)
    
    except requests.RequestException as e:
        print(f"An error occurred while fetching the study data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

def extract_study_id(user_input):
    """Extract numerical part from user input to use as the study ID."""
    match = re.search(r'\d+', user_input)
    return match.group(0) if match else user_input

def extract_relevant_study_files(studies_data, query_term):
    """Extract and format relevant study data from the API response."""
    if 'hits' in studies_data and 'hits' in studies_data['hits']:
        study_hits = studies_data['hits']['hits']
        
        if not study_hits:
            return f"Sorry, I couldn't find any studies related to '{query_term}'. Can you try rephrasing your query?"

        # Create a list to hold the formatted study details
        study_results = []
        
        for study in study_hits:
            study_source = study['_source']

            # Extract key details with fallback for missing data
            study_title = study_source.get('Study Title', 'No title available')
            author = study_source.get('Study Person', {}).get('First Name', '') + ' ' + study_source.get('Study Person', {}).get('Last Name', '')
            funding_agency = study_source.get('Study Funding Agency', 'Unknown funding agency')
            study_url = study_source.get('Authoritative Source URL', '')

            # If no URL, construct a fallback link using the Accession field
            if not study_url:
                study_url = f"https://osdr.nasa.gov/study/{study_source.get('Accession', '')}"

            # Add formatted result
            study_results.append(f"<li><b>{study_title}</b> by {author}. Funded by {funding_agency}. <a href='{study_url}'>View Study</a></li>")

        # Return the formatted list of studies
        return "<ul>" + ''.join(study_results) + "</ul>"
    else:
        return f"Sorry, I couldn't find any studies related to '{query_term}'. Can you try rephrasing your query?"

def display_study_results(user_input):
    """Display results or show an appropriate message if no studies are found."""
    
    study_files = search_studies_by_query(user_input)
    
    if study_files is None:
        return f"Sorry, an error occurred while searching for studies on '{user_input}'. Please try again later."
    
    if not study_files:
        return f"Sorry, I couldn't find any studies related to '{user_input}'. Can you try rephrasing your query or asking about a different topic?"
    
    # Generate the study links
    study_links = fetch_study_links(study_files)
    
    if study_links:
        return "Here are some relevant studies I found:\n" + "\n".join(study_links)
    
    return f"Sorry, I couldn't find any studies related to '{user_input}'. Can you try rephrasing your query or asking about a different topic?"

def fetch_study_links(study_files):
    """Extract and format study links from the study files data."""
    links = []

    if isinstance(study_files, list):
        for file in study_files:
            if isinstance(file, dict):  # Ensure each file is a dictionary
                source = file.get('_source', {})
                study_id = source.get('Accession', '').strip()
                study_title = source.get('Study Title', '').strip()
                study_url = f"https://osdr.nasa.gov/study/{study_id}"  # Construct the study URL

                if study_id:  # Only append if study_id is present
                    links.append(f'<a href="{study_url}" target="_blank">{study_title}</a>')

    return links

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

def categorize_celestial_body(name):
    name_lower = name.lower()
    if any(keyword in name_lower for keyword in ['star', 'sun', 'nova', 'supernova', 'dwarf']):
        return 'Star'
    elif any(keyword in name_lower for keyword in ['planet', 'mars', 'jupiter', 'saturn', 'venus', 'mercury', 'uranus', 'neptune']):
        return 'Planet'
    elif 'galaxy' in name_lower or 'milky way' in name_lower:
        return 'Galaxy'
    elif 'nebula' in name_lower:
        return 'Nebula'
    elif any(keyword in name_lower for keyword in ['asteroid', 'meteor', 'comet', 'kuiper', 'oort']):
        return 'Small Bodies'
    elif 'cluster' in name_lower:
        return 'Star Cluster'
    elif any(keyword in name_lower for keyword in ['black hole', 'pulsar', 'quasar']):
        return 'Extreme Objects'
    elif 'weather' in name_lower:
        return 'Mars Weather'
    else:
        return 'Other Phenomena'

def celestial_chatbot(pipeline, label_encoder, user_input, celestial_data):
    # Preprocess user input
    preprocessed_input = preprocess_text(user_input)

    # Check if user asks about studies or research
    if any(keyword in preprocessed_input for keyword in ['study', 'studies', 'research', 'paper', 'publication']):
        study_query = ' '.join([word for word in preprocessed_input.split() if word not in ['study', 'research', 'paper', 'publication', 'on', 'about', 'show', 'me', 'a']])
        study_files = search_studies_by_query(study_query)

        # Check if study_files is a dictionary
        if isinstance(study_files, dict):
            if study_files.get('hits', {}).get('total', 0) > 0:  # Check if any studies were found
                response = "<p>Here are some relevant studies I found:</p><ul>"
                # Fetch study links dynamically with study IDs attached
                links = fetch_study_links(study_files['hits']['hits'])  # Pass the list of hits to fetch links
                response += ''.join(f'<li>{link}</li>' for link in links)
                response += "</ul>"
                return response
            else:
                return f"<p>Sorry, I couldn't find any studies related to '{study_query}'. Can you try rephrasing your query or asking about a different topic?</p>"
        else:
            return f"<p>I'm sorry, but I encountered an error while searching for studies: {study_files}. Please try again later.</p>"

    # If no request for studies, proceed with celestial body classification
    user_input_transformed = pipeline.named_steps['tfidf'].transform([preprocessed_input])
    predicted_type_encoded = pipeline.named_steps['clf'].predict(user_input_transformed)[0]
    predicted_type = label_encoder.inverse_transform([predicted_type_encoded])[0]

    celestial_data['predicted_category'] = celestial_data['name'].apply(categorize_celestial_body)
    relevant_bodies = celestial_data[celestial_data['predicted_category'] == predicted_type]

    if not relevant_bodies.empty:
        relevant_bodies_preprocessed = relevant_bodies['description'].apply(preprocess_text)
        tfidf_matrix = pipeline.named_steps['tfidf'].transform(relevant_bodies_preprocessed)
        similarities = cosine_similarity(user_input_transformed, tfidf_matrix)
        
        most_similar_idx = similarities.argmax()
        relevant_body = relevant_bodies.iloc[most_similar_idx]
        
        response = f"<p>Based on your description, you might be talking about <strong>{relevant_body['name']}</strong>, " \
                   f"which is classified as a <strong>{predicted_type}</strong>.</p>" \
                   f"<p>Here's some information: {relevant_body['description']}</p>"
    else:
        general_info = {
            'Star': "Stars are massive, luminous spheres of plasma held together by their own gravity.",
            'Planet': "Planets are celestial bodies that orbit stars and do not produce their own light.",
            'Galaxy': "Galaxies are vast collections of stars, gas, and dust held together by gravity.",
            'Nebula': "Nebulae are clouds of gas and dust in space, often regions where new stars are born.",
            'Small Bodies': "Small bodies include asteroids, comets, and other minor planets in our solar system.",
            'Star Cluster': "Star clusters are groups of stars that formed together and are bound by gravity.",
            'Extreme Objects': "This category includes intense cosmic phenomena like black holes and supernovae.",
            'Other Phenomena': "This includes a variety of other celestial phenomena such as quasars or cosmic rays.",
        }
        general_description = general_info.get(predicted_type, "I'm sorry, I don't have enough information on that phenomenon right now.")
        response = f"<p>Based on your description, you might be referring to a <strong>{predicted_type}</strong>.</p>" \
                   f"<p>{general_description}</p>"

    return response

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide a user input as a command-line argument.")
        sys.exit(1)

    # Load the model and data
    pipeline, label_encoder = load_model('celestial_model.joblib')
    celestial_data = pd.read_csv('celestial_bodies.csv')
    
    user_input = sys.argv[1]
    response = celestial_chatbot(pipeline, label_encoder, user_input, celestial_data)
    print(response)