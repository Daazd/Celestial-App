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
import random

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
    
    study_id = extract_study_id(user_input)
    if study_id.isdigit():  
        query_term = f'OSD-{study_id}'
    else:
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
        study_results = []
        
        for study in study_hits:
            study_source = study['_source']
            study_title = study_source.get('Study Title', 'No title available')
            author = study_source.get('Study Person', {}).get('First Name', '') + ' ' + study_source.get('Study Person', {}).get('Last Name', '')
            funding_agency = study_source.get('Study Funding Agency', 'Unknown funding agency')
            study_url = study_source.get('Authoritative Source URL', '')
            if not study_url:
                study_url = f"https://osdr.nasa.gov/study/{study_source.get('Accession', '')}"

            study_results.append(f"<li><b>{study_title}</b> by {author}. Funded by {funding_agency}. <a href='{study_url}'>View Study</a></li>")
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
    
    study_links = fetch_study_links(study_files)
    
    if study_links:
        return "Here are some relevant studies I found:\n" + "\n".join(study_links)
    
    return f"Sorry, I couldn't find any studies related to '{user_input}'. Can you try rephrasing your query or asking about a different topic?"

def fetch_study_links(study_files):
    """Extract and format study links from the study files data."""
    links = []

    if isinstance(study_files, list):
        for file in study_files:
            if isinstance(file, dict):  
                source = file.get('_source', {})
                study_id = source.get('Accession', '').strip()
                study_title = source.get('Study Title', '').strip()
                study_url = f"https://osdr.nasa.gov/study/{study_id}" 

                if study_id:  
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

class ConversationalCelestialChatbot:
    def __init__(self, pipeline, label_encoder, celestial_data):
        self.pipeline = pipeline
        self.label_encoder = label_encoder
        self.celestial_data = celestial_data
        self.study_keywords = ['study', 'studies', 'research', 'paper', 'publication']
        self.context = []
        self.greetings = ["Hello, stargazer!", "Greetings, cosmic explorer!", "Welcome to the celestial realm!"]
        self.follow_ups = [
            "Is there anything specific you'd like to know about {}?",
            "What fascinates you most about {}?",
            "Did you know that {}? What else would you like to learn about it?"
        ]

    def respond(self, user_input):
        if self.is_greeting(user_input):
            return random.choice(self.greetings) + " What celestial wonder would you like to explore today?"

        if self.is_study_request(user_input):
            response = self.handle_study_query(user_input)
        else:
            response = self.get_celestial_response(user_input)

        follow_up = self.get_follow_up(response)
        self.context.append(user_input)

        return response + "\n\n" + follow_up
    
    def is_study_request(self, text):
        return any(keyword in text.lower() for keyword in self.study_keywords)

    def is_greeting(self, text):
        greetings = ["hello", "hi", "hey", "greetings"]
        return any(word in text.lower() for word in greetings)

    def get_celestial_response(self, user_input):
        preprocessed_input = preprocess_text(user_input)
        
        # Check for study-related queries
        if any(keyword in preprocessed_input for keyword in ['study', 'studies', 'research', 'paper', 'publication']):
            return self.handle_study_query(preprocessed_input)

        # Create a DataFrame with the user input and placeholder values for other features
        user_input_df = pd.DataFrame({
            'text': [preprocessed_input],
            'text_length': [len(preprocessed_input)],
            'keyword_count': [0],  # Placeholder
            'has_image': [False],  # Placeholder
            'magnitude': [0],  # Placeholder
            'distance': [0],  # Placeholder
            'color_index': [0]  # Placeholder
        })
        preprocessor = self.pipeline.named_steps['preprocessor']
        user_input_transformed = preprocessor.transform(user_input_df)
        
        clf = self.pipeline.named_steps['clf']
        predicted_type_encoded = clf.predict(user_input_transformed)[0]
        predicted_type = self.label_encoder.inverse_transform([predicted_type_encoded])[0]

        self.celestial_data['predicted_category'] = self.celestial_data['name'].apply(categorize_celestial_body)
        relevant_bodies = self.celestial_data[self.celestial_data['predicted_category'] == predicted_type]

        if not relevant_bodies.empty:
            return self.get_specific_body_response(relevant_bodies, user_input_transformed, preprocessor)
        else:
            return self.get_general_info_response(predicted_type)

    def handle_study_query(self, user_input):
        preprocessed_input = preprocess_text(user_input)
        study_query = ' '.join([word for word in preprocessed_input.split() if word not in self.study_keywords + ['on', 'about', 'show', 'me', 'a', 'the', 'for']])
        study_files = search_studies_by_query(study_query)
        
        if isinstance(study_files, str):
            return study_files
        elif isinstance(study_files, dict) and study_files.get('hits', {}).get('total', 0) > 0:
            response = "<p>Exciting news! I've found some stellar studies for you:</p><ul>"
            links = fetch_study_links(study_files['hits']['hits'])  
            response += ''.join(f'<li>{link}</li>' for link in links)
            response += "</ul>"
        else:
            response = f"<p>I've searched the cosmic archives, but couldn't find any studies related to '{study_query}'. Perhaps we could explore a different aspect of this topic?</p>"
        return response

    def get_specific_body_response(self, relevant_bodies, user_input_transformed, preprocessor):
        relevant_bodies_preprocessed = relevant_bodies['description'].apply(preprocess_text)
        relevant_bodies_df = pd.DataFrame({
            'text': relevant_bodies_preprocessed,
            'text_length': relevant_bodies_preprocessed.apply(len),
            'keyword_count': [0] * len(relevant_bodies),  # Placeholder
            'has_image': [False] * len(relevant_bodies),  # Placeholder
            'magnitude': relevant_bodies.get('magnitude', [0] * len(relevant_bodies)),
            'distance': relevant_bodies.get('distance', [0] * len(relevant_bodies)),
            'color_index': relevant_bodies.get('color_index', [0] * len(relevant_bodies))
        })
        
        relevant_bodies_transformed = preprocessor.transform(relevant_bodies_df)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_input_transformed, relevant_bodies_transformed)
        
        most_similar_idx = similarities.argmax()
        relevant_body = relevant_bodies.iloc[most_similar_idx]
        
        return f"<p>Ah, based on your cosmic curiosity, it seems you're exploring <strong>{relevant_body['name']}</strong>, " \
               f"a fascinating <strong>{relevant_body['predicted_category']}</strong>.</p>" \
               f"<p>Let me share some celestial wisdom: {relevant_body['description']}</p>"

    def get_general_info_response(self, predicted_type):
        general_info = {
            'Star': "Stars are the cosmic beacons of the universe, massive spheres of plasma that light up the night sky.",
            'Planet': "Planets are the wanderers of the cosmos, orbiting stars and each telling a unique story.",
            'Galaxy': "Galaxies are vast cosmic neighborhoods, home to billions of stars, planets, and mysteries.",
            'Nebula': "Nebulae are cosmic nurseries and graveyards, where stars are born and where they leave their final mark.",
            'Small Bodies': "Small bodies are the celestial wildcards, from asteroids to comets, each with a tale to tell.",
            'Star Cluster': "Star clusters are cosmic families, groups of stars born together and journeying through space as one.",
            'Extreme Objects': "Extreme objects are the universe's most enigmatic phenomena, pushing the boundaries of physics.",
            'Other Phenomena': "The cosmos is full of wonders, some defying categorization but all worthy of exploration.",
        }
        general_description = general_info.get(predicted_type, "This cosmic phenomenon is shrouded in mystery, waiting for further exploration.")
        return f"<p>Your query has led us to the realm of <strong>{predicted_type}</strong>.</p>" \
               f"<p>{general_description}</p>"

    def get_follow_up(self, main_response):
        topic = self.extract_topic(main_response)
        return random.choice(self.follow_ups).format(topic)

    def extract_topic(self, text):
        import re
        match = re.search(r'<strong>(.*?)</strong>', text)
        return match.group(1) if match else "this celestial topic"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide a user input as a command-line argument.")
        sys.exit(1)

    # Load the model and data
    pipeline, label_encoder = load_model('celestial_model.joblib')
    celestial_data = pd.read_csv('celestial_bodies.csv')
    
    user_input = sys.argv[1]
    chatbot = ConversationalCelestialChatbot(pipeline, label_encoder, celestial_data)
    response = chatbot.respond(user_input)
    print(response)
    
    # chatbot = ConversationalCelestialChatbot(pipeline, label_encoder, celestial_data)
    
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == 'quit':
    #         break
        
    #     try:
    #         response = chatbot.respond(user_input)
    #         print("Chatbot:", response)
    #     except Exception as e:
    #         print(f"An error occurred: {e}")