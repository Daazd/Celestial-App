import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from trainer import load_model, load_and_preprocess_data, TextPreprocessor

def load_model(file_path):
    loaded = joblib.load(file_path)
    return loaded['model'], loaded['label_encoder']

def celestial_chatbot(pipeline, label_encoder, user_input, celestial_data):
    # Process the user input through the pipeline
    user_input_transformed = pipeline.transform([user_input])

    # Make the prediction
    predicted_type_encoded = pipeline.named_steps['rf'].predict(user_input_transformed)[0]
    predicted_type = label_encoder.inverse_transform([predicted_type_encoded])[0]

    relevant_bodies = celestial_data[celestial_data['category'] == predicted_type]
    
    if not relevant_bodies.empty:
        # Use the pipeline's TfidfVectorizer for consistency
        tfidf = pipeline.named_steps['tfidf']
        tfidf_matrix = tfidf.transform(relevant_bodies['description'])
        user_tfidf = tfidf.transform([user_input])
        
        similarities = cosine_similarity(user_tfidf, tfidf_matrix)
        
        most_similar_idx = similarities.argmax()
        relevant_body = relevant_bodies.iloc[most_similar_idx]
        
        response = f"Based on your description, you might be talking about {relevant_body['name']}, which is classified as a {predicted_type}. Here's some information: {relevant_body['description']}"
    else:
        response = f"I've classified your query as related to {predicted_type}, but I don't have specific information about a celestial body matching your description. Here's some general information: "
        response += {
            'Star': "Stars are massive, luminous spheres of plasma held together by their own gravity.",
            'Planet': "Planets are celestial bodies that orbit stars and do not produce their own light.",
            'Galaxy': "Galaxies are vast collections of stars, gas, and dust held together by gravity.",
            'Nebula': "Nebulae are clouds of gas and dust in space, often regions where new stars are born.",
            'Small Bodies': "Small bodies include asteroids, comets, and other minor planets in our solar system.",
            'Mars Weather': "This category provides information about the weather conditions on Mars.",
            'Other': "This category includes various celestial phenomena that don't fit into the main categories."
        }.get(predicted_type, "I'm not sure what celestial body you're describing. Could you provide more details?")

    return response

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide a user input as a command-line argument.")
        sys.exit(1)

    pipeline, label_encoder = load_model('celestial_model.joblib')
    _, _, celestial_data = load_and_preprocess_data('celestial_bodies.csv')
    
    user_input = sys.argv[1]
    response = celestial_chatbot(pipeline, label_encoder, user_input, celestial_data)
    print(response)