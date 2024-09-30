from flask import Flask, send_file, request, jsonify
import requests
import io
from urllib.parse import unquote
from chatbot import celestial_chatbot, load_model, load_and_preprocess_data

app = Flask(__name__)

# Load the model and data
pipeline, label_encoder = load_model('celestial_model.joblib')
_, _, celestial_data = load_and_preprocess_data('celestial_bodies.csv')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    response = celestial_chatbot(pipeline, label_encoder, user_input, celestial_data)
    return jsonify({'response': response})

@app.route('/api/download')
def download_proxy():
    url = unquote(request.args.get('url', ''))
    filename = unquote(request.args.get('filename', 'download'))

    if not url:
        return "No URL provided", 400

    try:
        # Download the file
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()

        # Create an in-memory file-like object
        file_stream = io.BytesIO(response.content)

        # Determine the MIME type
        content_type = response.headers.get('Content-Type', 'application/octet-stream')

        # Send the file
        return send_file(
            file_stream,
            as_attachment=True,
            download_name=filename,
            mimetype=content_type
        )

    except requests.RequestException as e:
        return f"Error downloading file: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)