from flask import render_template, request, jsonify
import joblib
import numpy as np
from app import app

# Load the trained model from the model folder
model_path = 'd:/Projects/drought-prediction-app/model/rf_model.pkl'
rf_model = joblib.load(model_path)
print("model loded successfully")
# Home route to display the web interface
@app.route('/')
def home():
    return render_template('index.html')

# API route to handle drought prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract latitude and longitude from the user input
    lat = float(data['latitude'])
    lon = float(data['longitude'])

    # Example of additional inputs (e.g., temperature, precipitation)
    temperature = 25  # Placeholder or real-time data
    precipitation = 50  # Placeholder or real-time data

    # Prepare model input for prediction
    model_input = np.array([[lat, lon, temperature, precipitation]])

    # Predict drought
    drought_prediction = rf_model.predict(model_input)

    # Return prediction result as JSON
    return jsonify({'drought_prediction': int(drought_prediction[0])})