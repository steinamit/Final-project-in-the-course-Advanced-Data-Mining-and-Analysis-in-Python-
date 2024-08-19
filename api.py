from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and preprocessor
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    manufactor = request.form.get('manufactor')
    Year = request.form.get('Year')
    model_name = request.form.get('model')
    Hand = request.form.get('Hand')
    Gear = request.form.get('Gear')
    capacity_Engine = request.form.get('capacity_Engine')
    Engine_type = request.form.get('Engine_type')
    Prev_ownership = request.form.get('Prev_ownership')
    Curr_ownership = request.form.get('Curr_ownership')
    Area = request.form.get('Area')
    City = request.form.get('City')
    Pic_num = request.form.get('Pic_num')
    Color = request.form.get('Color')
    Km = request.form.get('Km')
    
    # Optional fields
    Cre_date = request.form.get('Cre_date')
    Repub_date = request.form.get('Repub_date')
    Description = request.form.get('Description')
    Test = request.form.get('Test')

    
    # Convert input data into a DataFrame with all required columns
    input_data = pd.DataFrame({
        'manufactor': [manufactor],
        'Year': [int(Year)],
        'model': [model_name],
        'Hand': [int(Hand)],
        'Gear': [Gear],
        'capacity_Engine': [float(capacity_Engine)],
        'Engine_type': [Engine_type],
        'Prev_ownership': [Prev_ownership],
        'Curr_ownership': [Curr_ownership],
        'Area': [Area],
        'City': [City],
        'Pic_num': [float(Pic_num)],
        'Color': [Color],
        'Km': [float(Km)],

    })

    # Preprocess the input data
    input_data_enc = preprocessor.transform(input_data)
    
    # Make prediction
    predicted_price = model.predict(input_data_enc)[0]
    
    # Output the prediction
    text_output = f"Predicted Car Price (ILS) : {predicted_price:.0f}"
    return render_template('index.html', prediction_text=text_output)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
