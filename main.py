import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = 'RF_model_new.pkl'
loaded_model = joblib.load(MODEL_PATH)

def predict_price(
    number_of_rooms,
    difficulty_level,
    pickup_address_count,
    dropoff_address_count,
    total_distance_km
):
    input_data = pd.DataFrame({
        'number_of_rooms': [number_of_rooms],
        'difficulty_level': [difficulty_level],
        'pickup_address_count': [pickup_address_count],
        'dropoff_address_count': [dropoff_address_count],
        'total_distance_km': [total_distance_km]
    })

    prediction = loaded_model.predict(input_data)[0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    if request.method == 'POST':
        try:
            number_of_rooms = float(request.form['number_of_rooms'])
            difficulty_level = float(request.form['difficulty_level'])
            pickup_address_count = int(request.form['pickup_address_count'])
            dropoff_address_count = int(request.form['dropoff_address_count'])
            total_distance_km = float(request.form['total_distance_km'])

            predicted_price = predict_price(
                number_of_rooms,
                difficulty_level,
                pickup_address_count,
                dropoff_address_count,
                total_distance_km
            )
            
            predicted_price = round(predicted_price, 2)

        except Exception as e:
            predicted_price = f"Error: {str(e)}"

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7872)