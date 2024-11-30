from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('Sleep Quality Model.h5')

# Define the mapping dictionaries
occupation_dict = {'Engineer': 0, 'Doctor': 1, 'Journalist': 2, 'HR manager': 3, 'Web developer': 4,
                   'Security guard': 5, 'Pharmacist': 6, 'Police officer': 7, 'Nurse': 8, 
                   'Truck driver': 9, 'Plumber': 10, 'Factory worker': 11}

activity_dict = {'Low': 0, 'Normal': 1, 'High': 2}
stress_dict = {'Low': 0, 'Normal': 1, 'High': 2}
athlete_dict = {'Yes': 0, 'No': 1}

# Mapping of model output to Sleep Quality
sleep_quality_dict = {0: 'Good', 1: 'Bad'}

@app.route('/')
def home():
    return "Sleep Quality Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Expecting JSON input
    data = request.get_json()
    
    try:
        # Extract values from the JSON data
        occupation = data['occupation']
        age = int(data['age'])
        sleep_duration = float(data['sleep_duration'])
        heart_rate = float(data['heart_rate'])
        physical_activity = data['physical_activity']
        stress_level = data['stress_level']
        athlete = data['athlete']
        height_cm = float(data['height'])
        weight = float(data['weight'])
        
        # Convert height from cm to meters and calculate BMI
        height_m = height_cm / 100.0
        bmi = weight / (height_m ** 2)
        
        # Map categorical values using the dictionaries
        occupation_mapped = occupation_dict.get(occupation, -1)
        physical_activity_mapped = activity_dict.get(physical_activity, -1)
        stress_level_mapped = stress_dict.get(stress_level, -1)
        athlete_mapped = athlete_dict.get(athlete, -1)
        
        # Check if any mapping failed (invalid input)
        if -1 in [occupation_mapped, physical_activity_mapped, stress_level_mapped, athlete_mapped]:
            return jsonify({'error': 'Invalid input, please check your categories.'}), 400

        # Prepare the data for prediction
        input_data = np.array([[occupation_mapped, age, sleep_duration, heart_rate,
                                physical_activity_mapped, stress_level_mapped, athlete_mapped, bmi]])
        
        # Make prediction
        prediction = int(model.predict(input_data)[0])

        # Map the prediction to 'Good' or 'Bad'
        sleep_quality = sleep_quality_dict.get(prediction, 'Unknown')
        
        # Return the result as JSON
        return jsonify({'predicted_sleep_quality': sleep_quality})

    except KeyError as e:
        # Handle missing fields
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        # Handle general errors
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
