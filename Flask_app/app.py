from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os
import traceback
import pandas as pd

# Note: scikit-learn is installed as 'scikit-learn' in requirements.txt
# but imported as 'sklearn' in Python code - this is the standard usage
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load the model
MODEL_PATH = 'NN2.pkl'

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found")

# Try to load the model at startup
try:
    model = load_model()
    print("Model loaded successfully")
    
    # Load the CSV to get the output feature names
    if os.path.exists('merged_6326.csv'):
        df = pd.read_csv('merged_6326.csv')
        # Inputs are: Gain, UGBW, Phase DB, Area, Slew Rate, Power Dissipation
        input_features = ['gain', 'ugbw', 'phase_db', 'area', 'slew_rate', 'power_dis']
        # All other columns are outputs
        output_features = [col for col in df.columns if col not in input_features]
    else:
        # Fallback if CSV not found
        output_features = ['W1', 'L1', 'W3', 'L3', 'W5', 'L5', 'W6', 'L6', 'W7', 'L7', 'IB', 'CC']
        
    print(f"Output features: {output_features}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    output_features = ['W1', 'L1', 'W3', 'L3', 'W5', 'L5', 'W6', 'L6', 'W7', 'L7', 'IB', 'CC']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if request.is_json:
            # Handle JSON request
            data = request.json
            gain = float(data.get('gain', 0))
            ugbw = float(data.get('ugbw', 0))
            phase_db = float(data.get('phase_db', 0))
            area = float(data.get('area', 0))
            slew_rate = float(data.get('slew_rate', 0))
            power_dis = float(data.get('power_dis', 0))
        else:
            # Handle form data
            gain = float(request.form.get('gain', 0))
            ugbw = float(request.form.get('ugbw', 0))
            phase_db = float(request.form.get('phase_db', 0))
            area = float(request.form.get('area', 0))
            slew_rate = float(request.form.get('slew_rate', 0))
            power_dis = float(request.form.get('power_dis', 0))
        
        # Input for the model includes all features
        input_array = np.array([[gain, ugbw, phase_db, area, slew_rate, power_dis]])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        # Format the prediction result
        result = {
            'parameters': dict(zip(output_features, prediction.tolist() if isinstance(prediction, np.ndarray) else [prediction])),
            'input': {
                'gain': gain,
                'ugbw': ugbw,
                'phase_db': phase_db,
                'area': area,
                'slew_rate': slew_rate,
                'power_dis': power_dis
            }
        }
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

