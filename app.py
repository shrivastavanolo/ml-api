from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from flask_cors import CORS
# from name import scaler

app = Flask(__name__)
CORS(app)

# Load your Keras model
model = load_model('budget_allocation_model_nn.keras')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive income as input
    income = float(request.json['income'])

    # Prepare input data for prediction
    input_data = np.array([[income]])

    # Make prediction
    prediction = model.predict(input_data)

    prediction = np.abs(prediction)

    # Convert prediction to list (assuming model output is a single array)
    prediction_list = prediction.tolist()

    # Return prediction as JSON response
    return jsonify(prediction_list)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
