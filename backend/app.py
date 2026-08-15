import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and preprocessor
model = joblib.load('best_random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        # Convert json to DataFrame, ensuring all expected columns are present
        data = pd.DataFrame(json_)

        # Ensure 'Product_Id' is handled correctly or dropped if not needed for preprocessing
        # Based on previous notebook, 'Product_Id' is dropped before preprocessing, and 'Product_Id_char' is derived.
        # The incoming data should either not have 'Product_Id' or it should be handled to derive 'Product_Id_char'
        # Assuming `Product_Id` is passed and `Product_Id_char` needs to be created, or `Product_Id_char` is passed directly.
        # For simplicity, let's assume raw data (before feature engineering) is sent and recreate Product_Id_char and Store_Age

        # Recreate Product_Id_char
        if 'Product_Id' in data.columns and 'Product_Id_char' not in data.columns:
            data['Product_Id_char'] = data['Product_Id'].str[:2]

        # Recreate Store_Age
        if 'Store_Establishment_Year' in data.columns and 'Store_Age' not in data.columns:
            data['Store_Age'] = 2026 - data['Store_Establishment_Year']

        # Ordinal encoding for Store_Size (if not already encoded in the incoming data)
        if 'Store_Size' in data.columns and data['Store_Size'].dtype == 'object':
            size_map = {'Small': 0, 'Medium': 1, 'High': 2}
            data['Store_Size'] = data['Store_Size'].map(size_map)

        # Drop original Product_Id and Store_Establishment_Year if they exist after deriving new features
        if 'Product_Id' in data.columns:
            data = data.drop('Product_Id', axis=1)
        if 'Store_Establishment_Year' in data.columns:
            data = data.drop('Store_Establishment_Year', axis=1)

        # Preprocess the data
        processed_data = preprocessor.transform(data)

        # Convert to DataFrame to ensure column names are preserved for prediction (if model expects them)
        # Note: preprocessor.transform outputs a numpy array. If your model needs named columns, you might need to map them.
        # For RandomForest, it generally works with array as long as feature order is consistent.
        # Let's assume the preprocessor correctly aligns features.

        # Make prediction
        prediction_log = model.predict(processed_data)

        # Inverse transform the log prediction
        prediction = np.exp(prediction_log)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/predictbatch', methods=['POST'])
def predict_batch():
    try:
        # Check that a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No CSV file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read CSV
        data = pd.read_csv(file)

        print("Received batch data:")
        print(data.head())
        print("Shape:", data.shape)

        # Remove target column if it exists
        if 'Product_Store_Sales_Total' in data.columns:
            data = data.drop('Product_Store_Sales_Total', axis=1)

        # Recreate Product_Id_char
        if 'Product_Id' in data.columns and 'Product_Id_char' not in data.columns:
            data['Product_Id_char'] = data['Product_Id'].str[:2]

        # Recreate Store_Age
        if 'Store_Establishment_Year' in data.columns and 'Store_Age' not in data.columns:
            data['Store_Age'] = 2026 - data['Store_Establishment_Year']

        # Encode Store_Size
        if 'Store_Size' in data.columns and data['Store_Size'].dtype == 'object':
            size_map = {
                'Small': 0,
                'Medium': 1,
                'High': 2
            }
            data['Store_Size'] = data['Store_Size'].map(size_map)

        # Drop columns not used by the model
        if 'Product_Id' in data.columns:
            data = data.drop('Product_Id', axis=1)

        if 'Store_Establishment_Year' in data.columns:
            data = data.drop('Store_Establishment_Year', axis=1)

        # Preprocess
        processed_data = preprocessor.transform(data)

        # Predict
        prediction_log = model.predict(processed_data)

        # Reverse log transformation
        prediction = np.exp(prediction_log)

        return jsonify({
            'prediction': prediction.tolist()
        })

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            'error': str(e)
        }), 400
        
if __name__ == '__main__':
    # Use 0.0.0.0 to make the server accessible from outside the container
    app.run(host='0.0.0.0', port=7860)
