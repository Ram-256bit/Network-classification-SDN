from flask import Flask, render_template_string, request, jsonify
import numpy as np
import joblib
import pandas as pd
from io import StringIO
import os
import sys

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_dir, 'models')

app = Flask(__name__)

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SDN Traffic Classification</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-section { background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .results-section { margin-top: 30px; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SDN Traffic Classification System</h1>
        
        <div class="upload-section">
            <h3>Upload Traffic Data</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" accept=".csv">
                <button type="submit">Classify Traffic</button>
            </form>
        </div>
        
        <div class="results-section">
            <h3>Results</h3>
            <div id="results"></div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('results').innerHTML = 
                        '<div class="success"><h4>Classification Results</h4>' +
                        '<p>Total flows classified: ' + data.total_flows + '</p>' +
                        '<p>Predictions: ' + JSON.stringify(data.predictions) + '</p></div>';
                } else {
                    document.getElementById('results').innerHTML = 
                        '<div class="error">Error: ' + data.error + '</div>';
                }
            } catch (error) {
                document.getElementById('results').innerHTML = 
                    '<div class="error">Error: ' + error.message + '</div>';
            }
        });
    </script>
</body>
</html>
"""

class TrafficClassifier:
    def __init__(self):
        try:
            self.model = joblib.load(os.path.join(models_dir, 'best_model_correct.pkl'))
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler_input_features.pkl'))
            self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
            self.feature_names = joblib.load(os.path.join(models_dir, 'input_features.pkl'))
            print("Classifier loaded successfully!")
        except Exception as e:
            print(f"Error loading classifier: {e}")
            raise

    def preprocess_data(self, df):
        df_processed = df.copy()
        df_processed = df_processed.fillna(0)
        
        # Handle categorical columns (proto, service, state)
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                # Convert categorical strings to numerical codes
                df_processed[col] = df_processed[col].astype('category').cat.codes
        
        # Add missing features with default values
        for col in self.feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Reorder columns to match training data order
        df_processed = df_processed[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        return X_scaled

# Initialize classifier
try:
    classifier = TrafficClassifier()
except Exception as e:
    print(f"Failed to initialize classifier: {e}")
    classifier = None

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if classifier is None:
        return jsonify({'error': 'Classifier not initialized'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        
        X_new = classifier.preprocess_data(df)
        predictions = classifier.model.predict(X_new)
        predicted_labels = classifier.label_encoder.inverse_transform(predictions)
        
        # Count predictions
        pred_counts = pd.Series(predicted_labels).value_counts().to_dict()
        
        return jsonify({
            'success': True,
            'total_flows': len(df),
            'predictions': pred_counts
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
