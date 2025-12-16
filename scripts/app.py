from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import threading
import time
import os
import sys

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, template_folder='../web/templates')

# Global variables for live capture
live_capture_active = False
capture_thread = None

# Load preprocessed data and models
try:
    X = np.load('../models/X_preprocessed.npy')
    y = np.load('../models/y_preprocessed.npy')
    feature_names = joblib.load('../models/feature_names.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    label_encoder = joblib.load('../models/label_encoder.pkl')
    models = joblib.load('../models/trained_models.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    models = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read the uploaded file
        content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        
        # Preprocess the data
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        df = df.fillna(0)
        
        # Ensure the dataframe has the same columns as training data
        for col in feature_names:
            if col not in df.columns and col != 'label' and col != 'attack_cat':
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Scale features
        X_new = scaler.transform(df.values)
        
        # Make predictions with all models
        predictions = {}
        for model_name, model_info in models.items():
            if hasattr(model_info['model'], 'predict'):
                pred = model_info['model'].predict(X_new)
                predictions[model_name] = label_encoder.inverse_transform(pred).tolist()
        
        # Get class distribution
        class_counts = {}
        for model_name, preds in predictions.items():
            unique, counts = np.unique(preds, return_counts=True)
            class_counts[model_name] = dict(zip(unique, counts))
        
        # Generate visualization
        plt.figure(figsize=(10, 6))
        for i, (model_name, counts) in enumerate(class_counts.items()):
            plt.bar(np.arange(len(counts)) + i * 0.2, list(counts.values()), 
                   width=0.2, label=model_name)
        
        plt.xlabel('Traffic Classes')
        plt.ylabel('Count')
        plt.title('Traffic Classification Results')
        plt.legend()
        plt.xticks(np.arange(len(counts)) + 0.2, list(counts.keys()), rotation=45)
        
        # Save plot to bytes
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight')
        img_bytes.seek(0)
        plot_url = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'class_counts': class_counts,
            'plot_url': f'data:image/png;base64,{plot_url}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/stats')
def get_stats():
    """Get dataset statistics"""
    try:
        # Calculate basic statistics
        stats = {
            'total_samples': len(X),
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y)),
            'class_distribution': dict(zip(*np.unique(label_encoder.inverse_transform(y), return_counts=True))),
            'model_performance': {name: {
                'accuracy': info['accuracy'],
                'training_time': info['training_time']
            } for name, info in models.items()}
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
