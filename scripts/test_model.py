#!/usr/bin/env python3
import numpy as np
import pandas as pd
import joblib
import os

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_dir, 'models')

print("=== TESTING MODEL CAPABILITIES ===")

try:
    # Load model components
    model = joblib.load(os.path.join(models_dir, 'best_model_correct.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler_input_features.pkl'))
    label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'input_features.pkl'))
    
    print(f"✓ Model loaded successfully!")
    print(f"Model classes: {label_encoder.classes_}")
    print(f"Number of features: {len(feature_names)}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Create test data for different classes
print("\n=== TESTING WITH ARTIFICIAL DATA ===")
test_samples = []

# Create distinctive samples for each class
for class_idx, class_name in enumerate(label_encoder.classes_):
    sample = {}
    
    # Make each class distinctive with different feature patterns
    if class_name == 'Normal':
        # Normal traffic characteristics - moderate values
        sample['dur'] = np.random.uniform(1, 10)
        sample['spkts'] = np.random.randint(5, 15)
        sample['dpkts'] = np.random.randint(3, 12)
        sample['sbytes'] = np.random.randint(500, 1500)
        sample['dbytes'] = np.random.randint(300, 1200)
        
    elif class_name == 'Reconnaissance':
        # Reconnaissance characteristics - short, probing traffic
        sample['dur'] = np.random.uniform(0.1, 2)
        sample['spkts'] = np.random.randint(1, 5)
        sample['dpkts'] = np.random.randint(1, 3)
        sample['sbytes'] = np.random.randint(100, 400)
        sample['dbytes'] = np.random.randint(50, 300)
        
    elif 'DoS' in class_name or 'DDoS' in class_name:
        # DoS characteristics - high volume, short duration
        sample['dur'] = np.random.uniform(0.1, 1)
        sample['spkts'] = np.random.randint(50, 200)
        sample['dpkts'] = np.random.randint(40, 150)
        sample['sbytes'] = np.random.randint(2000, 8000)
        sample['dbytes'] = np.random.randint(1500, 6000)
        
    else:
        # Other attack types - varied patterns
        sample['dur'] = np.random.uniform(0.5, 8)
        sample['spkts'] = np.random.randint(10, 40)
        sample['dpkts'] = np.random.randint(8, 30)
        sample['sbytes'] = np.random.randint(800, 3000)
        sample['dbytes'] = np.random.randint(600, 2500)
    
    # Fill remaining features with reasonable values
    for feat in feature_names:
        if feat not in sample:
            if feat in ['proto', 'service', 'state']:
                sample[feat] = np.random.randint(0, 5)
            elif 'ttl' in feat:
                sample[feat] = np.random.randint(50, 200)
            elif 'load' in feat:
                sample[feat] = np.random.uniform(1, 100)
            elif 'rate' in feat:
                sample[feat] = np.random.uniform(0.1, 50)
            else:
                sample[feat] = np.random.uniform(0, 10)
    
    test_samples.append(sample)

# Convert to DataFrame and scale
test_df = pd.DataFrame(test_samples)
test_scaled = scaler.transform(test_df)

# Make predictions
predictions = model.predict(test_scaled)
probabilities = model.predict_proba(test_scaled)

print("\n=== TEST RESULTS ===")
print("True Class -> Predicted Class (Confidence)")
print("-" * 50)

correct = 0
for i, (true_class, pred, probs) in enumerate(zip(label_encoder.classes_, predictions, probabilities)):
    pred_class = label_encoder.inverse_transform([pred])[0]
    confidence = max(probs)
    
    status = "✓" if true_class == pred_class else "✗"
    if true_class == pred_class:
        correct += 1
    
    print(f"{status} {true_class} -> {pred_class} ({confidence:.3f})")
    
    if true_class != pred_class:
        print(f"   Probabilities: {[f'{p:.3f}' for p in probs]}")

accuracy = correct / len(label_encoder.classes_) * 100
print(f"\n=== SUMMARY ===")
print(f"Accuracy on test data: {accuracy:.1f}%")
print(f"Unique predictions: {len(np.unique(predictions))} out of {len(label_encoder.classes_)} classes")
print(f"Can distinguish classes: {len(np.unique(predictions)) > 1}")

if len(np.unique(predictions)) == 1:
    print("\n⚠️  WARNING: Model is predicting the same class for all samples!")
    print("This suggests either:")
    print("1. Model was trained on imbalanced data")
    print("2. Model is overfitting to one class")
    print("3. Training data quality issues")
