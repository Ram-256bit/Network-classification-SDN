#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
import os

def preprocess_unsw_to_model_format(input_csv, output_csv):
    """Convert UNSW-NB15 format to model-compatible format"""
    
    # Load the expected feature names from your model
    models_dir = 'models'
    try:
        expected_features = joblib.load(os.path.join(models_dir, 'input_features.pkl'))
        print(f"✓ Model expects {len(expected_features)} features")
    except:
        print("✗ Could not load feature names from models/input_features.pkl")
        return None
    
    # Load the live data
    try:
        df = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df)} rows from {input_csv}")
    except:
        print(f"✗ Could not load {input_csv}")
        return None
    
    print(f"Raw data has {df.shape[1]} columns")
    
    # Remove columns that shouldn't be features
    columns_to_remove = ['id', 'attack_cat']  # These are not features
    if 'label' in df.columns:
        columns_to_remove.append('label')  # Label is target, not feature
    
    df_processed = df.drop(columns=columns_to_remove, errors='ignore')
    print(f"After removing metadata: {df_processed.shape[1]} columns")
    
    # Check if we have the right features
    missing_features = set(expected_features) - set(df_processed.columns)
    extra_features = set(df_processed.columns) - set(expected_features)
    
    if missing_features:
        print(f"Adding {len(missing_features)} missing features with default 0")
        for feat in missing_features:
            df_processed[feat] = 0
    
    if extra_features:
        print(f"Removing {len(extra_features)} extra features")
        df_processed = df_processed.drop(columns=list(extra_features))
    
    # Reorder columns to match model expectation
    df_processed = df_processed[expected_features]
    
    # Fill NaN values
    df_processed = df_processed.fillna(0)
    
    # Save processed data
    df_processed.to_csv(output_csv, index=False)
    print(f"✓ Processed data saved to {output_csv} with shape {df_processed.shape}")
    
    return df_processed

if __name__ == "__main__":
    # Preprocess the live traffic data
    preprocess_unsw_to_model_format('live_traffic.csv', 'live_traffic_processed.csv')
    
    # Test if it matches model expectations
    print("\n=== VERIFICATION ===")
    try:
        expected_features = joblib.load('models/input_features.pkl')
        processed_df = pd.read_csv('live_traffic_processed.csv', nrows=1)
        
        print(f"Expected features: {len(expected_features)}")
        print(f"Processed features: {len(processed_df.columns)}")
        print(f"Match: {len(expected_features) == len(processed_df.columns)}")
        
        if list(processed_df.columns) == expected_features:
            print("✓ Column names and order match perfectly!")
        else:
            print("⚠️  Column order doesn't match, but count is correct")
            
    except Exception as e:
        print(f"Verification error: {e}")
