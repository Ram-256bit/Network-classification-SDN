import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Add the parent directory to path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and combine UNSW-NB15 dataset files"""
        print("Loading UNSW-NB15 dataset...")
        
        # Load training and testing sets using the exact filenames you have
        train_df = pd.read_csv(os.path.join(self.data_path, 'UNSW_NB15_training-set.csv'))
        test_df = pd.read_csv(os.path.join(self.data_path, 'UNSW_NB15_testing-set.csv'))
        
        print(f"Training set shape: {train_df.shape}")
        print(f"Testing set shape: {test_df.shape}")
        
        # Combine datasets
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        print(f"Combined dataset shape: {df.shape}")
        return df
    
    def preprocess_data(self, df):
        """Preprocess the dataset for traffic classification"""
        print("Preprocessing data...")
        
        # Display column information
        print(f"Original columns: {df.columns.tolist()}")
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ['id']
        for col in columns_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
        
        # Handle categorical features
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in df_processed.columns:
                print(f"Encoding categorical feature: {col}")
                df_processed[col] = df_processed[col].astype('category').cat.codes
        
        # Handle missing values
        df_processed = df_processed.fillna(0)
        
        # For this project, we'll use attack_cat as our target for traffic classification
        if 'attack_cat' in df.columns:
            y = df['attack_cat']
            y = self.label_encoder.fit_transform(y)
            print(f"Using 'attack_cat' as target with {len(np.unique(y))} classes")
        elif 'label' in df.columns:
            y = df['label']
            print(f"Using 'label' as target with {len(np.unique(y))} classes")
        else:
            raise ValueError("Could not find target column ('attack_cat' or 'label')")
        
        # Separate features from target
        X = df_processed
        feature_names = X.columns.tolist()
        
        print(f"Final feature names: {feature_names[:10]}...")  # Show first 10 features
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_names
    
    def save_preprocessed_data(self, X, y, feature_names):
        """Save preprocessed data for later use"""
        # Create models directory if it doesn't exist
        os.makedirs('../models', exist_ok=True)
        
        np.save('../models/X_preprocessed.npy', X)
        np.save('../models/y_preprocessed.npy', y)
        joblib.dump(feature_names, '../models/feature_names.pkl')
        joblib.dump(self.scaler, '../models/scaler.pkl')
        joblib.dump(self.label_encoder, '../models/label_encoder.pkl')
        print("Preprocessed data saved successfully!")
    
    def load_preprocessed_data(self):
        """Load preprocessed data"""
        X = np.load('../models/X_preprocessed.npy')
        y = np.load('../models/y_preprocessed.npy')
        feature_names = joblib.load('../models/feature_names.pkl')
        scaler = joblib.load('../models/scaler.pkl')
        label_encoder = joblib.load('../models/label_encoder.pkl')
        return X, y, feature_names, scaler, label_encoder

# Main execution
if __name__ == "__main__":
    data_path = "../UNSW-NB15"  # Path to your dataset folder
    preprocessor = DataPreprocessor(data_path)
    
    try:
        # Load and preprocess data
        df = preprocessor.load_data()
        X, y, feature_names = preprocessor.preprocess_data(df)
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(X, y, feature_names)
        
        print(f"Data shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Sample feature names: {feature_names[:5]}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
