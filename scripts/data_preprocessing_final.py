import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir, 'UNSW-NB15')
models_dir = os.path.join(project_dir, 'models')

print(f"Project directory: {project_dir}")
print(f"Data directory: {data_dir}")
print(f"Models directory: {models_dir}")

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and combine UNSW-NB15 dataset files"""
        print("Loading UNSW-NB15 dataset...")
        
        # Load training and testing sets using the exact filenames
        train_path = os.path.join(self.data_path, 'UNSW_NB15_training-set.csv')
        test_path = os.path.join(self.data_path, 'UNSW_NB15_testing-set.csv')
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
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
        
        # Check for any remaining non-numeric columns and handle them
        numeric_errors = []
        for col in df_processed.columns:
            # Try to convert to numeric, coerce errors to NaN
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            
            # Check if there were any conversion errors
            if df_processed[col].isna().any():
                error_count = df_processed[col].isna().sum()
                numeric_errors.append((col, error_count))
                print(f"Warning: Column '{col}' had {error_count} non-numeric values converted to NaN")
        
        # Handle missing values
        df_processed = df_processed.fillna(0)
        
        # For this project, we'll use attack_cat as our target for traffic classification
        if 'attack_cat' in df.columns:
            y = df['attack_cat']
            y = self.label_encoder.fit_transform(y)
            print(f"Using 'attack_cat' as target with {len(np.unique(y))} classes")
            print(f"Target classes: {self.label_encoder.classes_}")
        elif 'label' in df.columns:
            y = df['label']
            print(f"Using 'label' as target with {len(np.unique(y))} classes")
        else:
            raise ValueError("Could not find target column ('attack_cat' or 'label')")
        
        # Separate features from target
        X = df_processed
        feature_names = X.columns.tolist()
        
        print(f"Final feature names: {feature_names[:10]}...")  # Show first 10 features
        
        # Convert to numpy arrays and ensure they are float32
        X = X.values.astype(np.float32)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_names
    
    def save_preprocessed_data(self, X, y, feature_names):
        """Save preprocessed data for later use"""
        np.save(os.path.join(models_dir, 'X_preprocessed.npy'), X)
        np.save(os.path.join(models_dir, 'y_preprocessed.npy'), y)
        joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
        joblib.dump(self.scaler, os.path.join(models_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))
        print("Preprocessed data saved successfully!")
    
    def load_preprocessed_data(self):
        """Load preprocessed data"""
        X = np.load(os.path.join(models_dir, 'X_preprocessed.npy'))
        y = np.load(os.path.join(models_dir, 'y_preprocessed.npy'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        return X, y, feature_names, scaler, label_encoder

# Main execution
if __name__ == "__main__":
    preprocessor = DataPreprocessor(data_dir)
    
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
