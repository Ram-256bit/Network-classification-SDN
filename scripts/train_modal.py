import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import json

class UNWB15Trainer:
    def __init__(self, data_path='./unsw-nb15/'):
        self.data_path = data_path
        self.train_df = None
        self.test_df = None
        self.combined_df = None
        
    def load_data(self):
        """Load the official training and testing sets and combine them."""
        print("Loading UNSW-NB15 dataset...")
        # Load training and testing sets
        self.train_df = pd.read_csv(os.path.join(self.data_path, 'UNSW_NB15_training-set.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_path, 'UNSW_NB15_testing-set.csv'))
        
        # Combine them for consistent preprocessing (we will split later)
        self.combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        print(f"Training set: {self.train_df.shape}, Testing set: {self.test_df.shape}, Combined: {self.combined_df.shape}")
        
    def preprocess_data(self):
        """Preprocess the combined dataset."""
        print("Preprocessing data...")
        df = self.combined_df.copy()
        
        # Data Cleaning
        df = df.drop(columns=['id'], errors='ignore')
        df = df.fillna(0)
        
        # Define features and target
        X = df.drop(columns=['label', 'attack_cat'])
        y = df['attack_cat']
        
        # Save the feature names for live prediction
        self.feature_names = list(X.columns)
        with open('feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        print(f"Saved {len(self.feature_names)} feature names.")
        
        # Encode categorical features
        print("Encoding categorical features...")
        categorical_cols = ['proto', 'service', 'state']
        self.label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Use astype(str) to handle any mixed types
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            joblib.dump(le, f'label_encoder_{col}.pkl')
            print(f"Saved encoder for {col} with {len(le.classes_)} classes.")
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        joblib.dump(self.target_encoder, 'target_encoder.pkl')
        print(f"Target classes: {list(self.target_encoder.classes_)}")
        
        # Feature Scaling
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, 'standard_scaler.pkl')
        
        # Now split back into train and test using the original indices
        train_size = len(self.train_df)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y_encoded[:train_size], y_encoded[train_size:]
        
        return X_train, X_test, y_train, y_test
        
    def train_model(self, X_train, y_train):
        """Train the MLP model."""
        print("Training MLP model...")
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            max_iter=500, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, 'mlp_classifier.pkl')
        print("Model training completed and saved.")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on the test set."""
        accuracy = self.model.score(X_test, y_test)
        print(f"\nModel Test Accuracy: {accuracy:.4f}")
        
        # More detailed evaluation
        from sklearn.metrics import classification_report
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.target_encoder.classes_))
        
    def run_training(self):
        """Main method to run the full training pipeline."""
        self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        print("\nTraining pipeline completed successfully!")

# Run the training
if __name__ == '__main__':
    trainer = UNWB15Trainer(data_path='./unsw-nb15/')  # Adjust path if needed
    trainer.run_training()
