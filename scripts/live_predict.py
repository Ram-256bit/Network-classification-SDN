import pandas as pd
import joblib
import json
from time import sleep
import os

class LivePredictor:
    def __init__(self):
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load all trained artifacts."""
        try:
            print("Loading model and preprocessors...")
            self.model = joblib.load('mlp_classifier.pkl')
            self.scaler = joblib.load('standard_scaler.pkl')
            self.target_encoder = joblib.load('target_encoder.pkl')
            
            # Load feature names
            with open('feature_names.json', 'r') as f:
                self.training_feature_names = json.load(f)
            
            # Load label encoders
            self.label_encoders = {}
            for col in ['proto', 'service', 'state']:
                self.label_encoders[col] = joblib.load(f'label_encoder_{col}.pkl')
                
            print("All artifacts loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Error: {e}. Please run train_model.py first.")
            exit(1)
    
    def preprocess_live_data(self, live_df):
        """Preprocess live data to match training format."""
        # Clean the live data
        live_df_clean = live_df.drop(columns=['id'], errors='ignore')
        live_df_clean = live_df_clean.fillna(0)
        
        # Ensure all training features are present
        missing_features = set(self.training_feature_names) - set(live_df_clean.columns)
        if missing_features:
            print(f"Adding missing features: {list(missing_features)}")
            for feature in missing_features:
                live_df_clean[feature] = 0
        
        # Reorder columns to match training data EXACTLY
        live_df_final = live_df_clean[self.training_feature_names]
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            # Convert to string and handle unseen labels
            live_df_final[col] = live_df_final[col].astype(str)
            # Map unseen labels to a default value (first class)
            live_df_final[col] = live_df_final[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            live_df_final[col] = encoder.transform(live_df_final[col])
        
        # Scale features
        X_live_scaled = self.scaler.transform(live_df_final)
        
        return X_live_scaled
    
    def predict_live_traffic(self, csv_path='../live_traffic.csv'):
        """Load and predict on live traffic data."""
        try:
            if not os.path.exists(csv_path):
                print(f"Waiting for live data file: {csv_path}")
                return None
            
            live_df = pd.read_csv(csv_path)
            if len(live_df) == 0:
                print("No new flows to classify.")
                return None
            
            print(f"Processing {len(live_df)} new flows...")
            
            # Preprocess and predict
            X_processed = self.preprocess_live_data(live_df)
            predictions_encoded = self.model.predict(X_processed)
            predictions = self.target_encoder.inverse_transform(predictions_encoded)
            
            # Count predictions
            from collections import Counter
            prediction_counts = Counter(predictions)
            
            return prediction_counts, len(live_df)
            
        except Exception as e:
            print(f"Error processing live data: {e}")
            return None
    
    def run_predictions(self):
        """Main prediction loop."""
        print("Starting live traffic classification...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                result = self.predict_live_traffic()
                if result:
                    prediction_counts, total_flows = result
                    
                    print("\n" + "="*50)
                    print("LIVE TRAFFIC CLASSIFICATION RESULTS")
                    print("="*50)
                    print(f"Total flows classified: {total_flows}")
                    print("Prediction distribution:")
                    for attack_type, count in prediction_counts.items():
                        print(f"  {attack_type}: {count}")
                    print("="*50)
                
                sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\nStopped live classification.")

# Run the predictor
if __name__ == '__main__':
    predictor = LivePredictor()
    predictor.run_predictions()
