import numpy as np
import pandas as pd
import joblib
import os
import sys

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_dir, 'models')

print(f"Project directory: {project_dir}")
print(f"Models directory: {models_dir}")

class LiveTrafficClassifier:
    def __init__(self):
        # Load the pre-trained model and preprocessing objects using absolute paths
        try:
            self.model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
            self.feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
            print("Model and preprocessing objects loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_live_data(self, df):
        """Preprocess live data to match training data format"""
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = df_processed.fillna(0)
        
        # Convert protocol to numerical if needed
        if 'proto' in df_processed.columns and df_processed['proto'].dtype == 'object':
            df_processed['proto'] = df_processed['proto'].astype('category').cat.codes
        
        # Convert service to numerical if needed
        if 'service' in df_processed.columns and df_processed['service'].dtype == 'object':
            df_processed['service'] = df_processed['service'].astype('category').cat.codes
        
        # Convert state to numerical if needed
        if 'state' in df_processed.columns and df_processed['state'].dtype == 'object':
            df_processed['state'] = df_processed['state'].astype('category').cat.codes
        
        # Add missing columns with default values
        for col in self.feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Reorder columns to match training data
        df_processed = df_processed[self.feature_names]
        
        # Remove target columns for prediction
        X = df_processed.drop(['attack_cat', 'label'], axis=1, errors='ignore')
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def classify_live_traffic(self, csv_file):
        """Classify live traffic from CSV file"""
        print(f"Classifying live traffic from {csv_file}...")
        
        # Load live data
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            print("No data to classify")
            return None, None
        
        # Preprocess the data
        X_live = self.preprocess_live_data(df)
        
        # Make predictions
        predictions = self.model.predict(X_live)
        
        # Convert numerical predictions back to original labels
        if hasattr(self.label_encoder, 'inverse_transform'):
            predicted_labels = self.label_encoder.inverse_transform(predictions)
        else:
            predicted_labels = predictions
        
        # Add predictions to dataframe
        df['prediction'] = predicted_labels
        df['prediction_numeric'] = predictions
        
        return df, None  # No accuracy calculation for live data
    
    def generate_report(self, df, output_file='live_classification_report.html'):
        """Generate a classification report"""
        html_content = f"""
        <html>
        <head>
            <title>Live Traffic Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Live Traffic Classification Report</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>Total flows classified: {len(df)}</p>
            
            <h2>Prediction Distribution</h2>
            <table>
                <tr>
                    <th>Traffic Type</th>
                    <th>Count</th>
                </tr>
        """
        
        # Add prediction distribution
        pred_counts = df['prediction'].value_counts()
        for pred, count in pred_counts.items():
            html_content += f"<tr><td>{pred}</td><td>{count}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Classification Details</h2>
            <table>
                <tr>
                    <th>Flow ID</th>
                    <th>Predicted Traffic Type</th>
                </tr>
        """
        
        # Add classification details
        for i, row in df.iterrows():
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['prediction']}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # Save report
        output_path = os.path.join(project_dir, output_file)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_path}")

def main():
    # Initialize classifier
    classifier = LiveTrafficClassifier()
    
    # Classify live traffic
    live_traffic_path = os.path.join(project_dir, 'live_traffic.csv')
    results_df, accuracy = classifier.classify_live_traffic(live_traffic_path)
    
    if results_df is not None:
        print(f"Classification completed successfully!")
        
        # Show distribution of predictions
        print("\nPrediction distribution:")
        print(results_df['prediction'].value_counts())
        
        # Generate report
        classifier.generate_report(results_df)
        
        # Save results
        results_path = os.path.join(project_dir, 'live_classification_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
