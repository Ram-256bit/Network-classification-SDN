import numpy as np
import pandas as pd
import joblib
import os
import sys
import time

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_dir, 'models')

print(f"Project directory: {project_dir}")
print(f"Models directory: {models_dir}")

class LiveTrafficClassifier:
    def __init__(self):
        # Load the pre-trained model and preprocessing objects using absolute paths
        try:
            self.model = joblib.load(os.path.join(models_dir, 'best_model_correct.pkl'))
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler_input_features.pkl'))
            self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
            self.feature_names = joblib.load(os.path.join(models_dir, 'input_features.pkl'))
            print("Model and preprocessing objects loaded successfully!")
            print(f"Using {len(self.feature_names)} features for prediction")
            print(f"Model classes: {self.label_encoder.classes_}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_live_data(self, df):
        """Preprocess live data to match training data format"""
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # DEBUG: Show what columns we have
        print(f"Original columns in live data: {list(df_processed.columns)}")
        print(f"Expected features: {self.feature_names}")
        
        # Remove the label column if it exists (it's the target, not a feature)
        ground_truth_labels = None
        if 'label' in df_processed.columns:
            # Store labels separately but remove from features
            ground_truth_labels = df_processed['label'].copy()
            df_processed = df_processed.drop(columns=['label'])
            print(f"Removed 'label' column from features")
        
        # Fill ALL NaN values with 0 first
        df_processed = df_processed.fillna(0)
        
        # Handle missing features - add them with default value 0
        for col in self.feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
                print(f"Added missing feature: {col}")
        
        # Handle categorical columns (proto, service, state) - convert to numeric
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                # Convert categorical strings to numerical codes
                df_processed[col] = df_processed[col].astype('category').cat.codes
                print(f"Converted categorical column {col} to numeric")
            elif col in df_processed.columns:
                # Already numeric, just ensure it's int
                df_processed[col] = df_processed[col].astype(int)
        
        # Reorder columns to match training data order
        df_processed = df_processed[self.feature_names]
        
        # DEBUG: Check final data shape
        print(f"Final processed data shape: {df_processed.shape}")
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        
        return X_scaled, ground_truth_labels
    
    def classify_live_traffic(self, csv_file):
        """Classify live traffic from CSV file with confidence scores"""
        print(f"Classifying live traffic from {csv_file}...")
        
        # Load live data
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} flows from live data")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None, None, None
        
        if len(df) == 0:
            print("No data to classify")
            return None, None, None
        
        # Preprocess the data
        X_live, ground_truth_labels = self.preprocess_live_data(df)
        
        # Make predictions
        predictions = self.model.predict(X_live)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_live)
        
        # Convert numerical predictions back to original labels
        if hasattr(self.label_encoder, 'inverse_transform'):
            predicted_labels = self.label_encoder.inverse_transform(predictions)
        else:
            predicted_labels = predictions
        
        # Add predictions to dataframe
        df['prediction'] = predicted_labels
        df['prediction_numeric'] = predictions
        
        # Calculate accuracy if we have ground truth labels
        accuracy = None
        if ground_truth_labels is not None:
            accuracy = np.mean(ground_truth_labels == predictions)
            print(f"Accuracy: {accuracy:.4f}")
        
        return df, accuracy, probabilities
    
    def generate_detailed_report(self, df, probabilities, output_file='live_classification_detailed.html'):
        """Generate a detailed classification report with confidence scores"""
        html_content = f"""
        <html>
        <head>
            <title>Detailed Traffic Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .high-confidence {{ color: green; font-weight: bold; }}
                .medium-confidence {{ color: orange; }}
                .low-confidence {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Detailed Traffic Classification Report</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>Total flows classified: {len(df)}</p>
        """
        
        # Add prediction distribution
        pred_counts = df['prediction'].value_counts()
        html_content += """
            <h2>Prediction Distribution</h2>
            <table>
                <tr>
                    <th>Traffic Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        for pred, count in pred_counts.items():
            percentage = (count / len(df)) * 100
            html_content += f"<tr><td>{pred}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Detailed Predictions with Confidence</h2>
            <table>
                <tr>
                    <th>Flow ID</th>
                    <th>Actual Label</th>
                    <th>Predicted Label</th>
                    <th>Top 3 Predictions</th>
                    <th>Confidence</th>
                </tr>
        """
        
        # Add detailed predictions with confidence
        for i, row in df.iterrows():
            if i < len(probabilities):
                probs = probabilities[i]
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3_predictions = []
                
                for idx in top3_idx:
                    label_name = self.label_encoder.inverse_transform([idx])[0]
                    confidence = probs[idx]
                    confidence_class = "high-confidence" if confidence > 0.7 else "medium-confidence" if confidence > 0.3 else "low-confidence"
                    top3_predictions.append(f'<span class="{confidence_class}">{label_name}: {confidence:.3f}</span>')
                
                top3_html = "<br>".join(top3_predictions)
                
                actual_label = row.get('label', 'N/A')
                if actual_label != 'N/A' and hasattr(self.label_encoder, 'inverse_transform'):
                    try:
                        actual_label = self.label_encoder.inverse_transform([int(actual_label)])[0]
                    except:
                        pass
                
                html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{actual_label}</td>
                    <td>{row['prediction']}</td>
                    <td>{top3_html}</td>
                    <td>{max(probs):.3f}</td>
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
        
        print(f"Detailed report saved to {output_path}")
        
        return output_path

def main():
    # Initialize classifier
    classifier = LiveTrafficClassifier()
    
    # Classify live traffic
    live_traffic_path = os.path.join(project_dir, 'live_traffic.csv')
    results_df, accuracy, probabilities = classifier.classify_live_traffic(live_traffic_path)
    
    if results_df is not None:
        print(f"Classification completed. Accuracy: {accuracy:.4f}")
        
        # Show distribution of predictions
        print("\nPrediction distribution:")
        pred_counts = results_df['prediction'].value_counts()
        print(pred_counts)
        
        # Show confidence scores for first 5 flows
        print("\nConfidence scores for first 5 flows:")
        for i, probs in enumerate(probabilities[:5]):
            top3_idx = np.argsort(probs)[-3:][::-1]
            print(f"Flow {i}:")
            for idx in top3_idx:
                label_name = classifier.label_encoder.inverse_transform([idx])[0]
                print(f"  {label_name}: {probs[idx]:.3f}")
        
        # Generate detailed report
        classifier.generate_detailed_report(results_df, probabilities)
        
        # Save results
        results_path = os.path.join(project_dir, 'live_classification_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
