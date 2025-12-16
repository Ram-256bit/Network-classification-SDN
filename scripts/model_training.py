import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_models():
    """Train and evaluate different models"""
    # Load preprocessed data
    X = np.load('../models/X_preprocessed.npy')
    y = np.load('../models/y_preprocessed.npy')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train various models for comparison
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Additional metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'training_time': training_time,
            'model': model,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
    
    # Save models
    joblib.dump(results, '../models/trained_models.pkl')
    
    # Generate comparison plot
    plot_results(results)
    
    return results

def plot_results(results):
    """Plot comparison of model results"""
    plt.figure(figsize=(14, 6))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    plt.subplot(1, 2, 2)
    times = [results[model]['training_time'] for model in models]
    bars = plt.bar(models, times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../models/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    print("\n" + "="*50)
    print("DETAILED MODEL COMPARISON")
    print("="*50)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Training Time: {result['training_time']:.2f}s")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    results = train_models()
    
    # Save the best model separately
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, '../models/best_model.pkl')
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
