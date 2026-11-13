import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def train_and_evaluate():
    print("Loading dataset...")
    dataset_path = os.path.join("dataset", "scada_synthetic_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run generate_data.py first.")
    
    df = pd.read_csv(dataset_path)
    
    # Separate features and target
    feature_names = [
        'pressure', 
        'flow_rate', 
        'temperature', 
        'pump_vibration', 
        'pump_current', 
        'ph', 
        'tank_level'
    ]
    
    X = df[feature_names]
    y = df['fault_type']
    
    print(f"Dataset shape: {df.shape}")
    
    # Split into 70% training (7,000 rows) and 30% testing (3,000 rows)
    # Stratify ensures each class maintains its exact 70/30 ratio across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    
    # Build a pipeline combining scaling and Random Forest classifier
    print("Building Random Forest pipeline...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Predict on unseen test data
    print("Evaluating model on test data...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\n--- Model Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Save model pipeline
    model_path = os.path.join("models", "sentinelflow_model.joblib")
    
    # Package model along with essential metadata for FastAPI integration later
    model_artifact = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "classes": list(pipeline.classes_),
        "version": "1.0.0"
    }
    
    joblib.dump(model_artifact, model_path)
    print(f"\nModel successfully saved to: {model_path}")
    
    # Save evaluation metrics to JSON
    metrics_data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": class_report
    }
    metrics_path = os.path.join("results", "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    # Generate and save Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.title("SentinelFlow - Confusion Matrix")
    plt.xlabel("Predicted Fault Type")
    plt.ylabel("Actual Fault Type")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join("results", "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")
    
    # Generate and save Feature Importance plot
    rf_model = pipeline.named_steps['classifier']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances for SCADA Fault Classification")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    fi_path = os.path.join("results", "feature_importance.png")
    plt.savefig(fi_path)
    plt.close()
    print(f"Feature importance plot saved to: {fi_path}")

if __name__ == "__main__":
    train_and_evaluate()