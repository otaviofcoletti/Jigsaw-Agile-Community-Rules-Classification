"""
Training script for Reddit comment rule classifier.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import RedditRuleClassifier
from data_generator import create_sample_datasets
import os


def load_data(data_path):
    """
    Load training data from CSV file.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Generating sample data...")
        create_sample_datasets()
    
    return pd.read_csv(data_path)


def train_model(data_path, model_save_path, test_size=0.2):
    """
    Train the Reddit rule classifier.
    
    Args:
        data_path (str): Path to training data CSV
        model_save_path (str): Path to save the trained model
        test_size (float): Fraction of data to use for testing
        
    Returns:
        RedditRuleClassifier: Trained classifier
        dict: Evaluation metrics
    """
    # Load data
    print("Loading data...")
    data = load_data(data_path)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Label distribution:\n{data['label'].value_counts()}")
    
    # Split data
    X = data['comment']
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Initialize and train classifier
    print("\nInitializing classifier...")
    classifier = RedditRuleClassifier(max_features=5000, ngram_range=(1, 2))
    
    print("Training classifier...")
    classifier.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = classifier.evaluate(X_test, y_test)
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print(f"\nROC AUC Score: {metrics['roc_auc_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = classifier.cross_validate(X_train, y_train, cv=5)
    print(f"CV ROC AUC: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']*2:.4f})")
    
    # Feature importance
    print("\nTop important features:")
    feature_importance = classifier.get_feature_importance(top_n=10)
    for feature, coef in feature_importance:
        print(f"{feature}: {coef:.4f}")
    
    # Save model
    print(f"\nSaving model to {model_save_path}")
    classifier.save_model(model_save_path)
    
    return classifier, metrics


def main():
    """
    Main training function.
    """
    # Set up paths
    base_path = "/home/runner/work/Jigsaw-Agile-Community-Rules-Classification/Jigsaw-Agile-Community-Rules-Classification"
    data_path = os.path.join(base_path, "data", "train_data.csv")
    model_path = os.path.join(base_path, "models", "reddit_classifier.joblib")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Train model
    classifier, metrics = train_model(data_path, model_path)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()