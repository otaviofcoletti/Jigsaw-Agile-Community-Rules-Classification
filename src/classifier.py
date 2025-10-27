"""
Binary classifier for Reddit comment rule violation detection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class RedditRuleClassifier:
    """
    Binary classifier to predict whether a Reddit comment violates community rules.
    """
    
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        """
        Initialize the classifier.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            ngram_range (tuple): Range of n-grams to use
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.pipeline = None
        self.is_fitted = False
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """
        Preprocess text for classification.
        
        Args:
            text (str): Raw comment text
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def create_pipeline(self):
        """
        Create the machine learning pipeline.
        
        Returns:
            Pipeline: sklearn pipeline with TF-IDF and Logistic Regression
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                preprocessor=self.preprocess_text
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
        
        return pipeline
    
    def fit(self, X, y):
        """
        Train the classifier.
        
        Args:
            X (array-like): Comment texts
            y (array-like): Binary labels (0=no violation, 1=violation)
        """
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Predict rule violations.
        
        Args:
            X (array-like): Comment texts
            
        Returns:
            array: Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Predict violation probabilities.
        
        Args:
            X (array-like): Comment texts
            
        Returns:
            array: Probability scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test (array-like): Test comment texts
            y_test (array-like): True test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'roc_auc_score': roc_auc_score(y_test, probabilities)
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X (array-like): Comment texts
            y (array-like): Binary labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        if self.pipeline is None:
            self.pipeline = self.create_pipeline()
        
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='roc_auc')
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        joblib.dump(self.pipeline, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.pipeline = joblib.load(filepath)
        self.is_fitted = True
    
    def get_feature_importance(self, top_n=20):
        """
        Get the most important features for classification.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            list: Top features with their coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # Get feature names and coefficients
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        coefficients = self.pipeline.named_steps['classifier'].coef_[0]
        
        # Create feature importance list
        feature_importance = list(zip(feature_names, coefficients))
        
        # Sort by absolute coefficient value
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return feature_importance[:top_n]