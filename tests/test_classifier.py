"""
Tests for the Reddit rule classifier.
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import RedditRuleClassifier
from data_generator import SampleDataGenerator


class TestRedditRuleClassifier(unittest.TestCase):
    """Test cases for RedditRuleClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = RedditRuleClassifier(max_features=100, ngram_range=(1, 1))
        
        # Create sample data
        self.generator = SampleDataGenerator()
        self.sample_data = self.generator.generate_dataset(n_samples=50, violation_ratio=0.4)
        
        self.X_sample = self.sample_data['comment']
        self.y_sample = self.sample_data['label']
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(self.classifier.max_features, 100)
        self.assertEqual(self.classifier.ngram_range, (1, 1))
        self.assertFalse(self.classifier.is_fitted)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "Hello World! This is a TEST with URLs: https://example.com"
        processed = self.classifier.preprocess_text(text)
        expected = "hello world this is a test with urls"
        self.assertEqual(processed, expected)
        
        # Test with None/NaN
        self.assertEqual(self.classifier.preprocess_text(None), "")
        self.assertEqual(self.classifier.preprocess_text(pd.NA), "")
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = self.classifier.create_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[0][0], 'tfidf')
        self.assertEqual(pipeline.steps[1][0], 'classifier')
    
    def test_fit_predict(self):
        """Test model fitting and prediction."""
        # Fit the model
        self.classifier.fit(self.X_sample, self.y_sample)
        self.assertTrue(self.classifier.is_fitted)
        
        # Make predictions
        predictions = self.classifier.predict(self.X_sample)
        self.assertEqual(len(predictions), len(self.X_sample))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Test probabilities
        probabilities = self.classifier.predict_proba(self.X_sample)
        self.assertEqual(probabilities.shape, (len(self.X_sample), 2))
        self.assertTrue(all(0 <= prob <= 1 for row in probabilities for prob in row))
    
    def test_predict_without_fitting(self):
        """Test that prediction fails without fitting."""
        with self.assertRaises(ValueError):
            self.classifier.predict(["test comment"])
        
        with self.assertRaises(ValueError):
            self.classifier.predict_proba(["test comment"])
    
    def test_evaluation(self):
        """Test model evaluation."""
        # Fit the model
        self.classifier.fit(self.X_sample, self.y_sample)
        
        # Evaluate
        metrics = self.classifier.evaluate(self.X_sample, self.y_sample)
        
        self.assertIn('classification_report', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('roc_auc_score', metrics)
        
        # ROC AUC should be between 0 and 1
        self.assertTrue(0 <= metrics['roc_auc_score'] <= 1)
    
    def test_cross_validation(self):
        """Test cross-validation."""
        cv_results = self.classifier.cross_validate(self.X_sample, self.y_sample, cv=3)
        
        self.assertIn('mean_score', cv_results)
        self.assertIn('std_score', cv_results)
        self.assertIn('scores', cv_results)
        
        # Check that we have the right number of CV scores
        self.assertEqual(len(cv_results['scores']), 3)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Fit the model first
        self.classifier.fit(self.X_sample, self.y_sample)
        
        # Get feature importance
        importance = self.classifier.get_feature_importance(top_n=5)
        
        self.assertIsInstance(importance, list)
        self.assertLessEqual(len(importance), 5)
        
        # Each item should be a tuple of (feature_name, coefficient)
        for item in importance:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)  # feature name
            self.assertIsInstance(item[1], (int, float))  # coefficient
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        # Fit the model
        self.classifier.fit(self.X_sample, self.y_sample)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            self.classifier.save_model(temp_path)
            
            # Create new classifier and load model
            new_classifier = RedditRuleClassifier()
            new_classifier.load_model(temp_path)
            
            # Test that loaded model works
            self.assertTrue(new_classifier.is_fitted)
            predictions = new_classifier.predict(self.X_sample)
            self.assertEqual(len(predictions), len(self.X_sample))
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSampleDataGenerator(unittest.TestCase):
    """Test cases for SampleDataGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SampleDataGenerator(random_seed=42)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertIsInstance(self.generator.violation_patterns, list)
        self.assertIsInstance(self.generator.clean_patterns, list)
        self.assertTrue(len(self.generator.violation_patterns) > 0)
        self.assertTrue(len(self.generator.clean_patterns) > 0)
    
    def test_generate_comment(self):
        """Test comment generation."""
        # Generate violating comment
        violation_comment = self.generator.generate_comment(is_violation=True)
        self.assertIsInstance(violation_comment, str)
        self.assertTrue(len(violation_comment) > 0)
        
        # Generate clean comment
        clean_comment = self.generator.generate_comment(is_violation=False)
        self.assertIsInstance(clean_comment, str)
        self.assertTrue(len(clean_comment) > 0)
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        dataset = self.generator.generate_dataset(n_samples=100, violation_ratio=0.3)
        
        # Check dataset structure
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertEqual(len(dataset), 100)
        self.assertIn('comment', dataset.columns)
        self.assertIn('label', dataset.columns)
        self.assertIn('subreddit', dataset.columns)
        
        # Check label distribution
        violation_count = sum(dataset['label'])
        expected_violations = int(100 * 0.3)
        self.assertEqual(violation_count, expected_violations)
        
        # Check that all labels are 0 or 1
        self.assertTrue(all(label in [0, 1] for label in dataset['label']))


if __name__ == '__main__':
    unittest.main()