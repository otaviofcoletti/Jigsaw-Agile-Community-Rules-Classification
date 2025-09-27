#!/usr/bin/env python3
"""
Example usage of the Reddit Rule Classifier.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from classifier import RedditRuleClassifier
from data_generator import SampleDataGenerator


def main():
    """
    Demonstrate the classifier functionality.
    """
    print("=" * 60)
    print("Reddit Comment Rule Classifier - Example Usage")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    generator = SampleDataGenerator()
    data = generator.generate_dataset(n_samples=200, violation_ratio=0.3)
    
    print(f"Generated {len(data)} comments")
    print(f"Violations: {sum(data['label'])}")
    print(f"Clean comments: {len(data) - sum(data['label'])}")
    
    # Prepare training data
    X = data['comment']
    y = data['label']
    
    # Initialize and train classifier
    print("\n2. Training classifier...")
    classifier = RedditRuleClassifier(max_features=1000, ngram_range=(1, 2))
    classifier.fit(X, y)
    print("Training completed!")
    
    # Example predictions
    print("\n3. Making predictions on example comments...")
    
    test_comments = [
        "Thanks for sharing this helpful information!",
        "Great discussion, I learned something new today.",
        "You're completely wrong about this topic.",
        "This is the stupidest thing I've ever read.",
        "I disagree with your point, but I respect your opinion.",
        "What a waste of time, this post is garbage.",
        "Could you provide more details about your argument?",
        "Anyone who believes this is an idiot."
    ]
    
    predictions = classifier.predict(test_comments)
    probabilities = classifier.predict_proba(test_comments)
    
    print("\nPrediction Results:")
    print("-" * 60)
    
    for i, comment in enumerate(test_comments):
        pred = predictions[i]
        prob_violation = probabilities[i][1]
        status = "VIOLATION" if pred == 1 else "CLEAN"
        
        print(f"\nComment: {comment}")
        print(f"Prediction: {status} (confidence: {prob_violation:.3f})")
    
    # Model evaluation
    print("\n4. Model evaluation...")
    metrics = classifier.evaluate(X, y)
    print(f"ROC AUC Score: {metrics['roc_auc_score']:.4f}")
    
    # Cross-validation
    cv_results = classifier.cross_validate(X, y, cv=3)
    print(f"Cross-validation ROC AUC: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']*2:.4f})")
    
    # Feature importance
    print("\n5. Top features for classification:")
    feature_importance = classifier.get_feature_importance(top_n=10)
    
    print("\nMost important features:")
    for feature, coef in feature_importance[:5]:
        direction = "violation" if coef > 0 else "clean"
        print(f"  '{feature}': {coef:.4f} (indicates {direction})")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()