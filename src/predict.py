"""
Prediction script for Reddit comment rule classifier.
"""

import argparse
import pandas as pd
import os
from classifier import RedditRuleClassifier


def predict_comment(model_path, comment_text):
    """
    Predict if a single comment violates rules.
    
    Args:
        model_path (str): Path to the trained model
        comment_text (str): Comment text to classify
        
    Returns:
        dict: Prediction result with probability
    """
    # Load model
    classifier = RedditRuleClassifier()
    classifier.load_model(model_path)
    
    # Make prediction
    prediction = classifier.predict([comment_text])[0]
    probability = classifier.predict_proba([comment_text])[0]
    
    return {
        'comment': comment_text,
        'prediction': int(prediction),
        'violation_probability': float(probability[1]),
        'clean_probability': float(probability[0])
    }


def predict_batch(model_path, input_csv, output_csv):
    """
    Predict for a batch of comments from CSV file.
    
    Args:
        model_path (str): Path to the trained model
        input_csv (str): Path to input CSV file with 'comment' column
        output_csv (str): Path to save output CSV with predictions
    """
    # Load data
    data = pd.read_csv(input_csv)
    
    if 'comment' not in data.columns:
        raise ValueError("Input CSV must have a 'comment' column")
    
    # Load model
    classifier = RedditRuleClassifier()
    classifier.load_model(model_path)
    
    # Make predictions
    predictions = classifier.predict(data['comment'])
    probabilities = classifier.predict_proba(data['comment'])
    
    # Add results to dataframe
    data['prediction'] = predictions
    data['violation_probability'] = probabilities[:, 1]
    data['clean_probability'] = probabilities[:, 0]
    
    # Save results
    data.to_csv(output_csv, index=False)
    
    print(f"Predictions saved to {output_csv}")
    print(f"Total comments processed: {len(data)}")
    print(f"Predicted violations: {sum(predictions)}")
    print(f"Predicted clean: {len(predictions) - sum(predictions)}")


def main():
    """
    Main prediction function with CLI interface.
    """
    parser = argparse.ArgumentParser(description='Reddit Comment Rule Classifier')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--comment', type=str, help='Single comment to classify')
    parser.add_argument('--input_csv', type=str, help='Input CSV file with comments')
    parser.add_argument('--output_csv', type=str, help='Output CSV file for batch predictions')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    if args.comment:
        # Single comment prediction
        result = predict_comment(args.model, args.comment)
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Comment: {result['comment']}")
        print(f"Prediction: {'VIOLATION' if result['prediction'] == 1 else 'CLEAN'}")
        print(f"Violation Probability: {result['violation_probability']:.4f}")
        print(f"Clean Probability: {result['clean_probability']:.4f}")
        print("="*50)
        
    elif args.input_csv and args.output_csv:
        # Batch prediction
        predict_batch(args.model, args.input_csv, args.output_csv)
        
    else:
        print("Please provide either --comment for single prediction or both --input_csv and --output_csv for batch prediction")


if __name__ == "__main__":
    main()