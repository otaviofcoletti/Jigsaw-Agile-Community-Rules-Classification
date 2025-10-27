# Jigsaw Agile Community Rules Classification

A binary classifier that predicts whether a Reddit comment violates specific community rules. This project uses natural language processing and machine learning techniques to automatically detect rule violations in Reddit comments.

## Overview

This classifier is designed to help moderate online communities by automatically identifying comments that may violate community guidelines. The model is trained on Reddit comment data and uses TF-IDF vectorization combined with logistic regression to make predictions.

## Features

- **Binary Classification**: Predicts whether comments violate community rules (0 = clean, 1 = violation)
- **Text Preprocessing**: Comprehensive text cleaning and normalization
- **Feature Engineering**: TF-IDF vectorization with n-gram support
- **Model Evaluation**: Built-in evaluation metrics including ROC AUC, classification report, and confusion matrix
- **Cross-Validation**: K-fold cross-validation for robust model assessment
- **Feature Importance**: Extract and analyze the most important features for classification
- **Model Persistence**: Save and load trained models
- **CLI Interface**: Command-line tools for training and prediction
- **Batch Processing**: Process multiple comments at once from CSV files

## Project Structure

```
├── src/
│   ├── classifier.py      # Main classifier implementation
│   ├── data_generator.py  # Sample data generation utilities
│   ├── train.py          # Training script
│   └── predict.py        # Prediction script with CLI interface
├── tests/
│   └── test_classifier.py # Unit tests
├── data/                 # Data directory (created during training)
├── models/               # Saved models directory
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/otaviofcoletti/Jigsaw-Agile-Community-Rules-Classification.git
cd Jigsaw-Agile-Community-Rules-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Training a Model

To train a model with sample data:

```bash
cd src
python train.py
```

This will:
- Generate sample training data if none exists
- Train a binary classifier
- Evaluate the model performance
- Save the trained model to `models/reddit_classifier.joblib`

### 2. Making Predictions

#### Single Comment Prediction
```bash
cd src
python predict.py --model ../models/reddit_classifier.joblib --comment "This is a test comment to classify"
```

#### Batch Prediction from CSV
```bash
cd src
python predict.py --model ../models/reddit_classifier.joblib --input_csv input_comments.csv --output_csv predictions.csv
```

The input CSV should have a `comment` column containing the text to classify.

## Usage Examples

### Using the Classifier Programmatically

```python
from src.classifier import RedditRuleClassifier
from src.data_generator import SampleDataGenerator

# Generate sample data
generator = SampleDataGenerator()
data = generator.generate_dataset(n_samples=1000, violation_ratio=0.3)

# Split into features and labels
X = data['comment']
y = data['label']

# Initialize and train classifier
classifier = RedditRuleClassifier(max_features=5000, ngram_range=(1, 2))
classifier.fit(X, y)

# Make predictions
comments_to_classify = [
    "Thanks for sharing this helpful information!",
    "You're an idiot for posting this garbage"
]

predictions = classifier.predict(comments_to_classify)
probabilities = classifier.predict_proba(comments_to_classify)

print("Predictions:", predictions)  # [0, 1] (clean, violation)
print("Probabilities:", probabilities)
```

### Evaluating Model Performance

```python
# Evaluate on test data
metrics = classifier.evaluate(X_test, y_test)
print("Classification Report:")
print(metrics['classification_report'])
print(f"ROC AUC Score: {metrics['roc_auc_score']:.4f}")

# Cross-validation
cv_results = classifier.cross_validate(X, y, cv=5)
print(f"CV ROC AUC: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']*2:.4f})")
```

### Feature Analysis

```python
# Get most important features
feature_importance = classifier.get_feature_importance(top_n=10)
for feature, coefficient in feature_importance:
    print(f"{feature}: {coefficient:.4f}")
```

## Model Details

### Architecture
- **Vectorization**: TF-IDF with configurable max features and n-gram range
- **Classifier**: Logistic Regression with balanced class weights
- **Preprocessing**: Text cleaning, URL removal, special character handling
- **Feature Selection**: Top-k feature selection based on TF-IDF scores

### Default Parameters
- `max_features`: 10,000 (configurable)
- `ngram_range`: (1, 2) - unigrams and bigrams
- `class_weight`: 'balanced' - handles class imbalance
- `random_state`: 42 - for reproducible results

### Performance Metrics
The model reports several metrics:
- **Precision**: Fraction of predicted violations that are actual violations
- **Recall**: Fraction of actual violations that are correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve (preferred metric for binary classification)

## Data Format

### Training Data
The training data should be a CSV file with at least these columns:
- `comment`: The text content of the comment
- `label`: Binary label (0 = clean, 1 = violation)

Optional columns:
- `subreddit`: The subreddit where the comment was posted

### Sample Data Generation
If you don't have labeled data, the project includes a data generator that creates synthetic Reddit comments:

```python
from src.data_generator import SampleDataGenerator

generator = SampleDataGenerator()
dataset = generator.generate_dataset(n_samples=1000, violation_ratio=0.3)
dataset.to_csv('training_data.csv', index=False)
```

## Testing

Run the test suite:

```bash
cd tests
python -m unittest test_classifier.py -v
```

Or run all tests:

```bash
python -m unittest discover tests/ -v
```

## Model Customization

### Custom Preprocessing
You can customize text preprocessing by modifying the `preprocess_text` method:

```python
class CustomClassifier(RedditRuleClassifier):
    def preprocess_text(self, text):
        # Add custom preprocessing steps
        text = super().preprocess_text(text)
        # Your custom logic here
        return text
```

### Different Algorithms
To use a different classifier algorithm:

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RedditRuleClassifier()
# Modify the pipeline creation
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Commit your changes (`git commit -am 'Add new feature'`)
7. Push to the branch (`git push origin feature/new-feature`)
8. Create a Pull Request

## License

This project is available under the MIT License. See LICENSE file for details.

## Acknowledgments

- Built for the Jigsaw Agile Community Rules Classification challenge
- Uses scikit-learn for machine learning capabilities
- NLTK for natural language processing utilities