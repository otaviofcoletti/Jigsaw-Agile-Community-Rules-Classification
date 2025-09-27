"""
Sample data generation for Reddit comment rule classification.
"""

import pandas as pd
import numpy as np
import random


class SampleDataGenerator:
    """
    Generate sample Reddit comment data for development and testing.
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize the data generator.
        
        Args:
            random_seed (int): Random seed for reproducibility
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Sample patterns for rule-violating comments
        self.violation_patterns = [
            "You're so stupid",
            "This is garbage content",
            "What an idiot",
            "This person is dumb",
            "Complete waste of time",
            "You have no idea what you're talking about",
            "This is the worst post ever",
            "Are you serious? This is terrible",
            "Anyone who believes this is an idiot",
            "This community is full of morons"
        ]
        
        # Sample patterns for non-violating comments
        self.clean_patterns = [
            "Thanks for sharing this information",
            "I disagree with your point, but respect your opinion",
            "Great discussion, learned something new",
            "This is helpful, appreciate the insight",
            "Interesting perspective on this topic",
            "Could you provide more details about this?",
            "I have a different view, here's why",
            "Thanks for the thoughtful response",
            "This adds valuable context to the discussion",
            "Well-reasoned argument, thanks for sharing"
        ]
        
        # Additional context words to make comments more realistic
        self.context_words = [
            "actually", "really", "honestly", "basically", "definitely",
            "probably", "apparently", "obviously", "clearly", "seriously",
            "however", "therefore", "meanwhile", "furthermore", "nevertheless"
        ]
        
        # Subreddit names for context
        self.subreddits = [
            "technology", "politics", "gaming", "movies", "books",
            "science", "worldnews", "askreddit", "showerthoughts", "todayilearned"
        ]
    
    def generate_comment(self, is_violation=False):
        """
        Generate a single comment.
        
        Args:
            is_violation (bool): Whether to generate a rule-violating comment
            
        Returns:
            str: Generated comment text
        """
        if is_violation:
            base_pattern = random.choice(self.violation_patterns)
        else:
            base_pattern = random.choice(self.clean_patterns)
        
        # Add some context words to make it more realistic
        if random.random() > 0.3:  # 70% chance to add context
            context = random.choice(self.context_words)
            if random.random() > 0.5:
                comment = f"{context}, {base_pattern.lower()}"
            else:
                comment = f"{base_pattern} {context}"
        else:
            comment = base_pattern
        
        # Occasionally add more text to make it longer
        if random.random() > 0.7:  # 30% chance
            additional_text = [
                "in my opinion",
                "based on my experience",
                "if you ask me",
                "from what I've seen",
                "to be honest"
            ]
            comment += f" {random.choice(additional_text)}"
        
        return comment
    
    def generate_dataset(self, n_samples=1000, violation_ratio=0.3):
        """
        Generate a dataset of comments with labels.
        
        Args:
            n_samples (int): Total number of samples to generate
            violation_ratio (float): Proportion of violating comments
            
        Returns:
            pd.DataFrame: Dataset with 'comment' and 'label' columns
        """
        n_violations = int(n_samples * violation_ratio)
        n_clean = n_samples - n_violations
        
        # Generate violating comments
        violating_comments = [self.generate_comment(is_violation=True) for _ in range(n_violations)]
        
        # Generate clean comments
        clean_comments = [self.generate_comment(is_violation=False) for _ in range(n_clean)]
        
        # Create labels
        violation_labels = [1] * n_violations
        clean_labels = [0] * n_clean
        
        # Combine and shuffle
        all_comments = violating_comments + clean_comments
        all_labels = violation_labels + clean_labels
        
        # Create DataFrame
        data = pd.DataFrame({
            'comment': all_comments,
            'label': all_labels,
            'subreddit': [random.choice(self.subreddits) for _ in range(n_samples)]
        })
        
        # Shuffle the dataset
        data = data.sample(frac=1).reset_index(drop=True)
        
        return data
    
    def save_dataset(self, filepath, n_samples=1000, violation_ratio=0.3):
        """
        Generate and save a dataset to CSV.
        
        Args:
            filepath (str): Path to save the CSV file
            n_samples (int): Total number of samples to generate
            violation_ratio (float): Proportion of violating comments
        """
        dataset = self.generate_dataset(n_samples, violation_ratio)
        dataset.to_csv(filepath, index=False)
        
        print(f"Dataset saved to {filepath}")
        print(f"Total samples: {len(dataset)}")
        print(f"Violating comments: {sum(dataset['label'])}")
        print(f"Clean comments: {len(dataset) - sum(dataset['label'])}")
        
        return dataset


def create_sample_datasets():
    """
    Create sample training and development datasets.
    """
    generator = SampleDataGenerator()
    
    # Create training dataset
    train_data = generator.save_dataset(
        "/home/runner/work/Jigsaw-Agile-Community-Rules-Classification/Jigsaw-Agile-Community-Rules-Classification/data/train_data.csv",
        n_samples=800,
        violation_ratio=0.3
    )
    
    # Create development dataset
    dev_data = generator.save_dataset(
        "/home/runner/work/Jigsaw-Agile-Community-Rules-Classification/Jigsaw-Agile-Community-Rules-Classification/data/dev_data.csv",
        n_samples=200,
        violation_ratio=0.3
    )
    
    return train_data, dev_data


if __name__ == "__main__":
    create_sample_datasets()