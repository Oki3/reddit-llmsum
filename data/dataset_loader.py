import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, DatasetDict, load_dataset
import re
from urllib.parse import urlparse


class WebISTLDRDatasetLoader:
    """
    Dataset loader for Webis-TLDR-17 dataset containing Reddit posts and TL;DR summaries.
    Supports both local files and Hugging Face datasets.
    """
    
    def __init__(self, data_dir: str = "data/webis-tldr-17", use_hf_dataset: bool = True):
        self.data_dir = data_dir
        self.use_hf_dataset = use_hf_dataset
        self.raw_data = None
        self.processed_data = None
        
    def load_from_huggingface(self, sample_size: int = None) -> List[Dict]:
        """
        Load dataset directly from Hugging Face.
        
        Args:
            sample_size: Number of samples to load (None for all)
            
        Returns:
            List of dictionaries containing post data
        """
        print("Loading dataset from Hugging Face...")
        
        # Load the dataset
        dataset = load_dataset("webis/tldr-17", split="train", trust_remote_code=True)
        
        if sample_size is not None:
            print(f"Sampling {sample_size} examples from {len(dataset)} total")
            # Shuffle and sample
            dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
        
        # Convert to list of dictionaries
        data = []
        for item in dataset:
            # Map HF dataset fields to our expected format
            record = {
                'id': item['id'],
                'title': '',  # HF dataset doesn't have separate title
                'content': item['content'],
                'summary': item['summary'],
                'subreddit': item['subreddit'],
                'score': 0,  # HF dataset doesn't have score
                'author': item['author']
            }
            data.append(record)
        
        self.raw_data = data
        print(f"Loaded {len(data)} records from Hugging Face")
        return data
        
    def download_dataset(self):
        """
        Instructions for downloading the dataset.
        """
        if self.use_hf_dataset:
            print("Dataset will be loaded automatically from Hugging Face.")
            print("No manual download required!")
        else:
            print("Please download the Webis-TLDR-17 dataset manually from:")
            print("https://webis.de/data/webis-tldr-17.html")
            print(f"Extract to: {self.data_dir}")
        
    def load_raw_data(self, file_path: str = None, sample_size: int = None) -> List[Dict]:
        """
        Load raw dataset from JSON file or Hugging Face.
        
        Args:
            file_path: Path to the dataset file (for local loading)
            sample_size: Number of samples to load from HF (None for all)
            
        Returns:
            List of dictionaries containing post data
        """
        if self.use_hf_dataset:
            return self.load_from_huggingface(sample_size=sample_size)
        
        # Original local file loading code
        if file_path is None:
            file_path = os.path.join(self.data_dir, "corpus-webis-tldr-17.json")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Handle different JSON formats
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                # Try loading as a complete JSON array first
                f.seek(0)
                data = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try JSON Lines format
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]
            
        self.raw_data = data
        return data
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""
            
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '[USER]', text)  # Replace usernames
        text = re.sub(r'/r/\w+', '[SUBREDDIT]', text)  # Replace subreddit mentions
        text = re.sub(r'\[deleted\]', '', text)
        text = re.sub(r'\[removed\]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def preprocess_data(self, min_content_length: int = 50, max_content_length: int = 2000,
                       min_summary_length: int = 10, max_summary_length: int = 200) -> pd.DataFrame:
        """
        Preprocess the raw data for training/evaluation.
        
        Args:
            min_content_length: Minimum length for post content
            max_content_length: Maximum length for post content
            min_summary_length: Minimum length for summary
            max_summary_length: Maximum length for summary
            
        Returns:
            Preprocessed DataFrame
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw_data() first.")
            
        processed_records = []
        
        for record in self.raw_data:
            # Extract relevant fields
            post_id = record.get('id', '')
            title = record.get('title', '')
            content = record.get('content', '')
            summary = record.get('summary', '')
            subreddit = record.get('subreddit', '')
            score = record.get('score', 0)
            
            # Clean text
            title = self.clean_text(title)
            content = self.clean_text(content)
            summary = self.clean_text(summary)
            
            # Combine title and content
            full_content = f"{title}. {content}" if title and content else (title or content)
            full_content = self.clean_text(full_content)
            
            # Apply filters
            if (len(full_content) < min_content_length or 
                len(full_content) > max_content_length or
                len(summary) < min_summary_length or 
                len(summary) > max_summary_length):
                continue
                
            processed_records.append({
                'id': post_id,
                'title': title,
                'content': content,
                'full_content': full_content,
                'summary': summary,
                'subreddit': subreddit,
                'score': score,
                'content_length': len(full_content),
                'summary_length': len(summary)
            })
        
        self.processed_data = pd.DataFrame(processed_records)
        return self.processed_data
    
    def create_huggingface_dataset(self, test_size: float = 0.2, 
                                 val_size: float = 0.1) -> DatasetDict:
        """
        Create a HuggingFace Dataset object for training.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            
        Returns:
            DatasetDict with train/val/test splits
        """
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
            
        df = self.processed_data.copy()
        
        # Create train/val/test splits
        n_samples = len(df)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df[:n_train]
        val_df = df[n_train:n_train + n_val]
        test_df = df[n_train + n_val:]
        
        # Convert to HuggingFace format
        def df_to_hf_format(subset_df):
            return {
                'id': subset_df['id'].tolist(),
                'input_text': subset_df['full_content'].tolist(),
                'target_text': subset_df['summary'].tolist(),
                'subreddit': subset_df['subreddit'].tolist(),
                'score': subset_df['score'].tolist()
            }
        
        dataset_dict = DatasetDict({
            'train': Dataset.from_dict(df_to_hf_format(train_df)),
            'validation': Dataset.from_dict(df_to_hf_format(val_df)),
            'test': Dataset.from_dict(df_to_hf_format(test_df))
        })
        
        return dataset_dict
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.processed_data is None:
            return {}
            
        df = self.processed_data
        
        stats = {
            'total_samples': len(df),
            'avg_content_length': df['content_length'].mean(),
            'avg_summary_length': df['summary_length'].mean(),
            'content_length_std': df['content_length'].std(),
            'summary_length_std': df['summary_length'].std(),
            'unique_subreddits': df['subreddit'].nunique(),
            'top_subreddits': df['subreddit'].value_counts().head(10).to_dict(),
            'score_distribution': {
                'mean': df['score'].mean(),
                'median': df['score'].median(),
                'std': df['score'].std()
            }
        }
        
        return stats 