import json
import os
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """
    Configuration class for Reddit summarization experiments.
    Contains all hyperparameters and experimental settings.
    """
    
    # Dataset parameters
    data_dir: str = "data/webis-tldr-17"
    min_content_length: int = 50
    max_content_length: int = 2000
    min_summary_length: int = 10
    max_summary_length: int = 200
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Model parameters
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    load_in_8bit: bool = True
    max_generation_length: int = 150
    
    # Training parameters
    num_epochs: int = 3
    learning_rate: float = 2e-5
    train_batch_size: int = 4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # Evaluation parameters
    eval_sample_size: int = 100
    use_llm_evaluator: bool = False
    
    # Experiment control
    run_zero_shot: bool = True
    run_fine_tuning: bool = True
    create_plots: bool = True
    
    # Output settings
    save_predictions: bool = True
    save_models: bool = True
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration, optionally loading from file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, config_path: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update attributes
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, config_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path to save configuration
        """
        config_dict = asdict(self)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check data parameters
        if self.min_content_length >= self.max_content_length:
            raise ValueError("min_content_length must be less than max_content_length")
        
        if self.min_summary_length >= self.max_summary_length:
            raise ValueError("min_summary_length must be less than max_summary_length")
        
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        
        if not (0 < self.val_size < 1):
            raise ValueError("val_size must be between 0 and 1")
        
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1")
        
        # Check training parameters
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Check evaluation parameters
        if self.eval_sample_size <= 0:
            raise ValueError("eval_sample_size must be positive")
        
        return True
    
    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name."""
        components = []
        
        if self.run_zero_shot:
            components.append("zero_shot")
        
        if self.run_fine_tuning:
            components.append("fine_tuned")
        
        components.append(f"epochs_{self.num_epochs}")
        components.append(f"lr_{self.learning_rate}")
        components.append(f"batch_{self.train_batch_size}")
        
        return "_".join(components)
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("Experiment Configuration:")
        print("=" * 40)
        
        print("\nDataset Parameters:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Content length: {self.min_content_length}-{self.max_content_length}")
        print(f"  Summary length: {self.min_summary_length}-{self.max_summary_length}")
        print(f"  Train/Val/Test split: {1-self.test_size-self.val_size:.1f}/{self.val_size:.1f}/{self.test_size:.1f}")
        
        print("\nModel Parameters:")
        print(f"  Model: {self.model_name}")
        print(f"  8-bit loading: {self.load_in_8bit}")
        print(f"  Max generation length: {self.max_generation_length}")
        
        print("\nTraining Parameters:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.train_batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        
        print("\nEvaluation Parameters:")
        print(f"  Sample size: {self.eval_sample_size}")
        print(f"  LLM evaluator: {self.use_llm_evaluator}")
        
        print("\nExperiment Control:")
        print(f"  Run zero-shot: {self.run_zero_shot}")
        print(f"  Run fine-tuning: {self.run_fine_tuning}")
        print(f"  Create plots: {self.create_plots}")
        
        print("=" * 40)


# Predefined configurations for different experiment types
class ExperimentConfigs:
    """Collection of predefined experiment configurations."""
    
    @staticmethod
    def quick_test() -> ExperimentConfig:
        """Configuration for quick testing (small dataset, no fine-tuning)."""
        config = ExperimentConfig()
        config.eval_sample_size = 10
        config.run_fine_tuning = False
        config.use_llm_evaluator = False
        config.create_plots = False
        return config
    
    @staticmethod
    def zero_shot_only() -> ExperimentConfig:
        """Configuration for zero-shot experiments only."""
        config = ExperimentConfig()
        config.run_fine_tuning = False
        config.eval_sample_size = 200
        config.use_llm_evaluator = True
        return config
    
    @staticmethod
    def fine_tuning_focus() -> ExperimentConfig:
        """Configuration focused on fine-tuning experiments."""
        config = ExperimentConfig()
        config.run_zero_shot = False
        config.num_epochs = 5
        config.eval_sample_size = 500
        config.gradient_accumulation_steps = 8
        return config
    
    @staticmethod
    def comprehensive() -> ExperimentConfig:
        """Configuration for comprehensive evaluation."""
        config = ExperimentConfig()
        config.eval_sample_size = 1000
        config.use_llm_evaluator = True
        config.num_epochs = 5
        config.create_plots = True
        return config
    
    @staticmethod
    def memory_efficient() -> ExperimentConfig:
        """Configuration optimized for limited memory."""
        config = ExperimentConfig()
        config.load_in_8bit = True
        config.train_batch_size = 2
        config.batch_size = 2
        config.gradient_accumulation_steps = 8
        config.eval_sample_size = 50
        return config 