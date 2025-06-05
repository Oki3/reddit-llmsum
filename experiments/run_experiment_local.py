#!/usr/bin/env python3
"""
Main experiment runner for Reddit summarization research.
Compares zero-shot vs fine-tuned approaches using Mistral-7B.
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Local imports
from data.dataset_loader import WebISTLDRDatasetLoader
from models.mistral_summarizer import MistralSummarizer
from evaluation.metrics import SummarizationEvaluator
from utils.experiment_config import ExperimentConfig
from utils.visualization import create_evaluation_plots


class ExperimentRunner:
    """
    Main class for running Reddit summarization experiments.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the experiment runner.
        
        Args:
            config_path: Path to experiment configuration file
        """
        self.config = ExperimentConfig(config_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir = f"results/experiment_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Experiment results will be saved to: {self.output_dir}")
    
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        print("Loading and preparing dataset...")
        
        # Initialize dataset loader
        self.loader = WebISTLDRDatasetLoader(self.config.data_dir, use_hf_dataset=True)
        
        # Calculate total sample size needed (with some buffer for filtering)
        total_needed = int(self.config.eval_sample_size * 1.5 / (1 - self.config.test_size - self.config.val_size))
        
        # Load raw data with sample size for faster loading
        print(f"Loading {total_needed} samples for evaluation size of {self.config.eval_sample_size}")
        self.loader.load_raw_data(sample_size=total_needed)
        
        # Preprocess data
        self.loader.preprocess_data(
            min_content_length=self.config.min_content_length,
            max_content_length=self.config.max_content_length,
            min_summary_length=self.config.min_summary_length,
            max_summary_length=self.config.max_summary_length
        )
        
        # Create HuggingFace dataset
        self.dataset = self.loader.create_huggingface_dataset(
            test_size=self.config.test_size,
            val_size=self.config.val_size
        )
        
        # Get dataset statistics
        stats = self.loader.get_statistics()
        self.results['dataset_stats'] = stats
        
        print(f"Dataset loaded successfully:")
        print(f"  - Total samples: {stats['total_samples']}")
        print(f"  - Train: {len(self.dataset['train'])}")
        print(f"  - Validation: {len(self.dataset['validation'])}")
        print(f"  - Test: {len(self.dataset['test'])}")
        
    def run_zero_shot_experiment(self):
        """Run zero-shot summarization experiment."""
        print("\n" + "="*50)
        print("RUNNING ZERO-SHOT EXPERIMENT")
        print("="*50)
        
        # Initialize model
        model = MistralSummarizer(
            model_name=self.config.model_name,
            load_in_8bit=self.config.load_in_8bit
        )
        
        # Get test data
        test_data = self.dataset['test']
        test_posts = test_data['input_text'][:self.config.eval_sample_size]
        test_references = test_data['target_text'][:self.config.eval_sample_size]
        
        # Test different prompt types
        prompt_types = ['instruct', 'few_shot']
        zero_shot_results = {}
        
        for prompt_type in prompt_types:
            print(f"\nTesting {prompt_type} prompting...")
            
            # Generate summaries
            predictions = model.batch_generate_summaries(
                test_posts,
                max_length=self.config.max_generation_length,
                prompt_type=prompt_type,
                batch_size=self.config.batch_size
            )
            
            # Evaluate
            evaluator = SummarizationEvaluator()
            results = evaluator.comprehensive_evaluation(
                predictions=predictions,
                references=test_references,
                original_texts=test_posts,
                use_llm_evaluator=self.config.use_llm_evaluator
            )
            
            zero_shot_results[f'zero_shot_{prompt_type}'] = results
            
            # Save predictions
            self.save_predictions(
                predictions, test_references, test_posts,
                f"{self.output_dir}/zero_shot_{prompt_type}_predictions.json"
            )
            
            print(f"Zero-shot {prompt_type} results:")
            print(f"  ROUGE-1: {results['rouge1']:.4f}")
            print(f"  ROUGE-2: {results['rouge2']:.4f}")
            print(f"  ROUGE-L: {results['rougeL']:.4f}")
            print(f"  BERTScore F1: {results['bert_f1']:.4f}")
        
        self.results['zero_shot'] = zero_shot_results
        
    def run_fine_tuning_experiment(self):
        """Run fine-tuning experiment."""
        print("\n" + "="*50)
        print("RUNNING FINE-TUNING EXPERIMENT")
        print("="*50)
        
        # Initialize model
        model = MistralSummarizer(
            model_name=self.config.model_name,
            load_in_8bit=self.config.load_in_8bit
        )
        
        # Fine-tune the model
        model.fine_tune(
            train_dataset=self.dataset,
            val_dataset=self.dataset,
            output_dir=f"{self.output_dir}/fine_tuned_model",
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )
        
        # Load the fine-tuned model
        model.load_fine_tuned(f"{self.output_dir}/fine_tuned_model")
        
        # Evaluate on test set
        test_data = self.dataset['test']
        test_posts = test_data['input_text'][:self.config.eval_sample_size]
        test_references = test_data['target_text'][:self.config.eval_sample_size]
        
        print("Evaluating fine-tuned model...")
        
        # Generate summaries
        predictions = model.batch_generate_summaries(
            test_posts,
            max_length=self.config.max_generation_length,
            prompt_type='instruct',
            batch_size=self.config.batch_size
        )
        
        # Evaluate
        evaluator = SummarizationEvaluator()
        results = evaluator.comprehensive_evaluation(
            predictions=predictions,
            references=test_references,
            original_texts=test_posts,
            use_llm_evaluator=self.config.use_llm_evaluator
        )
        
        self.results['fine_tuned'] = results
        
        # Save predictions
        self.save_predictions(
            predictions, test_references, test_posts,
            f"{self.output_dir}/fine_tuned_predictions.json"
        )
        
        print("Fine-tuned model results:")
        print(f"  ROUGE-1: {results['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['rougeL']:.4f}")
        print(f"  BERTScore F1: {results['bert_f1']:.4f}")
    
    def compare_approaches(self):
        """Compare different approaches and create analysis."""
        print("\n" + "="*50)
        print("COMPARING APPROACHES")
        print("="*50)
        
        # Create comparison table
        comparison_data = []
        
        for approach, results in self.results.items():
            if approach == 'dataset_stats':
                continue
                
            if isinstance(results, dict) and 'rouge1' in results:
                comparison_data.append({
                    'Approach': approach,
                    'ROUGE-1': results['rouge1'],
                    'ROUGE-2': results['rouge2'],
                    'ROUGE-L': results['rougeL'],
                    'BERTScore F1': results['bert_f1'],
                    'Coherence': results.get('avg_coherence', 0),
                    'Compression Ratio': results.get('avg_compression_ratio', 0)
                })
            else:
                # Handle nested results (like zero_shot with multiple prompt types)
                for sub_approach, sub_results in results.items():
                    comparison_data.append({
                        'Approach': sub_approach,
                        'ROUGE-1': sub_results['rouge1'],
                        'ROUGE-2': sub_results['rouge2'],
                        'ROUGE-L': sub_results['rougeL'],
                        'BERTScore F1': sub_results['bert_f1'],
                        'Coherence': sub_results.get('avg_coherence', 0),
                        'Compression Ratio': sub_results.get('avg_compression_ratio', 0)
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nComparison of approaches:")
        print(comparison_df.round(4))
        
        # Save comparison
        comparison_df.to_csv(f"{self.output_dir}/approach_comparison.csv", index=False)
        
        # Find best performing approach
        best_rouge1 = comparison_df.loc[comparison_df['ROUGE-1'].idxmax()]
        best_rouge2 = comparison_df.loc[comparison_df['ROUGE-2'].idxmax()]
        best_bertscore = comparison_df.loc[comparison_df['BERTScore F1'].idxmax()]
        
        print(f"\nBest ROUGE-1: {best_rouge1['Approach']} ({best_rouge1['ROUGE-1']:.4f})")
        print(f"Best ROUGE-2: {best_rouge2['Approach']} ({best_rouge2['ROUGE-2']:.4f})")
        print(f"Best BERTScore: {best_bertscore['Approach']} ({best_bertscore['BERTScore F1']:.4f})")
        
        self.results['comparison'] = comparison_df.to_dict('records')
    
    def create_analysis_report(self):
        """Create comprehensive analysis report."""
        print("\nCreating analysis report...")
        
        evaluator = SummarizationEvaluator()
        
        # Create report for each approach
        reports = {}
        for approach, results in self.results.items():
            if approach in ['dataset_stats', 'comparison']:
                continue
                
            if isinstance(results, dict) and 'rouge1' in results:
                reports[approach] = evaluator.create_evaluation_report(results)
            else:
                for sub_approach, sub_results in results.items():
                    reports[sub_approach] = evaluator.create_evaluation_report(sub_results)
        
        # Combine all reports
        full_report = "# Reddit Summarization Research Results\n\n"
        full_report += f"**Experiment Date:** {self.timestamp}\n\n"
        
        # Dataset statistics
        if 'dataset_stats' in self.results:
            stats = self.results['dataset_stats']
            full_report += "## Dataset Statistics\n"
            full_report += f"- Total samples: {stats['total_samples']}\n"
            full_report += f"- Average content length: {stats['avg_content_length']:.1f} characters\n"
            full_report += f"- Average summary length: {stats['avg_summary_length']:.1f} characters\n"
            full_report += f"- Unique subreddits: {stats['unique_subreddits']}\n\n"
        
        # Add individual reports
        for approach, report in reports.items():
            full_report += f"# {approach.replace('_', ' ').title()}\n\n"
            full_report += report + "\n"
        
        # Research insights
        full_report += "## Research Insights\n\n"
        full_report += self.generate_research_insights()
        
        # Save report
        with open(f"{self.output_dir}/analysis_report.md", 'w') as f:
            f.write(full_report)
        
        print(f"Analysis report saved to {self.output_dir}/analysis_report.md")
    
    def generate_research_insights(self) -> str:
        """Generate insights based on experimental results."""
        insights = []
        
        # Compare zero-shot approaches
        if 'zero_shot' in self.results:
            zero_shot_results = self.results['zero_shot']
            if 'zero_shot_instruct' in zero_shot_results and 'zero_shot_few_shot' in zero_shot_results:
                instruct_rouge1 = zero_shot_results['zero_shot_instruct']['rouge1']
                few_shot_rouge1 = zero_shot_results['zero_shot_few_shot']['rouge1']
                
                if few_shot_rouge1 > instruct_rouge1:
                    insights.append("Few-shot prompting outperforms instruction-based prompting for zero-shot summarization.")
                else:
                    insights.append("Instruction-based prompting is more effective than few-shot for zero-shot summarization.")
        
        # Compare zero-shot vs fine-tuned
        if 'zero_shot' in self.results and 'fine_tuned' in self.results:
            # Get best zero-shot result
            best_zero_shot_rouge1 = max([
                result['rouge1'] for result in self.results['zero_shot'].values()
            ])
            fine_tuned_rouge1 = self.results['fine_tuned']['rouge1']
            
            improvement = ((fine_tuned_rouge1 - best_zero_shot_rouge1) / best_zero_shot_rouge1) * 100
            
            if improvement > 5:
                insights.append(f"Fine-tuning provides significant improvement ({improvement:.1f}%) over zero-shot approaches.")
            elif improvement > 0:
                insights.append(f"Fine-tuning provides modest improvement ({improvement:.1f}%) over zero-shot approaches.")
            else:
                insights.append("Zero-shot approaches perform comparably to fine-tuning, suggesting strong baseline performance.")
        
        # Performance observations
        if 'comparison' in self.results:
            best_performance = max(self.results['comparison'], key=lambda x: x['ROUGE-1'])
            insights.append(f"Best overall performance achieved by {best_performance['Approach']} with ROUGE-1 of {best_performance['ROUGE-1']:.4f}.")
        
        return "\n".join(f"- {insight}" for insight in insights)
    
    def save_predictions(self, predictions: List[str], references: List[str], 
                        inputs: List[str], filepath: str):
        """Save predictions for analysis."""
        data = []
        for i, (pred, ref, inp) in enumerate(zip(predictions, references, inputs)):
            data.append({
                'id': i,
                'input': inp,
                'reference': ref,
                'prediction': pred
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_results(self):
        """Save all experiment results."""
        # Save complete results
        with open(f"{self.output_dir}/complete_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save experiment configuration
        self.config.save(f"{self.output_dir}/experiment_config.json")
        
        print(f"All results saved to {self.output_dir}")
    
    def run_complete_experiment(self):
        """Run the complete experimental pipeline."""
        print("Starting Reddit LLM Summarization Research Experiment")
        print("="*60)
        
        try:
            # Load and prepare data
            self.load_and_prepare_data()
            
            # Run zero-shot experiments
            if self.config.run_zero_shot:
                self.run_zero_shot_experiment()
            
            # Run fine-tuning experiment
            if self.config.run_fine_tuning:
                self.run_fine_tuning_experiment()
            
            # Compare approaches
            self.compare_approaches()
            
            # Create analysis report
            self.create_analysis_report()
            
            # Create visualizations
            if self.config.create_plots:
                create_evaluation_plots(self.results, self.output_dir)
            
            # Save all results
            self.save_results()
            
            print("\n" + "="*60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print(f"Results saved to: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"Experiment failed with error: {e}")
            raise


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description="Run Reddit summarization experiments")
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--data-dir', type=str, default='data/webis-tldr-17', 
                       help='Directory containing the dataset')
    parser.add_argument('--no-fine-tuning', action='store_true', 
                       help='Skip fine-tuning experiment')
    parser.add_argument('--no-zero-shot', action='store_true', 
                       help='Skip zero-shot experiment')
    parser.add_argument('--eval-sample-size', type=int, default=100,
                       help='Number of samples to evaluate on')
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = ExperimentConfig()
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.eval_sample_size:
        config.eval_sample_size = args.eval_sample_size
    if args.no_fine_tuning:
        config.run_fine_tuning = False
    if args.no_zero_shot:
        config.run_zero_shot = False
    
    # Run experiment
    runner = ExperimentRunner()
    runner.config = config
    runner.run_complete_experiment()


if __name__ == "__main__":
    main() 