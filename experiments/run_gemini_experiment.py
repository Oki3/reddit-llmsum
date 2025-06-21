#!/usr/bin/env python3
"""
Gemini API-based Reddit Summarization Experiment Runner
Uses Gemini API for fast, reliable experiments
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Local imports
from data.dataset_loader import WebISTLDRDatasetLoader
from models.gemini_api_summarizer import GeminiAPISummarizer
from evaluation.metrics import SummarizationEvaluator
from utils.visualization import create_evaluation_plots


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='Reddit Summarization Gemini API Experiment')
    parser.add_argument('--eval-sample-size', type=int, default=50,
                       help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Gemini model name (default: gemini-2.0-flash)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save individual predictions to JSON files (default: False)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API calls in seconds to respect rate limits (default: 1.0)')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/gemini_experiment_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Gemini API Experiment results will be saved to:", results_dir)
    print("=" * 60)
    print("🌟 Reddit LLM Summarization Research - Gemini Edition")
    print("=" * 60)
    print(f"⚙️  Configuration:")
    print(f"   • Sample size: {args.eval_sample_size}")
    print(f"   • Model: {args.model}")
    print(f"   • API delay: {args.delay}s (rate limiting)")
    print(f"   • Save predictions: {args.save_predictions}")
    print("=" * 60)
    
    try:
        # Load dataset
        print("\n📊 Loading and preparing dataset...")
        loader = WebISTLDRDatasetLoader()
        
        # Load raw data from HuggingFace
        sample_size = args.eval_sample_size * 2  # Load more to account for filtering
        loader.load_raw_data(sample_size=sample_size)
        
        # Preprocess data
        loader.preprocess_data()
        
        # Create dataset splits
        dataset = loader.create_huggingface_dataset(test_size=0.2, val_size=0.1)
        
        # Initialize API model
        print(f"\n🔗 Initializing Gemini API client...")
        model = GeminiAPISummarizer(
            model_name=args.model,
            api_key=args.api_key
        )
        
        # Initialize evaluator
        evaluator = SummarizationEvaluator()
        
        # Prepare test data
        test_data = dataset["test"]
        test_posts = [item["input_text"] for item in test_data]
        test_summaries = [item["target_text"] for item in test_data]
        
        print(f"📝 Running experiments on {len(test_posts)} samples...")
        
        results = {}
        
        # Run zero-shot instruction experiment
        print("\n" + "="*50)
        print("🎯 RUNNING ZERO-SHOT INSTRUCTION EXPERIMENT")
        print("="*50)
        
        instruction_summaries = model.batch_generate_summaries(
            test_posts, 
            prompt_type="instruct",
            delay=args.delay  # Configurable delay to respect rate limits
        )
        
        instruction_results = evaluator.comprehensive_evaluation(
            instruction_summaries,
            test_summaries,
            test_posts
        )
        results["zero_shot_instruct"] = instruction_results
        
        # Run few-shot experiment
        print("\n" + "="*50)
        print("🎯 RUNNING ZERO-SHOT FEW-SHOT EXPERIMENT")
        print("="*50)
        
        few_shot_summaries = model.batch_generate_summaries(
            test_posts, 
            prompt_type="few_shot",
            delay=args.delay  # Configurable delay to respect rate limits
        )
        
        few_shot_results = evaluator.comprehensive_evaluation(
            few_shot_summaries,
            test_summaries,
            test_posts
        )
        results["zero_shot_few_shot"] = few_shot_results
        
        # Save results
        print("\n💾 Saving results...")
        
        # Save the actual generated summaries (if requested)
        if args.save_predictions:
            print("💾 Saving generated summaries...")
            
            # Save instruction-based predictions
            instruction_predictions = []
            for i, (pred, ref, inp) in enumerate(zip(instruction_summaries, test_summaries, test_posts)):
                instruction_predictions.append({
                    "index": i,
                    "input_text": inp,
                    "generated_summary": pred,
                    "reference_summary": ref,
                    "approach": "zero_shot_instruct"
                })
            
            with open(results_dir / "instruction_predictions.json", "w") as f:
                json.dump(instruction_predictions, f, indent=2)
            
            # Save few-shot predictions
            few_shot_predictions = []
            for i, (pred, ref, inp) in enumerate(zip(few_shot_summaries, test_summaries, test_posts)):
                few_shot_predictions.append({
                    "index": i,
                    "input_text": inp,
                    "generated_summary": pred,
                    "reference_summary": ref,
                    "approach": "zero_shot_few_shot"
                })
            
            with open(results_dir / "few_shot_predictions.json", "w") as f:
                json.dump(few_shot_predictions, f, indent=2)
            
            print(f"📄 Individual predictions saved to {results_dir}/")
        else:
            print("📊 Skipping individual prediction saves (use --save-predictions to enable)")
        
        # Create experiment config
        config = {
            "model_name": model.model_name,
            "eval_sample_size": args.eval_sample_size,
            "api_based": True,
            "save_predictions": args.save_predictions,
            "api_delay": args.delay,
            "timestamp": timestamp,
            "api_provider": "gemini"
        }
        
        with open(results_dir / "experiment_config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        with open(results_dir / "complete_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Create comparison CSV
        comparison_data = []
        for approach, result in results.items():
            comparison_data.append({
                "approach": approach,
                "rouge_1": result.get("rouge1", 0),
                "rouge_2": result.get("rouge2", 0),
                "rouge_l": result.get("rougeL", 0),
                "bertscore_f1": result.get("bert_f1", 0),
                "coherence": result.get("avg_coherence", 0),
                "lexical_diversity": result.get("avg_lexical_diversity", 0)
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(results_dir / "approach_comparison.csv", index=False)
        
        # Generate summary report
        report = generate_summary_report(results, config, len(test_posts))
        with open(results_dir / "SUMMARY.md", "w") as f:
            f.write(report)
        
        # Create visualizations
        print("\n📊 Creating evaluation plots...")
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        create_evaluation_plots(results, plots_dir)
        
        print("\n🎉 Experiment completed successfully!")
        print(f"📁 Results saved to: {results_dir}")
        print(f"📋 Summary report: {results_dir}/SUMMARY.md")
        print(f"📊 Plots saved to: {plots_dir}")
        
        # Print quick summary
        print("\n" + "="*60)
        print("📈 QUICK RESULTS SUMMARY")
        print("="*60)
        
        for approach, result in results.items():
            print(f"\n{approach.upper()}:")
            print(f"  ROUGE-1: {result.get('rouge1', 0):.3f}")
            print(f"  ROUGE-2: {result.get('rouge2', 0):.3f}")
            print(f"  ROUGE-L: {result.get('rougeL', 0):.3f}")
            print(f"  BERTScore F1: {result.get('bert_f1', 0):.3f}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise


def generate_summary_report(results: Dict, config: Dict, num_samples: int) -> str:
    """Generate a markdown summary report."""
    
    report = f"""# Gemini API Summarization Experiment Results

## Experiment Configuration
- **Model**: {config['model_name']}
- **API Provider**: Gemini
- **Sample Size**: {num_samples}
- **Timestamp**: {config['timestamp']}
- **API Delay**: {config['api_delay']}s

## Results Overview

"""
    
    for approach, metrics in results.items():
        report += f"""### {approach.replace('_', ' ').title()}

| Metric | Score |
|--------|-------|
| ROUGE-1 | {metrics.get('rouge1', 0):.3f} |
| ROUGE-2 | {metrics.get('rouge2', 0):.3f} |
| ROUGE-L | {metrics.get('rougeL', 0):.3f} |
| BERTScore Precision | {metrics.get('bert_precision', 0):.3f} |
| BERTScore Recall | {metrics.get('bert_recall', 0):.3f} |
| BERTScore F1 | {metrics.get('bert_f1', 0):.3f} |
| Average Coherence | {metrics.get('avg_coherence', 0):.3f} |
| Average Lexical Diversity | {metrics.get('avg_lexical_diversity', 0):.3f} |

"""
    
    # Add comparison section
    report += """## Approach Comparison

The experiment compared two prompting strategies:
1. **Zero-shot Instruction**: Direct instruction-based prompting
2. **Zero-shot Few-shot**: Few-shot examples with prompting

"""
    
    return report


if __name__ == "__main__":
    main() 