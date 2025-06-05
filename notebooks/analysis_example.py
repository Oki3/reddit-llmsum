#!/usr/bin/env python3
"""
Example analysis script for Reddit LLM Summarization Research.
This script demonstrates how to use the project components for analysis.
"""

import sys
import os
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from data.dataset_loader import WebISTLDRDatasetLoader
from models.mistral_summarizer import MistralSummarizer
from evaluation.metrics import SummarizationEvaluator
from utils.visualization import create_evaluation_plots


def demo_dataset_loading():
    """Demonstrate dataset loading and preprocessing."""
    print("=== Dataset Loading Demo ===")
    
    # Initialize dataset loader
    loader = WebISTLDRDatasetLoader('data/webis-tldr-17')
    
    print("Dataset loader initialized")
    print("Note: Download the Webis-TLDR-17 dataset from:")
    print("https://webis.de/data/webis-tldr-17.html")
    print()


def demo_model_usage():
    """Demonstrate model usage with sample data."""
    print("=== Model Usage Demo ===")
    
    # Sample Reddit post
    sample_post = """I've been working at this company for about 2 years now, and I really love my job. 
The work is interesting, my colleagues are great, and I feel like I'm learning a lot. 
However, I just found out that someone who was hired 6 months ago in a similar position 
is making $15,000 more than me. I have more experience and consistently get better performance 
reviews. I'm not sure if I should ask for a raise, look for another job, or just accept it. 
Has anyone been in a similar situation? What would you do?"""

    print("Sample Reddit post:")
    print(sample_post)
    print(f"Length: {len(sample_post)} characters")
    print()
    
    print("To use the Mistral model, uncomment the following code:")
    print("# summarizer = MistralSummarizer(load_in_8bit=True)")
    print("# summary = summarizer.generate_summary(sample_post, prompt_type='instruct')")
    print("# print('Generated summary:', summary)")
    print()


def demo_evaluation():
    """Demonstrate evaluation metrics."""
    print("=== Evaluation Demo ===")
    
    # Sample predictions and references
    sample_predictions = [
        "Employee discovers colleague earns $15K more despite less experience, seeking advice on raise request.",
        "User loves job but learned newer hire makes significantly more, considering options for salary negotiation.",
        "Worker with 2 years experience finds out 6-month colleague earns more, unsure about asking for raise."
    ]

    sample_references = [
        "Employee with 2 years experience discovers newer colleague makes $15,000 more, seeking advice on whether to request raise or find new job.",
        "Worker loves current job but found out less experienced colleague earns significantly more, questioning how to handle salary disparity.",
        "Employee considering options after learning newer hire with less experience and performance makes $15K more."
    ]

    # Initialize evaluator
    evaluator = SummarizationEvaluator()
    
    # Compute evaluation metrics
    print("Computing evaluation metrics...")
    results = evaluator.comprehensive_evaluation(
        predictions=sample_predictions,
        references=sample_references,
        use_llm_evaluator=False
    )

    print("\nEvaluation Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
    print()


def demo_results_analysis():
    """Demonstrate results analysis and visualization."""
    print("=== Results Analysis Demo ===")
    
    # Sample results for demonstration
    sample_results = {
        'zero_shot_instruct': {
            'rouge1': 0.35,
            'rouge2': 0.18,
            'rougeL': 0.28,
            'bert_f1': 0.72,
            'avg_coherence': 0.65,
            'avg_lexical_diversity': 0.58,
            'avg_compression_ratio': 0.12
        },
        'zero_shot_few_shot': {
            'rouge1': 0.38,
            'rouge2': 0.21,
            'rougeL': 0.31,
            'bert_f1': 0.75,
            'avg_coherence': 0.68,
            'avg_lexical_diversity': 0.61,
            'avg_compression_ratio': 0.14
        },
        'fine_tuned': {
            'rouge1': 0.42,
            'rouge2': 0.25,
            'rougeL': 0.35,
            'bert_f1': 0.78,
            'avg_coherence': 0.72,
            'avg_lexical_diversity': 0.63,
            'avg_compression_ratio': 0.16
        }
    }
    
    # Create comparison DataFrame
    comparison_data = []
    for approach, results in sample_results.items():
        comparison_data.append({
            'Approach': approach,
            'ROUGE-1': results['rouge1'],
            'ROUGE-2': results['rouge2'],
            'ROUGE-L': results['rougeL'],
            'BERTScore F1': results['bert_f1']
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("Results Comparison:")
    print(comparison_df.round(4))
    print()
    
    # Find best approaches
    best_rouge1 = comparison_df.loc[comparison_df['ROUGE-1'].idxmax()]
    best_rouge2 = comparison_df.loc[comparison_df['ROUGE-2'].idxmax()]
    best_bertscore = comparison_df.loc[comparison_df['BERTScore F1'].idxmax()]

    print("Key Findings:")
    print(f"- Best ROUGE-1: {best_rouge1['Approach']} ({best_rouge1['ROUGE-1']:.4f})")
    print(f"- Best ROUGE-2: {best_rouge2['Approach']} ({best_rouge2['ROUGE-2']:.4f})")
    print(f"- Best BERTScore: {best_bertscore['Approach']} ({best_bertscore['BERTScore F1']:.4f})")
    print()


def demo_research_insights():
    """Demonstrate research insights generation."""
    print("=== Research Insights Demo ===")
    
    insights = [
        "1. Fine-tuning provides significant improvement (10.5%) over zero-shot approaches",
        "2. Few-shot prompting outperforms instruction-based prompting for zero-shot summarization",
        "3. Mistral-7B demonstrates strong baseline performance for Reddit summarization",
        "4. Privacy-preserving local LLMs are viable for social media content analysis",
        "5. BERTScore improvements indicate better semantic preservation in fine-tuned models"
    ]
    
    print("Research Insights:")
    for insight in insights:
        print(f"  {insight}")
    print()
    
    print("Conclusions:")
    print("  - Lightweight LLMs can effectively summarize social media discussions")
    print("  - Domain-specific fine-tuning improves performance meaningfully")
    print("  - Open-source approaches provide privacy-preserving alternatives")
    print("  - Multiple evaluation metrics capture different aspects of quality")
    print()


def show_usage_instructions():
    """Show how to run the full experimental pipeline."""
    print("=== Running Full Experiments ===")
    print()
    print("To run the complete experimental pipeline:")
    print()
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Download the dataset:")
    print("   Download Webis-TLDR-17 from https://webis.de/data/webis-tldr-17.html")
    print("   Extract to data/webis-tldr-17/")
    print()
    print("3. Run experiments:")
    print("   # Full experiment")
    print("   python experiments/run_experiment.py")
    print()
    print("   # Zero-shot only")
    print("   python experiments/run_experiment.py --no-fine-tuning")
    print()
    print("   # Custom evaluation size")
    print("   python experiments/run_experiment.py --eval-sample-size 500")
    print()
    print("   # Quick test")
    print("   python experiments/run_experiment.py --eval-sample-size 10 --no-fine-tuning")
    print()


def main():
    """Run the complete analysis demo."""
    print("Reddit LLM Summarization Research - Analysis Demo")
    print("=" * 60)
    print()
    
    demo_dataset_loading()
    demo_model_usage()
    demo_evaluation()
    demo_results_analysis()
    demo_research_insights()
    show_usage_instructions()
    
    print("Demo completed! Check the individual modules for more details.")


if __name__ == "__main__":
    main() 