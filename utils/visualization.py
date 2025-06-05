import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os


def create_evaluation_plots(results: Dict[str, Any], output_dir: str):
    """
    Create comprehensive evaluation plots from experiment results.
    
    Args:
        results: Dictionary containing experiment results
        output_dir: Directory to save plots
    """
    plt.style.use('seaborn-v0_8')
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: ROUGE scores comparison
    create_rouge_comparison_plot(results, plots_dir)
    
    # Plot 2: BERTScore comparison
    create_bertscore_comparison_plot(results, plots_dir)
    
    # Plot 3: Quality metrics radar chart
    create_quality_radar_chart(results, plots_dir)
    
    # Plot 4: Length distribution analysis
    create_length_analysis_plot(results, plots_dir)
    
    # Plot 5: Performance overview
    create_performance_overview(results, plots_dir)
    
    print(f"Evaluation plots saved to {plots_dir}")


def create_rouge_comparison_plot(results: Dict[str, Any], output_dir: str):
    """Create a comparison plot for ROUGE scores across different approaches."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    approaches = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for approach, result in results.items():
        if approach in ['dataset_stats', 'comparison']:
            continue
            
        if isinstance(result, dict) and 'rouge1' in result:
            approaches.append(approach)
            rouge1_scores.append(result['rouge1'])
            rouge2_scores.append(result['rouge2'])
            rougeL_scores.append(result['rougeL'])
        else:
            # Handle nested results
            for sub_approach, sub_result in result.items():
                if 'rouge1' in sub_result:
                    approaches.append(sub_approach)
                    rouge1_scores.append(sub_result['rouge1'])
                    rouge2_scores.append(sub_result['rouge2'])
                    rougeL_scores.append(sub_result['rougeL'])
    
    x = np.arange(len(approaches))
    width = 0.25
    
    bars1 = ax.bar(x - width, rouge1_scores, width, label='ROUGE-1', alpha=0.8)
    bars2 = ax.bar(x, rouge2_scores, width, label='ROUGE-2', alpha=0.8)
    bars3 = ax.bar(x + width, rougeL_scores, width, label='ROUGE-L', alpha=0.8)
    
    ax.set_xlabel('Approach')
    ax.set_ylabel('ROUGE Score')
    ax.set_title('ROUGE Scores Comparison Across Approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rouge_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_bertscore_comparison_plot(results: Dict[str, Any], output_dir: str):
    """Create a comparison plot for BERTScore metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    approaches = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for approach, result in results.items():
        if approach in ['dataset_stats', 'comparison']:
            continue
            
        if isinstance(result, dict) and 'bert_f1' in result:
            approaches.append(approach)
            precision_scores.append(result['bert_precision'])
            recall_scores.append(result['bert_recall'])
            f1_scores.append(result['bert_f1'])
        else:
            # Handle nested results
            for sub_approach, sub_result in result.items():
                if 'bert_f1' in sub_result:
                    approaches.append(sub_approach)
                    precision_scores.append(sub_result['bert_precision'])
                    recall_scores.append(sub_result['bert_recall'])
                    f1_scores.append(sub_result['bert_f1'])
    
    x = np.arange(len(approaches))
    width = 0.25
    
    ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
    ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1', alpha=0.8)
    
    ax.set_xlabel('Approach')
    ax.set_ylabel('BERTScore')
    ax.set_title('BERTScore Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bertscore_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_quality_radar_chart(results: Dict[str, Any], output_dir: str):
    """Create a radar chart showing different quality metrics."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Define metrics for radar chart
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore F1', 'Coherence', 'Lexical Diversity']
    
    approaches_data = {}
    
    for approach, result in results.items():
        if approach in ['dataset_stats', 'comparison']:
            continue
            
        if isinstance(result, dict) and 'rouge1' in result:
            values = [
                result['rouge1'],
                result['rouge2'],
                result['rougeL'],
                result['bert_f1'],
                result.get('avg_coherence', 0),
                result.get('avg_lexical_diversity', 0)
            ]
            approaches_data[approach] = values
        else:
            # Handle nested results
            for sub_approach, sub_result in result.items():
                if 'rouge1' in sub_result:
                    values = [
                        sub_result['rouge1'],
                        sub_result['rouge2'],
                        sub_result['rougeL'],
                        sub_result['bert_f1'],
                        sub_result.get('avg_coherence', 0),
                        sub_result.get('avg_lexical_diversity', 0)
                    ]
                    approaches_data[sub_approach] = values
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Colors for different approaches
    colors = plt.cm.Set3(np.linspace(0, 1, len(approaches_data)))
    
    for i, (approach, values) in enumerate(approaches_data.items()):
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=approach, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Quality Metrics Comparison (Radar Chart)', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_length_analysis_plot(results: Dict[str, Any], output_dir: str):
    """Create plots analyzing summary length characteristics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prepare data
    approaches = []
    pred_lengths = []
    ref_lengths = []
    compression_ratios = []
    length_differences = []
    
    for approach, result in results.items():
        if approach in ['dataset_stats', 'comparison']:
            continue
            
        if isinstance(result, dict) and 'avg_prediction_length' in result:
            approaches.append(approach)
            pred_lengths.append(result['avg_prediction_length'])
            ref_lengths.append(result['avg_reference_length'])
            compression_ratios.append(result.get('avg_compression_ratio', 0))
            length_differences.append(result.get('length_difference_mae', 0))
        else:
            # Handle nested results
            for sub_approach, sub_result in result.items():
                if 'avg_prediction_length' in sub_result:
                    approaches.append(sub_approach)
                    pred_lengths.append(sub_result['avg_prediction_length'])
                    ref_lengths.append(sub_result['avg_reference_length'])
                    compression_ratios.append(sub_result.get('avg_compression_ratio', 0))
                    length_differences.append(sub_result.get('length_difference_mae', 0))
    
    # Plot 1: Prediction vs Reference Length
    x = np.arange(len(approaches))
    width = 0.35
    
    ax1.bar(x - width/2, pred_lengths, width, label='Prediction', alpha=0.8)
    ax1.bar(x + width/2, ref_lengths, width, label='Reference', alpha=0.8)
    ax1.set_xlabel('Approach')
    ax1.set_ylabel('Average Length (words)')
    ax1.set_title('Prediction vs Reference Length')
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Compression Ratios
    ax2.bar(approaches, compression_ratios, alpha=0.8, color='green')
    ax2.set_xlabel('Approach')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Compression Ratios by Approach')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Length Difference (MAE)
    ax3.bar(approaches, length_differences, alpha=0.8, color='red')
    ax3.set_xlabel('Approach')
    ax3.set_ylabel('Length Difference (MAE)')
    ax3.set_title('Length Difference from Reference')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Length Distribution (if we have individual predictions)
    ax4.text(0.5, 0.5, 'Length Distribution\n(Requires individual predictions)', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Summary Length Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_overview(results: Dict[str, Any], output_dir: str):
    """Create an overview plot showing all key performance metrics."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for heatmap
    approaches = []
    metrics_data = []
    
    metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore F1', 'Coherence']
    
    for approach, result in results.items():
        if approach in ['dataset_stats', 'comparison']:
            continue
            
        if isinstance(result, dict) and 'rouge1' in result:
            approaches.append(approach)
            row = [
                result['rouge1'],
                result['rouge2'],
                result['rougeL'],
                result['bert_f1'],
                result.get('avg_coherence', 0)
            ]
            metrics_data.append(row)
        else:
            # Handle nested results
            for sub_approach, sub_result in result.items():
                if 'rouge1' in sub_result:
                    approaches.append(sub_approach)
                    row = [
                        sub_result['rouge1'],
                        sub_result['rouge2'],
                        sub_result['rougeL'],
                        sub_result['bert_f1'],
                        sub_result.get('avg_coherence', 0)
                    ]
                    metrics_data.append(row)
    
    # Create DataFrame for heatmap
    df = pd.DataFrame(metrics_data, index=approaches, columns=metric_names)
    
    # Create heatmap
    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax, 
                cbar_kws={'label': 'Score'})
    ax.set_title('Performance Overview Heatmap', fontsize=16, pad=20)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Approaches', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_dataset_statistics_plot(stats: Dict[str, Any], output_dir: str):
    """Create visualizations for dataset statistics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Content and Summary Length Distribution
    content_length = stats['avg_content_length']
    summary_length = stats['avg_summary_length']
    content_std = stats['content_length_std']
    summary_std = stats['summary_length_std']
    
    categories = ['Content', 'Summary']
    means = [content_length, summary_length]
    stds = [content_std, summary_std]
    
    ax1.bar(categories, means, yerr=stds, capsize=5, alpha=0.8)
    ax1.set_ylabel('Length (characters)')
    ax1.set_title('Average Text Length with Standard Deviation')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top Subreddits
    if 'top_subreddits' in stats:
        top_subreddits = stats['top_subreddits']
        subreddits = list(top_subreddits.keys())[:10]
        counts = list(top_subreddits.values())[:10]
        
        ax2.barh(subreddits, counts, alpha=0.8)
        ax2.set_xlabel('Number of Posts')
        ax2.set_title('Top 10 Subreddits by Post Count')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Score Distribution
    if 'score_distribution' in stats:
        score_stats = stats['score_distribution']
        ax3.text(0.1, 0.7, f"Mean Score: {score_stats['mean']:.1f}", transform=ax3.transAxes)
        ax3.text(0.1, 0.6, f"Median Score: {score_stats['median']:.1f}", transform=ax3.transAxes)
        ax3.text(0.1, 0.5, f"Std Dev: {score_stats['std']:.1f}", transform=ax3.transAxes)
        ax3.set_title('Reddit Score Statistics')
        ax3.axis('off')
    
    # Plot 4: Dataset Overview
    total_samples = stats['total_samples']
    unique_subreddits = stats['unique_subreddits']
    
    ax4.text(0.1, 0.7, f"Total Samples: {total_samples:,}", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.6, f"Unique Subreddits: {unique_subreddits:,}", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.5, f"Avg Content/Summary Ratio: {content_length/summary_length:.1f}", 
             transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Dataset Overview')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_results_table(results: Dict[str, Any], output_dir: str):
    """Save results as a formatted table."""
    # Prepare data for table
    table_data = []
    
    for approach, result in results.items():
        if approach in ['dataset_stats', 'comparison']:
            continue
            
        if isinstance(result, dict) and 'rouge1' in result:
            row = {
                'Approach': approach,
                'ROUGE-1': f"{result['rouge1']:.4f} ± {result.get('rouge1_std', 0):.4f}",
                'ROUGE-2': f"{result['rouge2']:.4f} ± {result.get('rouge2_std', 0):.4f}",
                'ROUGE-L': f"{result['rougeL']:.4f} ± {result.get('rougeL_std', 0):.4f}",
                'BERTScore F1': f"{result['bert_f1']:.4f} ± {result.get('bert_f1_std', 0):.4f}",
                'Coherence': f"{result.get('avg_coherence', 0):.4f}",
                'Compression Ratio': f"{result.get('avg_compression_ratio', 0):.4f}"
            }
            table_data.append(row)
        else:
            # Handle nested results
            for sub_approach, sub_result in result.items():
                if 'rouge1' in sub_result:
                    row = {
                        'Approach': sub_approach,
                        'ROUGE-1': f"{sub_result['rouge1']:.4f} ± {sub_result.get('rouge1_std', 0):.4f}",
                        'ROUGE-2': f"{sub_result['rouge2']:.4f} ± {sub_result.get('rouge2_std', 0):.4f}",
                        'ROUGE-L': f"{sub_result['rougeL']:.4f} ± {sub_result.get('rougeL_std', 0):.4f}",
                        'BERTScore F1': f"{sub_result['bert_f1']:.4f} ± {sub_result.get('bert_f1_std', 0):.4f}",
                        'Coherence': f"{sub_result.get('avg_coherence', 0):.4f}",
                        'Compression Ratio': f"{sub_result.get('avg_compression_ratio', 0):.4f}"
                    }
                    table_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(output_dir, 'results_table.csv'), index=False)
    
    # Save as formatted text table
    with open(os.path.join(output_dir, 'results_table.txt'), 'w') as f:
        f.write(df.to_string(index=False))
    
    print(f"Results table saved to {output_dir}")


# Set the style for all plots
plt.style.use('default')
sns.set_palette("husl") 