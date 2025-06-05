import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import re

# ROUGE and BERTScore
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# For LLM-as-evaluator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

# Statistical measures
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error


class SummarizationEvaluator:
    """
    Comprehensive evaluation suite for summarization quality.
    Includes ROUGE, BERTScore, coherence metrics, and LLM-as-evaluator.
    """
    
    def __init__(self, evaluator_model: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the evaluator.
        
        Args:
            evaluator_model: Model to use for LLM-as-evaluator metrics
        """
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize LLM evaluator (optional)
        self.evaluator_model = None
        self.evaluator_tokenizer = None
        self.evaluator_model_name = evaluator_model
        
    def load_llm_evaluator(self):
        """Load the LLM evaluator model."""
        if self.evaluator_model is None:
            print(f"Loading LLM evaluator: {self.evaluator_model_name}")
            self.evaluator_tokenizer = AutoTokenizer.from_pretrained(self.evaluator_model_name)
            self.evaluator_model = AutoModelForCausalLM.from_pretrained(self.evaluator_model_name)
            
            if self.evaluator_tokenizer.pad_token is None:
                self.evaluator_tokenizer.pad_token = self.evaluator_tokenizer.eos_token
    
    def compute_rouge_scores(self, predictions: List[str], 
                           references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores for predictions vs references.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'rouge1_std': np.std(rouge1_scores),
            'rouge2_std': np.std(rouge2_scores),
            'rougeL_std': np.std(rougeL_scores)
        }
    
    def compute_bert_score(self, predictions: List[str], 
                          references: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with BERTScore metrics
        """
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item(),
            'bert_precision_std': P.std().item(),
            'bert_recall_std': R.std().item(),
            'bert_f1_std': F1.std().item()
        }
    
    def compute_length_metrics(self, predictions: List[str], 
                             references: List[str]) -> Dict[str, float]:
        """
        Compute length-based metrics.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with length metrics
        """
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        compression_ratios = []
        for pred_len, ref_len in zip(pred_lengths, ref_lengths):
            if ref_len > 0:
                compression_ratios.append(pred_len / ref_len)
        
        return {
            'avg_prediction_length': np.mean(pred_lengths),
            'avg_reference_length': np.mean(ref_lengths),
            'avg_compression_ratio': np.mean(compression_ratios),
            'length_difference_mae': mean_absolute_error(ref_lengths, pred_lengths)
        }
    
    def compute_lexical_diversity(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute lexical diversity metrics.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with diversity metrics
        """
        diversities = []
        
        for text in texts:
            words = word_tokenize(text.lower())
            words = [w for w in words if w not in string.punctuation and w not in self.stop_words]
            
            if len(words) > 0:
                unique_words = len(set(words))
                total_words = len(words)
                diversity = unique_words / total_words
                diversities.append(diversity)
        
        return {
            'avg_lexical_diversity': np.mean(diversities),
            'lexical_diversity_std': np.std(diversities)
        }
    
    def compute_coherence_score(self, text: str) -> float:
        """
        Compute a simple coherence score based on sentence connectivity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Coherence score (0-1)
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence measure: overlap between consecutive sentences
        overlaps = []
        for i in range(len(sentences) - 1):
            words1 = set(word_tokenize(sentences[i].lower()))
            words2 = set(word_tokenize(sentences[i + 1].lower()))
            
            # Remove stopwords and punctuation
            words1 = {w for w in words1 if w not in self.stop_words and w not in string.punctuation}
            words2 = {w for w in words2 if w not in self.stop_words and w not in string.punctuation}
            
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def compute_coherence_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """
        Compute coherence metrics for a list of predictions.
        
        Args:
            predictions: List of generated summaries
            
        Returns:
            Dictionary with coherence metrics
        """
        coherence_scores = [self.compute_coherence_score(pred) for pred in predictions]
        
        return {
            'avg_coherence': np.mean(coherence_scores),
            'coherence_std': np.std(coherence_scores)
        }
    
    def llm_evaluate_summary(self, original_text: str, summary: str, 
                           aspects: List[str] = ["accuracy", "coherence", "fluency"]) -> Dict[str, float]:
        """
        Use LLM to evaluate summary quality on multiple aspects.
        
        Args:
            original_text: Original text being summarized
            summary: Generated summary
            aspects: Aspects to evaluate
            
        Returns:
            Dictionary with LLM evaluation scores
        """
        if self.evaluator_model is None:
            self.load_llm_evaluator()
        
        scores = {}
        
        for aspect in aspects:
            prompt = f"""Rate the quality of this summary on a scale of 1-5 for {aspect}:

Original text: {original_text[:500]}...

Summary: {summary}

Rate the {aspect} (1=very poor, 5=excellent): """

            inputs = self.evaluator_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.evaluator_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.evaluator_tokenizer.eos_token_id
                )
            
            response = self.evaluator_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract score from response
            score_match = re.search(r'(\d+)', response.split(prompt)[-1])
            if score_match:
                score = int(score_match.group(1))
                score = max(1, min(5, score))  # Clamp to 1-5 range
                scores[f"llm_{aspect}"] = score / 5.0  # Normalize to 0-1
            else:
                scores[f"llm_{aspect}"] = 0.5  # Default score if parsing fails
        
        return scores
    
    def comprehensive_evaluation(self, predictions: List[str], references: List[str],
                               original_texts: List[str] = None,
                               use_llm_evaluator: bool = False) -> Dict[str, float]:
        """
        Perform comprehensive evaluation of summarization quality.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            original_texts: Original texts (needed for LLM evaluation)
            use_llm_evaluator: Whether to use LLM-based evaluation
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # ROUGE scores
        print("Computing ROUGE scores...")
        rouge_scores = self.compute_rouge_scores(predictions, references)
        results.update(rouge_scores)
        
        # BERTScore
        print("Computing BERTScore...")
        bert_scores = self.compute_bert_score(predictions, references)
        results.update(bert_scores)
        
        # Length metrics
        print("Computing length metrics...")
        length_metrics = self.compute_length_metrics(predictions, references)
        results.update(length_metrics)
        
        # Lexical diversity
        print("Computing lexical diversity...")
        diversity_metrics = self.compute_lexical_diversity(predictions)
        results.update(diversity_metrics)
        
        # Coherence metrics
        print("Computing coherence metrics...")
        coherence_metrics = self.compute_coherence_metrics(predictions)
        results.update(coherence_metrics)
        
        # LLM-based evaluation (optional)
        if use_llm_evaluator and original_texts is not None:
            print("Running LLM evaluation...")
            llm_scores = {'llm_accuracy': [], 'llm_coherence': [], 'llm_fluency': []}
            
            # Sample a subset for LLM evaluation (expensive)
            sample_size = min(50, len(predictions))
            indices = np.random.choice(len(predictions), sample_size, replace=False)
            
            for idx in indices:
                scores = self.llm_evaluate_summary(
                    original_texts[idx], 
                    predictions[idx]
                )
                for key, value in scores.items():
                    if key in llm_scores:
                        llm_scores[key].append(value)
            
            # Average LLM scores
            for key, values in llm_scores.items():
                if values:
                    results[f"avg_{key}"] = np.mean(values)
                    results[f"{key}_std"] = np.std(values)
        
        return results
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {output_path}")
    
    def create_evaluation_report(self, results: Dict) -> str:
        """
        Create a human-readable evaluation report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted report string
        """
        report = "# Summarization Evaluation Report\n\n"
        
        # ROUGE scores
        report += "## ROUGE Scores\n"
        report += f"- ROUGE-1: {results.get('rouge1', 0):.4f} (±{results.get('rouge1_std', 0):.4f})\n"
        report += f"- ROUGE-2: {results.get('rouge2', 0):.4f} (±{results.get('rouge2_std', 0):.4f})\n"
        report += f"- ROUGE-L: {results.get('rougeL', 0):.4f} (±{results.get('rougeL_std', 0):.4f})\n\n"
        
        # BERTScore
        report += "## BERTScore (Semantic Similarity)\n"
        report += f"- Precision: {results.get('bert_precision', 0):.4f} (±{results.get('bert_precision_std', 0):.4f})\n"
        report += f"- Recall: {results.get('bert_recall', 0):.4f} (±{results.get('bert_recall_std', 0):.4f})\n"
        report += f"- F1: {results.get('bert_f1', 0):.4f} (±{results.get('bert_f1_std', 0):.4f})\n\n"
        
        # Length and quality metrics
        report += "## Summary Quality Metrics\n"
        report += f"- Average coherence: {results.get('avg_coherence', 0):.4f}\n"
        report += f"- Lexical diversity: {results.get('avg_lexical_diversity', 0):.4f}\n"
        report += f"- Compression ratio: {results.get('avg_compression_ratio', 0):.4f}\n"
        report += f"- Length difference (MAE): {results.get('length_difference_mae', 0):.2f} words\n\n"
        
        # LLM evaluation (if available)
        if 'avg_llm_accuracy' in results:
            report += "## LLM-based Evaluation\n"
            report += f"- Accuracy: {results.get('avg_llm_accuracy', 0):.4f}\n"
            report += f"- Coherence: {results.get('avg_llm_coherence', 0):.4f}\n"
            report += f"- Fluency: {results.get('avg_llm_fluency', 0):.4f}\n\n"
        
        return report 