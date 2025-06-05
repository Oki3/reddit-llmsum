# Reddit Summarization Research Results

**Experiment Date:** 20250605_204105

## Dataset Statistics
- Total samples: 7
- Average content length: 1109.4 characters
- Average summary length: 114.3 characters
- Unique subreddits: 7

# Zero Shot Instruct

# Summarization Evaluation Report

## ROUGE Scores
- ROUGE-1: 0.1042 (±0.0000)
- ROUGE-2: 0.0213 (±0.0000)
- ROUGE-L: 0.0833 (±0.0000)

## BERTScore (Semantic Similarity)
- Precision: 0.8392 (±nan)
- Recall: 0.8600 (±nan)
- F1: 0.8494 (±nan)

## Summary Quality Metrics
- Average coherence: 0.0625
- Lexical diversity: 0.9143
- Compression ratio: 2.3214
- Length difference (MAE): 37.00 words


# Zero Shot Few Shot

# Summarization Evaluation Report

## ROUGE Scores
- ROUGE-1: 0.1463 (±0.0000)
- ROUGE-2: 0.0331 (±0.0000)
- ROUGE-L: 0.0813 (±0.0000)

## BERTScore (Semantic Similarity)
- Precision: 0.8350 (±nan)
- Recall: 0.8633 (±nan)
- F1: 0.8489 (±nan)

## Summary Quality Metrics
- Average coherence: 0.0278
- Lexical diversity: 0.8750
- Compression ratio: 3.2857
- Length difference (MAE): 64.00 words


## Research Insights

- Few-shot prompting outperforms instruction-based prompting for zero-shot summarization.
- Best overall performance achieved by zero_shot_few_shot with ROUGE-1 of 0.1463.