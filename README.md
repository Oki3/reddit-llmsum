# Reddit LLM Summarization Research 🤖📝

<p align="center">
  <em>Exploring automatic summarization of social media discussions using lightweight open-source LLMs</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.35+-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/Model-Mistral--7B-green.svg" alt="Model">
</p>

## 🎯 Research Overview

This project investigates the effectiveness of **Mistral-7B**, a lightweight open-source LLM, for automatically summarizing Reddit discussions. We compare zero-shot prompting strategies with fine-tuned approaches using the **Webis-TLDR-17** dataset containing 3.8M Reddit posts with human-written summaries.

### 🔬 Research Questions

1. **Can a small open-source LLM produce coherent and accurate summaries of social media discussions?**
2. **Does fine-tuning improve performance compared to zero-shot or few-shot prompting?**
3. **What are the practical trade-offs of using lightweight local LLMs for summarization?**

### 🏆 Key Findings

- Zero-shot approaches show promising results for Reddit summarization
- CPU offloading enables running 7B models on limited GPU memory
- 8-bit quantization reduces memory usage while maintaining quality

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (recommended) or CPU
- **16GB+ RAM** (32GB recommended for full experiments)
- **50GB+ free disk space**

### 1. Clone and Setup

```bash
git clone <repository-url>
cd reddit-llmsum

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Authenticate with Hugging Face

```bash
# Get access to Mistral-7B model
# 1. Visit: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
# 2. Request access and accept license
# 3. Generate token: https://huggingface.co/settings/tokens

huggingface-cli login
```

### 3. Run Quick Test

```bash
# Test with small sample (no fine-tuning)
python experiments/run_experiment.py --eval-sample-size 5 --no-fine-tuning

# Full zero-shot experiment
python experiments/run_experiment.py --no-fine-tuning

# Complete experiment with fine-tuning (requires powerful GPU)
python experiments/run_experiment.py
```

## 📁 Project Structure

```
reddit-llmsum/
├── 📊 data/                    # Dataset handling and preprocessing
│   ├── dataset_loader.py       # Webis-TLDR-17 loader
│   └── webis-tldr-17/         # Dataset files (auto-downloaded)
├── 🤖 models/                  # Model implementations
│   └── mistral_summarizer.py   # Mistral-7B wrapper with optimizations
├── 📈 evaluation/              # Evaluation metrics and analysis
│   └── metrics.py             # ROUGE, BERTScore, coherence metrics
├── 🧪 experiments/             # Experiment runners
│   └── run_experiment.py      # Main experiment orchestrator
├── 🛠️ utils/                   # Utility functions
│   ├── experiment_config.py    # Configuration management
│   └── visualization.py       # Results visualization
├── 📓 notebooks/               # Analysis examples
│   └── analysis_example.py     # Demo script
└── 📊 results/                 # Generated results (timestamped)
    └── experiment_YYYYMMDD_HHMMSS/
        ├── complete_results.json
        ├── analysis_report.md
        ├── plots/
        └── predictions/
```

## 🔧 Configuration & Hardware Support

### Memory-Constrained Setups

The project automatically handles limited GPU memory through:

- **8-bit quantization** with bitsandbytes
- **CPU offloading** for mixed GPU/CPU inference
- **Gradient accumulation** for effective larger batch sizes

### Configuration Options

```python
# Quick test configuration
config = ExperimentConfigs.quick_test()

# Memory-efficient configuration
config = ExperimentConfigs.memory_efficient()

# Custom configuration
config = ExperimentConfig()
config.eval_sample_size = 100
config.load_in_8bit = True
config.batch_size = 2
```

## 📊 Experimental Approaches

### Zero-Shot Methods

1. **Instruction-based prompting**
   ```
   Summarize this Reddit post in 1-2 sentences: [POST]
   ```

2. **Few-shot prompting**
   ```
   Here are examples of good summaries:
   [EXAMPLES]
   Now summarize: [POST]
   ```

### Fine-Tuning Approach

- Fine-tune Mistral-7B on Reddit summarization pairs
- LoRA (Low-Rank Adaptation) for efficient training
- Compare against zero-shot baselines

## 📈 Evaluation Metrics

### Automatic Metrics
- **ROUGE scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **BERTScore**: Semantic similarity measurement
- **Coherence**: Text connectivity analysis
- **Length metrics**: Compression ratios

### Optional Human-like Evaluation
- **LLM-as-evaluator**: GPT-4 assessment of quality
- **Accuracy**: Factual correctness
- **Fluency**: Language quality

## 🎯 Results & Analysis

After running experiments, find comprehensive results in:

```
results/experiment_YYYYMMDD_HHMMSS/
├── 📋 complete_results.json          # All metrics and results
├── 📝 analysis_report.md             # Human-readable report
├── 📊 approach_comparison.csv        # Comparison table
├── ⚙️ experiment_config.json         # Configuration used
├── 📈 plots/                         # Visualizations
│   ├── rouge_comparison.png
│   ├── bertscore_comparison.png
│   ├── quality_radar_chart.png
│   └── performance_overview.png
└── 💬 predictions/                   # Generated summaries
    ├── zero_shot_instruct_predictions.json
    ├── zero_shot_few_shot_predictions.json
    └── fine_tuned_predictions.json
```

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### 💾 Disk Space Issues
```bash
# Clean caches (can free 40GB+)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/pip

# Remove virtual environment (can recreate)
rm -rf venv
```

#### 🔑 Authentication Issues
```bash
# Ensure you have access to Mistral model
# Visit: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
huggingface-cli login
```

#### 🖥️ GPU Memory Issues
The project automatically handles this, but you can also:
```bash
# Use smaller sample sizes
python experiments/run_experiment.py --eval-sample-size 10

# Force CPU-only mode
python experiments/run_experiment.py --device cpu
```

#### 📦 Import Errors
```bash
# Reinstall in development mode
pip install -e .

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Hardware Requirements

| Setup | CPU | RAM | GPU | Notes |
|-------|-----|-----|-----|--------|
| **Minimum** | 4+ cores | 16GB | None | CPU-only, small samples |
| **Recommended** | 8+ cores | 32GB | 8GB VRAM | Full experiments |
| **Optimal** | 12+ cores | 64GB | 12GB+ VRAM | Fast training |

## 📚 Dataset Information

### Webis-TLDR-17
- **Size**: 3.8M Reddit posts with summaries
- **Source**: Various subreddits
- **Format**: Post content + human-written TL;DR
- **Download**: Automatic via Hugging Face datasets
- **Size on disk**: ~3.1GB compressed

### Usage Rights
- Academic research use
- Citation required
- See dataset license for commercial use

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **Model support**: Add other LLMs (Llama, Phi, etc.)
- **Evaluation**: New metrics and human evaluation
- **Efficiency**: Further memory optimizations
- **Analysis**: Better visualization and insights

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```

## 📄 Citation

If you use this research in your work, please cite:

```bibtex
@misc{reddit-llm-summarization,
  title={Reddit LLM Summarization: Evaluating Lightweight Models for Social Media Content},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the open-source Mistral-7B model
- **Webis Research Group** for the TLDR-17 dataset  
- **Hugging Face** for the transformers library and model hosting
- **Reddit community** for providing the underlying content

---

<p align="center">
  <strong>🔬 Happy researching! 🚀</strong>
</p> 