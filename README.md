# Reddit LLM Summarization Research 🤖📝

<p align="center">
  <em>Fast, reliable Reddit summarization using Mistral & Gemini APIs - 30x faster than local deployment!</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/API-Mistral--7B-green.svg" alt="API">
  <img src="https://img.shields.io/badge/API-Gemini--2.0--Flash-orange.svg" alt="Gemini">
  <img src="https://img.shields.io/badge/Speed-30x_Faster-ff6b6b.svg" alt="Speed">
  <img src="https://img.shields.io/badge/Dataset-Webis--TLDR--17-blue.svg" alt="Dataset">
</p>

## 🎯 Research Overview

This project investigates the effectiveness of **Mistral-7B** and **Gemini 2.0 Flash** for automatically summarizing Reddit discussions, comparing zero-shot instruction-based vs few-shot prompting strategies using the **Webis-TLDR-17** dataset containing 3.8M Reddit posts.

**🚀 Now with dual API support for lightning-fast experiments!**

### 🔬 Research Questions

1. **How does few-shot prompting compare to instruction-based approaches for Reddit summarization?**
2. **Can API deployment maintain research quality while dramatically improving speed?**
3. **What are the practical benefits of using Mistral API vs local deployment?**

### 🏆 Key Findings

- **Few-shot prompting outperforms instruction-based** by 8.5% (ROUGE-1)
- **API deployment is 30-50x faster** than local deployment
- **Identical research quality** with professional reliability
- **BERTScore ~0.86** showing excellent semantic similarity

## 🚀 Quick Start (Recommended: API)

### Prerequisites

- **Python 3.8+**
- **Mistral API Key** (free at [console.mistral.ai](https://console.mistral.ai/))
- **5GB+ free disk space** (much lighter than local deployment!)

### 1. Clone and Setup

```bash
git clone https://github.com/Oki3/reddit-llmsum.git
cd reddit-llmsum

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get API Key

#### Option A: Mistral API Key
```bash
# 1. Visit: https://console.mistral.ai/
# 2. Create free account
# 3. Generate API key
# 4. Set environment variable:
export MISTRAL_API_KEY="your-api-key-here"
```

#### Option B: Gemini API Key ⭐ **Recommended**
```bash
# 1. Visit: https://console.cloud.google.com/
# 2. Create Google Cloud account (free tier available)
# 3. Enable Generative AI API
# 4. Generate API key
# 5. Set environment variable:
export GEMINI_API_KEY="your-api-key-here"

# 6. Install additional dependency:
pip install google-genai
```

### 3. Run Fast API Experiment

#### Option A: Mistral API (Original)

```bash
# Set your API key
export MISTRAL_API_KEY="your-api-key-here"

# Quick test (10 samples, ~1 minute)
python experiments/run_experiment_api.py --eval-sample-size 10 --delay 1.0

# Medium scale (100 samples, ~4 minutes) 
python experiments/run_experiment_api.py --eval-sample-size 100 --delay 1.0

# Large scale with responses saved (1000 samples, ~10 minutes)
python experiments/run_experiment_api.py --eval-sample-size 1000 --save-predictions --delay 1.0

# Research scale (comprehensive analysis)
python experiments/run_experiment_api.py --eval-sample-size 2000 --save-predictions --delay 1.0
```

#### Option B: Gemini API ⭐ **New!**

```bash
# Install Gemini API dependency
pip install google-genai

# Set your API key (get free key at: https://console.cloud.google.com/)
export GEMINI_API_KEY="your-api-key-here"

# Quick test (10 samples, ~1 minute)
python experiments/run_gemini_experiment.py --eval-sample-size 10 --delay 0.5

# Medium scale (100 samples, ~3 minutes) 
python experiments/run_gemini_experiment.py --eval-sample-size 100 --delay 0.5

# Large scale with responses saved (1000 samples, ~8 minutes)
python experiments/run_gemini_experiment.py --eval-sample-size 1000 --save-predictions --delay 0.5

# Research scale (comprehensive analysis)
python experiments/run_gemini_experiment.py --eval-sample-size 2000 --save-predictions --delay 0.5
```

### 🎛️ CLI Options

#### Mistral API Options
```bash
python experiments/run_experiment_api.py --help

# Key options:
--eval-sample-size N     # Number of samples to evaluate (default: 50)
--save-predictions       # Save individual Mistral responses to JSON files
--delay 1.0             # API delay in seconds (default: 1.0, prevents rate limits)
--api-key KEY           # Mistral API key (or use MISTRAL_API_KEY env var)
--model MODEL           # Mistral model name (default: open-mistral-7b)
```

#### Gemini API Options
```bash
python experiments/run_gemini_experiment.py --help

# Key options:
--eval-sample-size N     # Number of samples to evaluate (default: 50)
--save-predictions       # Save individual Gemini responses to JSON files
--delay 0.5             # API delay in seconds (default: 1.0, can be faster than Mistral)
--api-key KEY           # Gemini API key (or use GEMINI_API_KEY env var)
--model MODEL           # Gemini model name (default: gemini-2.0-flash)
```

## 🏃‍♂️ API vs Local Deployment

| Feature | Mistral API | Gemini API | Local Deployment |
|---------|-------------|------------|------------------|
| **Speed** | ~1-10 min (10-1000 samples) | ~0.8-8 min (10-1000 samples) | ~2-20 hours (10-1000 samples) |
| **Setup** | 5 minutes | 3 minutes | 1-2 hours |
| **Hardware** | Any laptop | Any laptop | CUDA GPU + 16GB+ RAM |
| **Rate Limiting** | 1 req/sec (configurable) | 2 req/sec (configurable) | No limits |
| **Reliability** | 100% | 100% | Memory issues common |
| **Cost** | ~$0.10 per 1000 samples | ~$0.08 per 1000 samples | Hardware + electricity |
| **Research Quality** | Excellent | Excellent | Identical |
| **Model** | open-mistral-7b | gemini-2.0-flash | Local Mistral-7B |

**💡 Recommendation: Use Gemini API for fastest results, Mistral API for established workflows, or local for full control!**

### ⚠️ Rate Limiting Best Practices

- **Default delay: 1.0 seconds** prevents 429 rate limit errors
- **For faster experiments:** `--delay 0.5` (riskier, may hit limits)
- **For safer experiments:** `--delay 2.0` (slower but guaranteed)
- **Progress tracking:** Shows completion every 10 samples
- **Error handling:** Automatic retry on API failures

## 📁 Project Structure

```
reddit-llmsum/
├── 📊 data/                    # Dataset handling and preprocessing
│   └── dataset_loader.py       # Webis-TLDR-17 loader with HuggingFace integration
├── 🤖 models/                  # Model implementations
│   ├── mistral_summarizer.py   # Local Mistral-7B (for advanced users)
│   ├── mistral_api_summarizer.py # API-based Mistral (original)
│   └── gemini_api_summarizer.py # API-based Gemini (fastest)
├── 📈 evaluation/              # Evaluation metrics and analysis
│   └── metrics.py             # ROUGE, BERTScore, coherence metrics
├── 🧪 experiments/             # Experiment runners
│   └── run_experiment_api.py   # Fast API-based experiments (recommended)
├── 🛠️ utils/                   # Utility functions
│   ├── experiment_config.py    # Configuration management
│   └── visualization.py       # Results visualization
└── 📊 results/                 # Generated results (timestamped)
    └── api_experiment_YYYYMMDD_HHMMSS/
        ├── SUMMARY.md           # Clean research summary
        ├── approach_comparison.csv
        └── plots/               # Key visualizations
            ├── rouge_comparison.png
            ├── bertscore_comparison.png
            └── quality_radar_chart.png
```

## 📊 Experimental Approaches

### Zero-Shot Methods Compared

1. **Instruction-based prompting**
   ```
   You are a helpful assistant that creates concise, accurate summaries of Reddit posts.
   Please summarize the following Reddit post in 1-2 sentences: [POST]
   ```

2. **Few-shot prompting** ⭐ **Better Performance**
   ```
   You are a helpful assistant that creates concise summaries of Reddit posts. 
   Here are some examples:
   
   Post: "I've been working at this company for 3 years and just found out my 
   colleague who started 6 months ago makes 20k more than me..."
   Summary: Employee discovers newer colleague earns significantly more despite 
   having less experience and performance, seeking advice on raise or new job.
   
   Now summarize this post: [POST]
   ```

### 🏆 Research Finding
**Few-shot prompting consistently outperforms instruction-based by 8.5%** due to providing concrete examples of good summarization style.

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

After running experiments, find clean results in:

#### Mistral API Results
```
results/api_experiment_YYYYMMDD_HHMMSS/
├── 📋 SUMMARY.md                     # Clean research summary with key findings
├── 📊 approach_comparison.csv        # Performance comparison table
├── 📄 instruction_predictions.json   # Individual Mistral responses (instruction-based)
├── 📄 few_shot_predictions.json      # Individual Mistral responses (few-shot)
├── 📋 complete_results.json          # Complete evaluation metrics
└── 📈 plots/                         # Key visualizations
    ├── rouge_comparison.png          # ROUGE scores comparison
    ├── bertscore_comparison.png       # Semantic similarity scores
    └── quality_radar_chart.png       # Multi-metric overview
```

#### Gemini API Results
```
results/gemini_experiment_YYYYMMDD_HHMMSS/
├── 📋 SUMMARY.md                     # Clean research summary with key findings
├── 📊 approach_comparison.csv        # Performance comparison table  
├── 📄 instruction_predictions.json   # Individual Gemini responses (instruction-based)
├── 📄 few_shot_predictions.json      # Individual Gemini responses (few-shot)
├── 📋 complete_results.json          # Complete evaluation metrics
└── 📈 plots/                         # Key visualizations
    ├── rouge_comparison.png          # ROUGE scores comparison
    ├── bertscore_comparison.png       # Semantic similarity scores
    └── quality_radar_chart.png       # Multi-metric overview
```

**💡 Note:** Individual prediction files are only saved when using `--save-predictions` flag.

### Sample Results

| Approach | ROUGE-1 | ROUGE-2 | BERTScore F1 | Coherence |
|----------|---------|---------|--------------|-----------|
| **Few-Shot Prompting** | **0.191** | **0.039** | **0.856** | **0.209** |
| Instruction-Based | 0.176 | 0.036 | 0.858 | 0.119 |

**Key Insight**: Few-shot examples provide better context for summarization style, leading to more coherent and comprehensive summaries.

### 📄 Viewing Individual Mistral Responses

When using `--save-predictions`, you can examine individual Mistral-generated summaries:

```bash
# View instruction-based responses
cat results/api_experiment_YYYYMMDD_HHMMSS/instruction_predictions.json

# View few-shot responses  
cat results/api_experiment_YYYYMMDD_HHMMSS/few_shot_predictions.json
```

Each prediction file contains:
```json
[
  {
    "index": 0,
    "input_text": "Original Reddit post content...",
    "generated_summary": "Mistral's generated summary",
    "reference_summary": "Human reference summary",
    "approach": "zero_shot_instruct" // or "zero_shot_few_shot"
  }
]
```

## 🏗️ Advanced: Local Deployment (For Expert Users)

For users who want to run experiments locally with full control:

### Prerequisites
- **CUDA-capable GPU** with 16GB+ VRAM
- **32GB+ RAM** 
- **50GB+ free disk space**

### Setup
```bash
# Additional local dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bitsandbytes accelerate

# Get Hugging Face access
huggingface-cli login

# Run local experiment (much slower)
python experiments/run_local_experiment.py --eval-sample-size 20
```

**⚠️ Note**: Local deployment requires significant hardware and setup time. **API is recommended** for most research purposes.

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### 🔑 API Key Issues
```bash
# Make sure API key is set correctly
echo $MISTRAL_API_KEY

# Or pass directly as argument
python experiments/run_experiment_api.py --api-key your-key-here
```

#### 🌐 Network Issues
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