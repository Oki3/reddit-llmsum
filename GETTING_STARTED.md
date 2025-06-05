# Getting Started with Reddit LLM Summarization Research

This guide will help you set up and run the Reddit summarization research project using Mistral-7B.

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Verify Setup

```bash
# Test the project setup (will fail on imports until dependencies are installed)
python test_setup.py
```

### 3. Download Dataset

Download the Webis-TLDR-17 dataset:
- Visit: https://webis.de/data/webis-tldr-17.html
- Download the dataset files
- Extract to `data/webis-tldr-17/`

The dataset should have this structure:
```
data/webis-tldr-17/
├── corpus-webis-tldr-17.json
└── (other dataset files)
```

### 4. Run Analysis Demo

```bash
# Run the analysis example (works without dataset/models)
python notebooks/analysis_example.py
```

### 5. Run Experiments

```bash
# Quick test (small sample, no fine-tuning)
python experiments/run_experiment.py --eval-sample-size 10 --no-fine-tuning

# Zero-shot experiments only
python experiments/run_experiment.py --no-fine-tuning

# Full experiment (requires GPU for fine-tuning)
python experiments/run_experiment.py
```

## Project Structure

```
reddit-llmsum/
├── data/                   # Dataset handling
│   ├── dataset_loader.py   # Webis-TLDR-17 loader
│   └── webis-tldr-17/     # Dataset files (download required)
├── models/                 # Model implementations
│   └── mistral_summarizer.py  # Mistral-7B wrapper
├── evaluation/            # Evaluation metrics
│   └── metrics.py         # ROUGE, BERTScore, coherence
├── experiments/           # Experiment runners
│   └── run_experiment.py  # Main experiment script
├── utils/                 # Utility functions
│   ├── experiment_config.py   # Configuration management
│   └── visualization.py   # Plotting and analysis
├── notebooks/             # Analysis examples
│   └── analysis_example.py    # Demo script
└── results/               # Generated results (created during experiments)
```

## Research Questions

This project addresses three key research questions:

1. **Can a small open-source LLM produce coherent and accurate summaries of social media discussions?**
2. **Does fine-tuning the LLM on domain-specific data improve summarization performance compared to zero-shot or few-shot prompting?**
3. **What are the trade-offs in using a lightweight local LLM for this task in terms of performance and accuracy?**

## Experimental Approaches

### Zero-Shot Approaches
- **Instruction-based prompting**: Direct instructions for summarization
- **Few-shot prompting**: Examples provided in the prompt

### Fine-Tuning Approach
- Fine-tune Mistral-7B on Reddit summarization data
- Compare against zero-shot baselines

## Evaluation Metrics

### Automatic Metrics
- **ROUGE scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **BERTScore**: Semantic similarity measurement
- **Coherence**: Text connectivity analysis
- **Length metrics**: Compression ratios and length differences

### Optional LLM-as-Evaluator
- Accuracy assessment
- Coherence evaluation
- Fluency scoring

## Hardware Requirements

### Minimum (CPU-only)
- 16GB RAM
- Can run zero-shot experiments with small samples
- Limited fine-tuning capability

### Recommended (GPU)
- NVIDIA GPU with 12GB+ VRAM (RTX 3080/4080 or better)
- 32GB+ system RAM
- Enables full fine-tuning experiments

### Memory-Efficient Options
- Use 8-bit quantization (`load_in_8bit=True`)
- Reduce batch sizes
- Use gradient accumulation

## Configuration

The project uses a flexible configuration system. You can:

1. **Use default settings**:
```python
from utils.experiment_config import ExperimentConfig
config = ExperimentConfig()
```

2. **Use predefined configurations**:
```python
from utils.experiment_config import ExperimentConfigs
config = ExperimentConfigs.quick_test()  # For testing
config = ExperimentConfigs.memory_efficient()  # For limited resources
```

3. **Customize parameters**:
```python
config = ExperimentConfig()
config.eval_sample_size = 200
config.num_epochs = 5
config.use_llm_evaluator = True
```

## Results and Analysis

Experiment results are saved in timestamped directories:
```
results/experiment_YYYYMMDD_HHMMSS/
├── complete_results.json          # All metrics and results
├── analysis_report.md             # Human-readable report
├── approach_comparison.csv        # Comparison table
├── experiment_config.json         # Configuration used
├── plots/                         # Visualizations
│   ├── rouge_comparison.png
│   ├── bertscore_comparison.png
│   ├── quality_radar_chart.png
│   └── performance_overview.png
└── predictions/                   # Generated summaries
    ├── zero_shot_instruct_predictions.json
    ├── zero_shot_few_shot_predictions.json
    └── fine_tuned_predictions.json
```

## Troubleshooting

### Memory Issues
- Reduce `eval_sample_size` and `train_batch_size`
- Enable `load_in_8bit=True`
- Increase `gradient_accumulation_steps`

### CUDA/GPU Issues
- Install correct PyTorch version for your CUDA version
- Set `device="cpu"` for CPU-only inference
- Use smaller models if needed

### Dataset Issues
- Ensure dataset is downloaded and extracted correctly
- Check file paths in configuration
- Verify JSON format is correct

### Import Errors
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Install project in development mode: `pip install -e .`

## Contributing

To contribute to this research:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Run tests: `python test_setup.py`
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{reddit-llm-summarization,
  title={Summarizing Social Media Discussions using Lightweight LLMs},
  author={Research Team},
  year={2024},
  url={https://github.com/example/reddit-llm-summarization}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the analysis examples in `notebooks/` 