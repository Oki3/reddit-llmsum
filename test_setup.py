#!/usr/bin/env python3
"""
Quick test script to verify project setup.
This script tests basic functionality without requiring the full dataset or models.
"""

import sys
import os
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data.dataset_loader import WebISTLDRDatasetLoader
        print("✓ Dataset loader import successful")
    except Exception as e:
        print(f"✗ Dataset loader import failed: {e}")
        return False
    
    try:
        from models.mistral_summarizer import MistralSummarizer
        print("✓ Mistral summarizer import successful")
    except Exception as e:
        print(f"✗ Mistral summarizer import failed: {e}")
        return False
    
    try:
        from evaluation.metrics import SummarizationEvaluator
        print("✓ Evaluation metrics import successful")
    except Exception as e:
        print(f"✗ Evaluation metrics import failed: {e}")
        return False
    
    try:
        from experiments.run_experiment import ExperimentRunner
        print("✓ Experiment runner import successful")
    except Exception as e:
        print(f"✗ Experiment runner import failed: {e}")
        return False
    
    try:
        from utils.experiment_config import ExperimentConfig
        print("✓ Experiment config import successful")
    except Exception as e:
        print(f"✗ Experiment config import failed: {e}")
        return False
    
    try:
        from utils.visualization import create_evaluation_plots
        print("✓ Visualization utils import successful")
    except Exception as e:
        print(f"✗ Visualization utils import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Test dataset loader initialization
        from data.dataset_loader import WebISTLDRDatasetLoader
        loader = WebISTLDRDatasetLoader("test_data_dir")
        print("✓ Dataset loader initialization successful")
    except Exception as e:
        print(f"✗ Dataset loader initialization failed: {e}")
        return False
    
    try:
        # Test evaluation with sample data
        from evaluation.metrics import SummarizationEvaluator
        evaluator = SummarizationEvaluator()
        
        sample_preds = ["Short summary.", "Another summary."]
        sample_refs = ["Reference summary.", "Another reference."]
        
        # Test ROUGE computation
        rouge_results = evaluator.compute_rouge_scores(sample_preds, sample_refs)
        assert 'rouge1' in rouge_results
        print("✓ ROUGE evaluation successful")
    except Exception as e:
        print(f"✗ ROUGE evaluation failed: {e}")
        return False
    
    try:
        # Test configuration
        from utils.experiment_config import ExperimentConfig
        config = ExperimentConfig()
        config.validate()
        print("✓ Configuration validation successful")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
    
    return True


def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data',
        'models', 
        'evaluation',
        'experiments',
        'utils',
        'notebooks'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing")
            all_exist = False
    
    return all_exist


def test_required_files():
    """Test that all required files exist."""
    print("\nTesting required files...")
    
    required_files = [
        'requirements.txt',
        'README.md',
        'setup.py',
        'data/__init__.py',
        'data/dataset_loader.py',
        'models/__init__.py',
        'models/mistral_summarizer.py',
        'evaluation/__init__.py',
        'evaluation/metrics.py',
        'experiments/__init__.py',
        'experiments/run_experiment.py',
        'utils/__init__.py',
        'utils/experiment_config.py',
        'utils/visualization.py'
    ]
    
    all_exist = True
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("Reddit LLM Summarization Project - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Required Files", test_required_files),
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The project setup is working correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download the Webis-TLDR-17 dataset")
        print("3. Run the analysis example: python notebooks/analysis_example.py")
        print("4. Run experiments: python experiments/run_experiment.py")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 