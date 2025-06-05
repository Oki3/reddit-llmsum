#!/usr/bin/env python3
"""
Setup script for Reddit LLM Summarization Research.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reddit-llm-summarization",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Summarizing Reddit discussions using lightweight LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/reddit-llm-summarization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "reddit-summarize=experiments.run_experiment:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 