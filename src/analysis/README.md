# QA System Analysis Tool

This directory contains analysis tools for evaluating question-answering systems.

## Overview

The QA System Analyzer provides comprehensive analysis of question-answering performance, with support for both individual system analysis and multi-agent learning progression studies. The tool focuses on ROUGE-1 score evaluation with extensible architecture for future analysis types.

## Usage

### Basic Usage

```bash
# Run from the src directory using module syntax
cd src

# Analyze with whole system view (default)
python -m analysis.analyzer -f <file_path> -d <dataset_name>

# Specify analysis mode explicitly
python -m analysis.analyzer -f <file_path> -d <dataset_name> -m <mode>
```

### Command Line Arguments

#### Required Arguments
- `-f, --file`: Path to the JSONL file containing QA results (required)
- `-d, --dataset`: Name of the dataset - choices: `musique`, `locomo`, `hotpot`, `twowiki` (required)

#### Optional Arguments
- `-m, --mode`: Analysis mode - choices: `whole`, `2-agent` (default: `whole`)
  - `whole`: Analyze as a single continuous system
  - `2-agent`: Split analysis into two agents at the midpoint
- `-l, --limit`: Maximum number of scores to process (optional)
- `-h, --help`: Show help message and exit

### Analysis Modes
- `whole`: Analyze as a single continuous system (default)
- `2-agent`: Split analysis into two agents at question midpoint

### Input File Formats

#### JSONL Files (Preferred)

JSONL files containing QA results with this structure:
```json
{"custom_id": "question_id", "response": {"body": {"choices": [{"message": {"content": "answer"}}]}}}
```

#### Log Files (Legacy)

Text log files containing ROUGE-1 scores in this format:
```
ROUGE-1: 0.856
```

### Examples

```bash
# Analyze a JSONL file as whole system
python -m analysis.analyzer -f output/qa_jobs/qa_results.jsonl -d musique

# Analyze with two-agent comparison
python -m analysis.analyzer -f output/qa_jobs/qa_results.jsonl -d musique -m 2-agent

# Analyze with a limit on the number of scores processed
python -m analysis.analyzer -f output/qa_jobs/qa_results.jsonl -d locomo -m whole -l 100

# Analyze HotpotQA dataset with 2-agent mode
python -m analysis.analyzer -f output/qa_jobs/hotpot_results.jsonl -d hotpot -m 2-agent

# Get help
python -m analysis.analyzer -h
```

## Output

### Console Output
- **Statistical Summary**: Count, average, median, min/max, standard deviation
- **Score Distribution**: Breakdown across performance ranges
- **Learning Analysis**: Trend detection, improvement metrics, R-squared values
- **Agent Comparison**: Performance comparison when using 2-agent mode

### Generated Files
- **Timestamped Plots**: `rouge_1_learning_progression_<dataset>_<mode>_YYYYMMDD_HHMMSS.eps`
- **Comprehensive Visualizations**: 4-panel plots showing different analysis perspectives
- **EPS Format**: Vector graphics suitable for LaTeX document integration
