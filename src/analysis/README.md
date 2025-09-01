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
python -m analysis.analyzer <file_path>

# Specify analysis mode explicitly
python -m analysis.analyzer <file_path> <mode>
```

### Analysis Modes
- `whole`: Analyze as a single continuous system (default)
- `2-agent`: Split analysis into two agents at question 1208

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
python -m analysis.analyzer output/qa_jobs/qa_results.jsonl whole

# Analyze with two-agent comparison
python -m analysis.analyzer output/qa_jobs/qa_results.jsonl 2-agent

# Analyze a log file (auto-detected format)
python -m analysis.analyzer logs/evaluation.log

# Get help (run without arguments)
python -m analysis.analyzer
```

## Output

### Console Output
- **Statistical Summary**: Count, average, median, min/max, standard deviation
- **Score Distribution**: Breakdown across performance ranges
- **Learning Analysis**: Trend detection, improvement metrics, R-squared values
- **Agent Comparison**: Performance comparison when using 2-agent mode

### Generated Files
- **Timestamped Plots**: `rouge_1_learning_progression_YYYYMMDD_HHMMSS.eps`
- **Comprehensive Visualizations**: 4-panel plots showing different analysis perspectives
- **EPS Format**: Vector graphics suitable for LaTeX document integration
