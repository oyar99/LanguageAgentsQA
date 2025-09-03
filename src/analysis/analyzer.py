#!/usr/bin/env python3
"""
QA System Analysis Tool

This script analyzes QA system performance with ROUGE-1 score evaluation and generates
learning progression visualizations. It supports both two-agent analysis (split at
JSONL file midpoint) and whole system analysis modes. The modular design allows 
for easy extension to support additional analysis metrics.

Usage:
    python -m analysis.analyzer <jsonl_file_path> <dataset_name> [mode] [limit]
    
Arguments:
    jsonl_file_path: Path to the JSONL file containing QA results
    dataset_name: Name of the dataset ('musique', 'locomo', 'hotpot', 'twowikimultihopqa')
    mode: Analysis mode - 'whole' (default) or '2-agent'
    limit: Optional limit on number of scores to process (processes all if not specified)

Examples:
    python -m analysis.analyzer /path/to/results.jsonl musique
    python -m analysis.analyzer /path/to/results.jsonl locomo whole
    python -m analysis.analyzer /path/to/results.jsonl musique 2-agent
    python -m analysis.analyzer /path/to/results.jsonl locomo 2-agent 100
"""

import json
import statistics
import os
import sys
from typing import Dict, List
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from logger.logger import Logger, MainProcessLogger
from models.question_answer import QuestionAnswer
from data.musique.musique import MuSiQue
from data.locomo.locomo import Locomo
from data.hotpot.hotpot import Hotpot
from data.twowikimultihopqa.two_wiki import TwoWiki
from evaluator.rogue_evaluator import rouge_score


def calculate_rouge1_score(expected_answers, actual_answer):
    """
    Calculate ROUGE-1 F1, precision, and recall using the exact same method as the evaluator.

    Args:
        expected_answers (list): List of possible correct answers
        actual_answer (str): The predicted answer

    Returns:
        tuple: (f1_score, precision, recall)
    """
    scores = rouge_score(expected_answers, actual_answer)
    # Return only ROUGE-1 scores
    return scores[0]


def load_dataset(dataset_name: str = "musique") -> Dict[str, QuestionAnswer]:
    """
    Load dataset using the project's existing Dataset classes.

    Args:
        dataset_name: Name of the dataset to load ('musique', 'locomo', 'hotpot', 'twowiki')

    Returns:
        Dictionary mapping question IDs to QuestionAnswer objects
    """
    class MockArgs:  # pylint: disable=too-few-public-methods
        """
        Mock args object for dataset initialization 
        """

        def __init__(self):
            self.conversation = None
            self.questions = None
            self.category = None
            self.shuffle = False
            self.limit = None  # No limit
            self.model = None  # Required for prompt selection

    # Map dataset names to classes
    dataset_classes = {
        'musique': MuSiQue,
        'locomo': Locomo,
        'hotpot': Hotpot,
        'twowiki': TwoWiki
    }

    if dataset_name.lower() not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Supported datasets: {list(dataset_classes.keys())}")

    dataset_class = dataset_classes[dataset_name.lower()]

    # Use the existing Dataset class
    args = MockArgs()
    dataset = dataset_class(args)
    samples = dataset.read()

    result = {}
    for sample in samples:
        for qa in sample['sample']['qa']:
            if qa:
                result[qa['question_id']] = qa

    Logger().info(
        f"Loaded {len(result)} questions from {dataset.name} dataset")
    return result


def calculate_rolling_averages(scores, window_size):
    """Calculate rolling averages for the scores."""
    if len(scores) < window_size:
        return list(range(1, len(scores) + 1)), scores, scores

    rolling_avg = []
    cumulative_avg = []

    for i in range(len(scores)):
        # Rolling average
        start_idx = max(0, i - window_size + 1)
        window_scores = scores[start_idx:i + 1]
        rolling_avg.append(sum(window_scores) / len(window_scores))

        # Cumulative average
        cumulative_scores = scores[:i + 1]
        cumulative_avg.append(sum(cumulative_scores) / len(cumulative_scores))

    question_numbers = list(range(1, len(scores) + 1))
    return question_numbers, rolling_avg, cumulative_avg


def analyze_learning_metrics(scores):  # pylint: disable=too-many-locals
    """Analyze learning progression metrics."""
    if len(scores) < 4:
        return {"error": "Not enough scores to analyze learning progression"}

    # Split into first and second half
    mid_point = len(scores) // 2
    early_scores = scores[:mid_point]
    late_scores = scores[mid_point:]

    early_avg = sum(early_scores) / len(early_scores)
    late_avg = sum(late_scores) / len(late_scores)

    improvement = late_avg - early_avg
    improvement_percent = (improvement / early_avg) * \
        100 if early_avg > 0 else 0

    # Calculate trend using linear regression
    x = np.array(range(len(scores)))
    y = np.array(scores)
    z = np.polyfit(x, y, 1)
    slope = z[0]

    # Calculate R-squared
    y_pred = np.polyval(z, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Determine trend strength
    if r_squared >= 0.7:
        trend_strength = "Strong"
    elif r_squared >= 0.3:
        trend_strength = "Moderate"
    else:
        trend_strength = "Weak"

    # Determine if learning is detected
    learning_detected = improvement > 0 and slope > 0 and r_squared > 0.1

    return {
        "early_avg": early_avg,
        "late_avg": late_avg,
        "improvement": improvement,
        "improvement_percent": improvement_percent,
        "slope": slope,
        "r_squared": r_squared,
        "trend_strength": trend_strength,
        "learning_detected": learning_detected
    }

# pylint: disable-next=too-many-locals,too-many-statements


def create_whole_system_plot(scores, save_path="rouge1_whole_system.eps",
                             score_type="ROUGE-1"):
    """
    Create a learning progression plot for the whole system (no agent split).

    Args:
        scores (list): List of scores in chronological order
        save_path (str): Path to save the plot image
        score_type (str): Type of score for labeling
    """
    if not scores:
        Logger().info("No scores to plot.")
        return

    # Calculate rolling averages
    question_numbers = list(range(1, len(scores) + 1))
    _, rolling_10, cumulative = calculate_rolling_averages(scores, 10)
    _, rolling_25, _ = calculate_rolling_averages(scores, 25)

    # Create the plot
    plt.figure(figsize=(16, 12))

    # Plot 1: Individual scores and rolling averages
    plt.subplot(2, 2, 1)
    plt.scatter(question_numbers, scores, alpha=0.4, s=15,
                color='skyblue', label='Individual Scores')
    plt.plot(question_numbers, rolling_10, color='blue',
             linewidth=2, label='Rolling Avg (10)')
    plt.plot(question_numbers, rolling_25, color='darkblue',
             linewidth=2, label='Rolling Avg (25)')
    plt.plot(question_numbers, cumulative, color='red',
             linewidth=2, label='Cumulative Avg')

    plt.xlabel('Question Number')
    plt.ylabel(f'{score_type} Score')
    plt.title(f'Learning Progression: Whole System ({score_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Cumulative average with trend line
    plt.subplot(2, 2, 2)
    plt.plot(question_numbers, cumulative, color='red',
             linewidth=2, label='Cumulative Average')

    # Add trend line
    if len(question_numbers) > 1:
        z = np.polyfit(question_numbers, cumulative, 1)
        p = np.poly1d(z)
        plt.plot(question_numbers, p(question_numbers), color='orange', linestyle='--',
                 linewidth=2, label=f'Trend (slope: {z[0]:.6f})')

    plt.xlabel('Question Number')
    plt.ylabel(f'Cumulative Average {score_type} Score')
    plt.title(f'Learning Trend Analysis: Whole System ({score_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Score distribution
    plt.subplot(2, 2, 3)
    plt.hist(scores, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel(f'{score_type} Score')
    plt.ylabel('Frequency')
    plt.title(f'{score_type} Score Distribution')
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (f"Mean: {np.mean(scores):.4f}\n"
                  f"Std: {np.std(scores):.4f}\n"
                  f"Median: {np.median(scores):.4f}")
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox={'boxstyle': 'round', 'facecolor': 'lightyellow', 'alpha': 0.8})

    # Plot 4: Performance over time (binned averages)
    plt.subplot(2, 2, 4)

    # Divide scores into bins for clearer trend visualization
    num_bins = min(20, len(scores) // 50) if len(scores) > 100 else 5
    bin_size = len(scores) // num_bins

    bin_centers = []
    bin_averages = []

    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(scores)
        bin_scores = scores[start_idx:end_idx]

        if bin_scores:
            bin_centers.append(start_idx + len(bin_scores) // 2)
            bin_averages.append(np.mean(bin_scores))

    if bin_centers:
        plt.plot(bin_centers, bin_averages, 'o-', linewidth=2,
                 markersize=8, color='green', label='Binned Averages')
        plt.xlabel('Question Number')
        plt.ylabel(f'Average {score_type} Score')
        plt.title(f'Performance Trend Over Time ({score_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    Logger().info(
        f"Whole system learning progression plot saved to: {save_path}")

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def create_learning_progression_plot(scores, save_path="rouge1_learning_progression.eps",
                                     score_type="ROUGE-1"):
    """
    Create a comprehensive learning progression plot with separate curves for two agents.
    Split point is calculated based on the actual JSONL file length (midpoint).

    Args:
        scores (list): List of scores in chronological order
        save_path (str): Path to save the plot image
        score_type (str): Type of score for labeling
        dataset_name (str): Name of the dataset for labeling purposes
    """
    if not scores:
        Logger().info("No scores to plot.")
        return

    # Calculate split point based on actual JSONL length, not dataset size
    # This allows for subset testing while maintaining proper agent split
    agent1_split = len(scores) // 2

    if len(scores) > agent1_split:
        agent1_scores = scores[:agent1_split]
        agent2_scores = scores[agent1_split:]

        # Create question numbers for each agent
        agent1_questions = list(range(1, len(agent1_scores) + 1))
        agent2_questions = list(
            range(agent1_split + 1, agent1_split + 1 + len(agent2_scores)))
    else:
        # If we don't have enough scores, treat all as agent 1
        agent1_scores = scores
        agent2_scores = []
        agent1_questions = list(range(1, len(agent1_scores) + 1))
        agent2_questions = []

    # Calculate rolling averages for each agent
    if agent1_scores:
        _, agent1_rolling_10, agent1_cumulative = calculate_rolling_averages(
            agent1_scores, 10)
        _, _, _ = calculate_rolling_averages(agent1_scores, 25)

    if agent2_scores:
        _, agent2_rolling_10, agent2_cumulative = calculate_rolling_averages(
            agent2_scores, 10)
        _, _, _ = calculate_rolling_averages(agent2_scores, 25)
        # Adjust question numbers for agent 2 rolling averages
        agent2_rolling_questions = list(
            range(agent1_split + 1, agent1_split + 1 + len(agent2_rolling_10)))

    # Create the plot
    plt.figure(figsize=(16, 12))

    # Plot 1: Individual scores and rolling averages for both agents
    plt.subplot(2, 2, 1)

    # Agent 1 data
    if agent1_scores:
        plt.scatter(agent1_questions, agent1_scores, alpha=0.4,
                    s=15, color='lightblue', label='Agent 1 Scores')
        plt.plot(agent1_questions, agent1_rolling_10, color='blue',
                 linewidth=2, label='Agent 1 Rolling Avg (10)')
        plt.plot(agent1_questions, agent1_cumulative, color='darkblue',
                 linewidth=2, label='Agent 1 Cumulative')

    # Agent 2 data
    if agent2_scores:
        plt.scatter(agent2_questions, agent2_scores, alpha=0.4,
                    s=15, color='lightcoral', label='Agent 2 Scores')
        plt.plot(agent2_rolling_questions, agent2_rolling_10,
                 color='red', linewidth=2, label='Agent 2 Rolling Avg (10)')
        plt.plot(agent2_questions, agent2_cumulative, color='darkred',
                 linewidth=2, label='Agent 2 Cumulative')

    # Add vertical line to separate agents
    if len(scores) > agent1_split:
        plt.axvline(x=agent1_split, color='gray', linestyle='--',
                    alpha=0.7, label='Agent Switch')

    plt.xlabel('Question Number')
    plt.ylabel(f'{score_type} Score')
    plt.title(f'Learning Progression: Two Agent System ({score_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Cumulative averages with trend lines for each agent
    plt.subplot(2, 2, 2)

    # Agent 1 trend
    if agent1_scores and len(agent1_questions) > 1:
        plt.plot(agent1_questions, agent1_cumulative, color='darkblue',
                 linewidth=2, label='Agent 1 Cumulative')
        z1 = np.polyfit(agent1_questions, agent1_cumulative, 1)
        p1 = np.poly1d(z1)
        plt.plot(agent1_questions, p1(agent1_questions), color='blue', linestyle='--',
                 linewidth=2, label=f'Agent 1 Trend (slope: {z1[0]:.6f})')

    # Agent 2 trend
    if agent2_scores and len(agent2_questions) > 1:
        plt.plot(agent2_questions, agent2_cumulative, color='darkred',
                 linewidth=2, label='Agent 2 Cumulative')
        z2 = np.polyfit(agent2_questions, agent2_cumulative, 1)
        p2 = np.poly1d(z2)
        plt.plot(agent2_questions, p2(agent2_questions), color='red', linestyle='--',
                 linewidth=2, label=f'Agent 2 Trend (slope: {z2[0]:.6f})')

    # Add vertical line to separate agents
    if len(scores) > agent1_split:
        plt.axvline(x=agent1_split, color='gray', linestyle='--',
                    alpha=0.7, label='Agent Switch')

    plt.xlabel('Question Number')
    plt.ylabel(f'Cumulative Average {score_type} Score')
    plt.title(f'Learning Trend Analysis: Two Agents ({score_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Agent comparison statistics
    plt.subplot(2, 2, 3)

    agent_stats = []
    if agent1_scores:
        agent_stats.append({
            'name': f'Agent 1\n(Q1-{len(agent1_scores)})',
            'avg': sum(agent1_scores) / len(agent1_scores),
            'std': statistics.stdev(agent1_scores) if len(agent1_scores) > 1 else 0,
            'count': len(agent1_scores)
        })

    if agent2_scores:
        agent_stats.append({
            'name': f'Agent 2\n(Q{agent1_split+1}-{agent1_split+len(agent2_scores)})',
            'avg': sum(agent2_scores) / len(agent2_scores),
            'std': statistics.stdev(agent2_scores) if len(agent2_scores) > 1 else 0,
            'count': len(agent2_scores)
        })

    if agent_stats:
        names = [stat['name'] for stat in agent_stats]
        averages = [stat['avg'] for stat in agent_stats]
        std_devs = [stat['std'] for stat in agent_stats]

        x_pos = range(len(names))
        colors = ['lightblue', 'lightcoral'][:len(names)]
        plt.bar(x_pos, averages, yerr=std_devs, alpha=0.7, color=colors,
                capsize=5, error_kw={'linewidth': 2})
        plt.xlabel('Agent')
        plt.ylabel(f'Average {score_type} Score')
        plt.title(f'Agent Performance Comparison ({score_type})')
        plt.xticks(x_pos, names)
        plt.grid(True, alpha=0.3)

        # Add performance comparison text
        if len(agent_stats) == 2:
            improvement = agent_stats[1]['avg'] - agent_stats[0]['avg']
            improvement_pct = (
                improvement / agent_stats[0]['avg']) * 100 if agent_stats[0]['avg'] > 0 else 0

            comparison_text = (f"Agent 2 vs Agent 1:\n"
                               f"{improvement:+.4f} ({improvement_pct:+.1f}%)")
            plt.text(0.05, 0.95, comparison_text, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox={'boxstyle': 'round', 'facecolor': 'lightyellow', 'alpha': 0.8})

    # Plot 4: Score distribution comparison
    plt.subplot(2, 2, 4)

    if agent1_scores and agent2_scores:
        # Create histograms for both agents
        plt.hist(agent1_scores, bins=20, alpha=0.6,
                 color='blue', label='Agent 1', density=True)
        plt.hist(agent2_scores, bins=20, alpha=0.6,
                 color='red', label='Agent 2', density=True)
        plt.xlabel(f'{score_type} Score')
        plt.ylabel('Density')
        plt.title(f'Score Distribution Comparison ({score_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    elif agent1_scores:
        plt.hist(agent1_scores, bins=20, alpha=0.7,
                 color='blue', label='Agent 1')
        plt.xlabel(f'{score_type} Score')
        plt.ylabel('Frequency')
        plt.title(f'Agent 1 {score_type} Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    Logger().info(f"Learning progression plot saved to: {save_path}")

    # Print agent performance summary
    Logger().info(f"\n{'=' * 60}")
    Logger().info("AGENT PERFORMANCE SUMMARY")
    Logger().info("=" * 60)

    if agent1_scores:
        Logger().info(f"Agent 1 (Questions 1-{len(agent1_scores)}):")
        Logger().info(
            f"  Average {score_type} Score: "
            f"{sum(agent1_scores)/len(agent1_scores):.6f}")
        Logger().info(
            f"  Standard Deviation: "
            f"{statistics.stdev(agent1_scores) if len(agent1_scores) > 1 else 0:.6f}")
        Logger().info(f"  Questions Answered: {len(agent1_scores)}")

    if agent2_scores:
        Logger().info(
            f"Agent 2 (Questions "
            f"{agent1_split+1}-{agent1_split+len(agent2_scores)}):")
        Logger().info(
            f"  Average {score_type} Score: "
            f"{sum(agent2_scores)/len(agent2_scores):.6f}")
        Logger().info(
            f"  Standard Deviation: "
            f"{statistics.stdev(agent2_scores) if len(agent2_scores) > 1 else 0:.6f}")
        Logger().info(f"  Questions Answered: {len(agent2_scores)}")

        if agent1_scores:
            improvement = (sum(agent2_scores)/len(agent2_scores)) - \
                (sum(agent1_scores)/len(agent1_scores))
            improvement_pct = (improvement / (sum(agent1_scores) /
                               len(agent1_scores))) * 100 if agent1_scores else 0
            Logger().info(
                f"  Improvement over Agent 1: {improvement:+.6f} ({improvement_pct:+.2f}%)")


def calculate_statistics(scores):
    """Calculate comprehensive statistics for the scores."""
    if not scores:
        return {}

    return {
        'count': len(scores),
        'average': sum(scores) / len(scores),
        'median': statistics.median(scores),
        'minimum': min(scores),
        'maximum': max(scores),
        'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
        'range': max(scores) - min(scores),
        'q1': np.percentile(scores, 25),
        'q3': np.percentile(scores, 75),
        'iqr': np.percentile(scores, 75) - np.percentile(scores, 25)
    }


def print_results(scores, stats, score_type="ROUGE-1"):
    """
    Print the results in a formatted way.

    Args:
        scores (list): List of scores
        stats (dict): Statistical measures
        score_type (str): Type of score ("ROUGE-1", "F1", etc.)
    """
    Logger().info("=" * 60)
    Logger().info(f"{score_type} SCORE ANALYSIS RESULTS")
    Logger().info("=" * 60)

    if not scores:
        Logger().info(f"No {score_type} scores found in the file.")
        return

    Logger().info(f"Total {score_type} scores found: {stats['count']}")
    Logger().info(f"Average {score_type} score: {stats['average']:.6f}")
    Logger().info(f"Median {score_type} score: {stats['median']:.6f}")
    Logger().info(f"Minimum {score_type} score: {stats['minimum']:.6f}")
    Logger().info(f"Maximum {score_type} score: {stats['maximum']:.6f}")
    Logger().info(f"Standard deviation: {stats['std_dev']:.6f}")

    # Distribution analysis
    Logger().info("\n" + "-" * 40)
    Logger().info("SCORE DISTRIBUTION")
    Logger().info("-" * 40)

    # Create score ranges
    ranges = [
        (0.0, 0.1, "Very Low (0.0-0.1)"),
        (0.1, 0.3, "Low (0.1-0.3)"),
        (0.3, 0.5, "Medium-Low (0.3-0.5)"),
        (0.5, 0.7, "Medium (0.5-0.7)"),
        (0.7, 0.9, "High (0.7-0.9)"),
        (0.9, 1.0, "Very High (0.9-1.0)")
    ]

    for low, high, label in ranges:
        count = sum(1 for score in scores if low <= score <= high)
        percentage = (count / len(scores)) * 100
        Logger().info(f"{label}: {count:4d} scores ({percentage:5.1f}%)")

    # Sample scores
    Logger().info("\n" + "-" * 40)
    Logger().info("SAMPLE SCORES")
    Logger().info("-" * 40)

    Logger().info(
        f"First 5 scores: {[f'{score:.6f}' for score in scores[:5]]}")
    if len(scores) > 5:
        Logger().info(
            f"Last 5 scores: {[f'{score:.6f}' for score in scores[-5:]]}")


def process_jsonl_file(file_path: str, dataset_name: str, limit: int = None) -> List[float]:
    """Process JSONL file and calculate ROUGE-1 scores against dataset.

    Args:
        file_path: Path to the JSONL file
        dataset_name: Name of the dataset to load
        limit: Optional maximum number of scores to process

    Returns:
        List of ROUGE-1 scores
    """
    # Load dataset
    dataset_qa = load_dataset(dataset_name)
    if not dataset_qa:
        Logger().error("Error: Could not load dataset")
        return []

    # Process JSONL file
    rouge1_scores = []
    processed_count = 0
    matched_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Stop processing if we've reached the limit
            if limit is not None and len(rouge1_scores) >= limit:
                break

            data = json.loads(line.strip())
            processed_count += 1

            # Extract question ID from custom_id field
            custom_id = data.get('custom_id', '')
            if not custom_id:
                Logger().warn(
                    f"Warning: No custom_id found on line {line_num}")
                continue

            # The custom_id should directly match the dataset ID
            question_id = custom_id

            # Find corresponding entry in dataset
            if question_id not in dataset_qa:
                Logger().warn(
                    f"Warning: Question ID '{question_id}' not found in dataset (line {line_num})")
                continue

            matched_count += 1

            # Extract predicted answer from response
            predicted_answer = data['response']['body']['choices'][0]['message']['content']

            # Get expected answers from QuestionAnswer object
            qa_obj = dataset_qa[question_id]
            # This is already a list with main answer + aliases
            all_expected_answers = qa_obj['answer']

            # Calculate ROUGE-1 score (takes max across all possible answers)
            rouge1_score, _, _ = calculate_rouge1_score(
                all_expected_answers, predicted_answer)
            rouge1_scores.append(rouge1_score)

    Logger().info(f"Processed {processed_count} entries from JSONL file")
    Logger().info(f"Matched {matched_count} entries with dataset")
    Logger().info(
        f"Calculated ROUGE-1 scores for {len(rouge1_scores)} questions")

    if limit is not None:
        Logger().info(f"Limited to {limit} scores as requested")

    return rouge1_scores


def parse_arguments():
    """
    Parse command line arguments for file path, dataset name, analysis mode, and optional limit.

    Returns:
        tuple: (filename, dataset_name, mode, limit) where mode is either 'whole' or '2-agent'
               and limit is an optional integer or None
    """
    # Check command line arguments
    if len(sys.argv) < 3:
        print(
            "Usage: python -m analysis.analyzer <file_path> <dataset_name> [mode] [limit]")
        print("  file_path: Path to the JSONL file to analyze")
        print(
            "  dataset_name: Name of the dataset ('musique', 'locomo', 'hotpot', 'twowiki')")
        print("  mode: 'whole' (default) or '2-agent' (split at JSONL file midpoint)")
        print("  limit: Optional maximum number of scores to process")
        print("  Examples:")
        print("    python -m analysis.analyzer output/qa_jobs/qa_results.jsonl musique")
        print(
            "    python -m analysis.analyzer output/qa_jobs/qa_results.jsonl locomo 2-agent")
        print(
            "    python -m analysis.analyzer output/qa_jobs/qa_results.jsonl musique 2-agent 100")
        sys.exit(1)

    filename = sys.argv[1]
    dataset_name = sys.argv[2]

    # Validate dataset name
    valid_datasets = ['musique', 'locomo', 'hotpot', 'twowiki']
    if dataset_name.lower() not in valid_datasets:
        print(f"Error: Invalid dataset name '{dataset_name}'. "
              f"Valid options: {', '.join(valid_datasets)}")
        sys.exit(1)

    # Parse mode argument (default to "whole")
    mode = "whole"
    if len(sys.argv) >= 4:
        mode_arg = sys.argv[3].lower()
        if mode_arg in ["2-agent", "2agent", "two-agent", "agent"]:
            mode = "2-agent"
        elif mode_arg in ["whole", "system", "single"]:
            mode = "whole"
        else:
            # Try to parse as a number (limit argument without mode)
            try:
                limit = int(sys.argv[3])
                return filename, dataset_name, mode, limit
            except ValueError:
                print(
                    f"Warning: Unknown mode '{sys.argv[3]}'. Using 'whole' mode.")
                mode = "whole"

    # Parse limit argument (optional)
    limit = None
    if len(sys.argv) >= 5:
        try:
            limit = int(sys.argv[4])
            if limit <= 0:
                print("Error: Limit must be a positive integer.")
                sys.exit(1)
        except ValueError:
            print("Error: Limit must be a valid integer.")
            sys.exit(1)
    elif len(sys.argv) == 4 and mode == "whole":
        # Check if the 4th argument is actually a limit
        try:
            limit = int(sys.argv[3])
            if limit <= 0:
                print("Error: Limit must be a positive integer.")
                sys.exit(1)
        except ValueError:
            pass  # It's actually the mode argument

    return filename, dataset_name, mode, limit


def process_file_and_get_scores(filename, dataset_name, limit=None):
    """
    Process the JSONL file and extract ROUGE-1 scores.

    Args:
        filename: Path to the JSONL file to process
        dataset_name: Name of the dataset to use
        limit: Optional maximum number of scores to process

    Returns:
        tuple: (scores_list, score_type) or (None, None) if processing fails
    """
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        print("Please check the file path and try again.")
        return None, None

    # Only process JSONL files
    if not filename.endswith('.jsonl'):
        print(f"Error: Only JSONL files are supported. Got: {filename}")
        print("Please provide a JSONL file path.")
        return None, None

    Logger().info(f"Processing JSONL file: {filename}")
    Logger().info("Calculating ROUGE-1 scores against dataset...")
    scores = process_jsonl_file(filename, dataset_name, limit)
    score_type = "ROUGE-1"

    if not scores:
        Logger().warn(f"No {score_type} scores found.")
        return None, None

    return scores, score_type


def perform_agent_analysis(scores, mode):
    """
    Perform agent-specific analysis for 2-agent mode.

    Args:
        scores: List of ROUGE-1 scores
        mode: Analysis mode ('2-agent' or 'whole')
        dataset_name: Name of the dataset (used for labeling)
    """
    # Agent-specific analysis only for 2-agent mode
    if mode == "2-agent":
        agent1_split = len(scores) // 2

        if len(scores) > agent1_split:
            Logger().info("\n" + "=" * 60)
            Logger().info("AGENT-SPECIFIC LEARNING ANALYSIS")
            Logger().info("=" * 60)

            agent1_scores = scores[:agent1_split]
            agent2_scores = scores[agent1_split:]

            Logger().info(
                f"Agent 1 Learning Analysis (Questions 1-{agent1_split}):")
            agent1_metrics = analyze_learning_metrics(agent1_scores)
            if 'error' not in agent1_metrics:
                Logger().info(
                    f"  Early avg: {agent1_metrics['early_avg']:.6f}")
                Logger().info(f"  Late avg: {agent1_metrics['late_avg']:.6f}")
                Logger().info(
                    f"  Improvement: {agent1_metrics['improvement']:+.6f} "
                    f"({agent1_metrics['improvement_percent']:+.2f}%)")
                Logger().info(
                    f"  Learning detected: "
                    f"{'Yes' if agent1_metrics['learning_detected'] else 'No'}")

            Logger().info(
                f"\nAgent 2 Learning Analysis (Questions {agent1_split + 1}-{len(scores)}):")
            agent2_metrics = analyze_learning_metrics(agent2_scores)
            if 'error' not in agent2_metrics:
                Logger().info(
                    f"  Early avg: {agent2_metrics['early_avg']:.6f}")
                Logger().info(f"  Late avg: {agent2_metrics['late_avg']:.6f}")
                Logger().info(
                    f"  Improvement: {agent2_metrics['improvement']:+.6f} "
                    f"({agent2_metrics['improvement_percent']:+.2f}%)")
                Logger().info(
                    f"  Learning detected: "
                    f"{'Yes' if agent2_metrics['learning_detected'] else 'No'}")
        else:
            Logger().info(f"\nNote: Dataset has only {len(scores)} questions, "
                          f"which is less than the calculated split point ({agent1_split}). "
                          f"Using whole dataset analysis only.")


def main():
    """
    Main function to process command line arguments and run ROUGE-1 analysis.

    Parses command line arguments for file path, dataset name, analysis mode, and optional limit,
    then processes the specified file and generates analysis results.
    """
    filename, dataset_name, mode, limit = parse_arguments()
    Logger().info(f"Dataset: {dataset_name}")
    Logger().info(f"Analysis mode: {mode}")
    if limit is not None:
        Logger().info(f"Processing limit: {limit}")

    scores, score_type = process_file_and_get_scores(
        filename, dataset_name, limit)
    if scores is None:
        return

    # Calculate statistics and print results
    stats = calculate_statistics(scores)
    print_results(scores, stats, score_type)

    # Create output directory if it doesn't exist
    # Go up from src/analysis to project root, then to output/misc
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(project_root, "output", "misc")
    os.makedirs(output_dir, exist_ok=True)

    # Create learning progression plot based on mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "2agent" if mode == "2-agent" else "whole"
    plot_filename = os.path.join(output_dir,
                                 f"{score_type.lower().replace('-', '_')}_learning_progression_"
                                 f"{dataset_name}_{mode_suffix}_{timestamp}.eps")

    if mode == "2-agent":
        create_learning_progression_plot(
            scores, plot_filename, score_type)
    else:
        # For whole system, create a simpler single-system plot
        create_whole_system_plot(scores, plot_filename, score_type)

    # Analyze learning metrics for the complete dataset
    Logger().info("\n" + "=" * 60)
    Logger().info("OVERALL LEARNING PROGRESSION ANALYSIS")
    Logger().info("=" * 60)
    learning_metrics = analyze_learning_metrics(scores)

    if 'error' not in learning_metrics:
        Logger().info(
            f"Early half average: {learning_metrics['early_avg']:.6f}")
        Logger().info(f"Late half average: {learning_metrics['late_avg']:.6f}")
        Logger().info(
            f"Improvement: {learning_metrics['improvement']:+.6f} "
            f"({learning_metrics['improvement_percent']:+.2f}%)")
        Logger().info(f"Trend slope: {learning_metrics['slope']:.8f}")
        Logger().info(f"R-squared: {learning_metrics['r_squared']:.6f}")
        Logger().info(f"Trend strength: {learning_metrics['trend_strength']}")
        Logger().info(
            f"Learning detected: "
            f"{'Yes' if learning_metrics['learning_detected'] else 'No'}")
    else:
        Logger().info(learning_metrics['error'])

    # Perform agent-specific analysis
    perform_agent_analysis(scores, mode)

    # Clean up matplotlib resources
    plt.close('all')


if __name__ == "__main__":
    Logger().info("Starting QA analysis")

    main()
    Logger().info("QA analysis completed successfully")

    MainProcessLogger().shutdown()
