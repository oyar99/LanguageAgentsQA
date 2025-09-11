#!/usr/bin/env python3
"""
QA System Comparison Tool

This script compares two JSONL files with answers from the MuSiQue dataset and outputs
two JSONL files: one with questions that were answered correctly in file1 but incorrectly 
in file2, and another with questions answered correctly in file2 but incorrectly in file1.

The comparison is based on ROUGE-1 F1 scores, where a score >= threshold is considered correct.

Usage:
    python -m analysis.comparator -f1 <file1.jsonl> -f2 <file2.jsonl> [options]
    
Examples:
    python -m analysis.comparator -f1 results1.jsonl -f2 results2.jsonl
    python -m analysis.comparator -f1 results1.jsonl -f2 results2.jsonl -t 0.5
    python -m analysis.comparator -f1 results1.jsonl -f2 results2.jsonl -o comparison_output
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime
from logger.logger import Logger, MainProcessLogger
from models.question_answer import QuestionAnswer
from data.musique.musique import MuSiQue
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


def load_musique_dataset() -> Dict[str, QuestionAnswer]:
    """
    Load MuSiQue dataset using the project's existing Dataset class.

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
            self.dataset = 'musique'

    # Use the existing Dataset class
    args = MockArgs()
    dataset = MuSiQue(args)
    samples = dataset.read()

    result = {}
    for sample in samples:
        for qa in sample['sample']['qa']:
            if qa:
                result[qa['question_id']] = qa

    Logger().info(f"Loaded {len(result)} questions from MuSiQue dataset")
    return result


def process_jsonl_file(file_path: str, dataset_qa: Dict[str, QuestionAnswer]) -> Dict[str, Dict]:
    """
    Process JSONL file and extract question answers with their ROUGE-1 scores.

    Args:
        file_path: Path to the JSONL file
        dataset_qa: Dictionary mapping question IDs to QuestionAnswer objects

    Returns:
        Dictionary mapping question IDs to result info (answer, score, correct)
    """
    results = {}
    processed_count = 0
    matched_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                processed_count += 1

                # Extract question ID from custom_id field
                custom_id = data.get('custom_id', '')
                if not custom_id:
                    Logger().warn(f"Warning: No custom_id found on line {line_num} in {file_path}")
                    continue

                question_id = custom_id

                # Find corresponding entry in dataset
                if question_id not in dataset_qa:
                    Logger().warn(f"Warning: Question ID '{question_id}' not found in dataset (line {line_num})")
                    continue

                matched_count += 1

                # Extract predicted answer from response
                predicted_answer = data['response']['body']['choices'][0]['message']['content']

                # Get expected answers from QuestionAnswer object
                qa_obj = dataset_qa[question_id]
                all_expected_answers = qa_obj['answer']

                # Calculate ROUGE-1 score
                rouge1_score, _, _ = calculate_rouge1_score(all_expected_answers, predicted_answer)

                # Store result info
                results[question_id] = {
                    'predicted_answer': predicted_answer,
                    'expected_answers': all_expected_answers,
                    'rouge1_score': rouge1_score,
                    'original_data': data,
                    'question': qa_obj['question']
                }

            except json.JSONDecodeError as e:
                Logger().error(f"Error parsing JSON on line {line_num} in {file_path}: {e}")
                continue
            except KeyError as e:
                Logger().error(f"Error extracting data on line {line_num} in {file_path}: {e}")
                continue

    Logger().info(f"Processed {processed_count} entries from {file_path}")
    Logger().info(f"Matched {matched_count} entries with dataset")
    Logger().info(f"Successfully extracted {len(results)} question results")

    return results


def compare_results(results1: Dict[str, Dict], results2: Dict[str, Dict], 
                   threshold: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
    """
    Compare two sets of results and find questions where one performed better than the other.

    Args:
        results1: Results from first file
        results2: Results from second file  
        threshold: ROUGE-1 threshold for considering an answer correct

    Returns:
        Tuple of (file1_better, file2_better) lists containing question info
    """
    # Find common questions
    common_questions = set(results1.keys()) & set(results2.keys())
    Logger().info(f"Found {len(common_questions)} common questions between files")

    file1_better = []  # Correct in file1, incorrect in file2
    file2_better = []  # Correct in file2, incorrect in file1

    for question_id in common_questions:
        result1 = results1[question_id]
        result2 = results2[question_id]

        score1 = result1['rouge1_score']
        score2 = result2['rouge1_score']

        correct1 = score1 >= threshold
        correct2 = score2 >= threshold

        if correct1 and not correct2:
            # File1 got it right, file2 got it wrong
            comparison_info = {
                'question_id': question_id,
                'question': result1['question'],
                'expected_answers': result1['expected_answers'],
                'file1_answer': result1['predicted_answer'],
                'file1_score': score1,
                'file2_answer': result2['predicted_answer'],
                'file2_score': score2,
                'score_difference': score1 - score2,
                'file1_data': result1['original_data'],
                'file2_data': result2['original_data']
            }
            file1_better.append(comparison_info)

        elif correct2 and not correct1:
            # File2 got it right, file1 got it wrong
            comparison_info = {
                'question_id': question_id,
                'question': result2['question'],
                'expected_answers': result2['expected_answers'],
                'file1_answer': result1['predicted_answer'],
                'file1_score': score1,
                'file2_answer': result2['predicted_answer'],
                'file2_score': score2,
                'score_difference': score2 - score1,
                'file1_data': result1['original_data'],
                'file2_data': result2['original_data']
            }
            file2_better.append(comparison_info)

    return file1_better, file2_better


def save_comparison_results(file1_better: List[Dict], file2_better: List[Dict], 
                          output_prefix: str, file1_name: str, file2_name: str):
    """
    Save comparison results to JSONL files.

    Args:
        file1_better: Questions where file1 performed better
        file2_better: Questions where file2 performed better
        output_prefix: Prefix for output filenames
        file1_name: Name of first file (for labeling)
        file2_name: Name of second file (for labeling)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(project_root, "output", "misc")
    os.makedirs(output_dir, exist_ok=True)

    # File1 better results
    file1_better_path = os.path.join(output_dir, f"{output_prefix}_file1_better_{timestamp}.jsonl")
    with open(file1_better_path, 'w', encoding='utf-8') as f:
        for item in file1_better:
            json.dump(item, f)
            f.write('\n')

    # File2 better results  
    file2_better_path = os.path.join(output_dir, f"{output_prefix}_file2_better_{timestamp}.jsonl")
    with open(file2_better_path, 'w', encoding='utf-8') as f:
        for item in file2_better:
            json.dump(item, f)
            f.write('\n')

    Logger().info(f"Saved {len(file1_better)} questions where {file1_name} performed better to: {file1_better_path}")
    Logger().info(f"Saved {len(file2_better)} questions where {file2_name} performed better to: {file2_better_path}")

    # Also save a summary report
    summary_path = os.path.join(output_dir, f"{output_prefix}_comparison_summary_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("QA System Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"File 1: {file1_name}\n")
        f.write(f"File 2: {file2_name}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write(f"Questions where File 1 performed better: {len(file1_better)}\n")
        f.write(f"Questions where File 2 performed better: {len(file2_better)}\n")
        f.write(f"Total compared questions: {len(file1_better) + len(file2_better)}\n\n")
        
        if file1_better:
            f.write("File 1 Better - Score Differences:\n")
            score_diffs = [item['score_difference'] for item in file1_better]
            f.write(f"  Average score difference: {sum(score_diffs) / len(score_diffs):.4f}\n")
            f.write(f"  Max score difference: {max(score_diffs):.4f}\n")
            f.write(f"  Min score difference: {min(score_diffs):.4f}\n\n")
            
        if file2_better:
            f.write("File 2 Better - Score Differences:\n")
            score_diffs = [item['score_difference'] for item in file2_better]
            f.write(f"  Average score difference: {sum(score_diffs) / len(score_diffs):.4f}\n")
            f.write(f"  Max score difference: {max(score_diffs):.4f}\n")
            f.write(f"  Min score difference: {min(score_diffs):.4f}\n\n")

    Logger().info(f"Saved comparison summary to: {summary_path}")


def generate_detailed_report(file1_better: List[Dict], file2_better: List[Dict], 
                           threshold: float, file1_name: str, file2_name: str):
    """
    Generate a detailed comparison report.

    Args:
        file1_better: Questions where file1 performed better
        file2_better: Questions where file2 performed better
        threshold: ROUGE-1 threshold used
        file1_name: Name of first file
        file2_name: Name of second file
    """
    Logger().info("=" * 80)
    Logger().info("QA SYSTEM COMPARISON RESULTS")
    Logger().info("=" * 80)
    
    Logger().info(f"File 1: {file1_name}")
    Logger().info(f"File 2: {file2_name}")
    Logger().info(f"ROUGE-1 Threshold: {threshold}")
    Logger().info(f"Total differential questions: {len(file1_better) + len(file2_better)}")
    
    Logger().info("\n" + "-" * 50)
    Logger().info(f"QUESTIONS WHERE FILE 1 PERFORMED BETTER: {len(file1_better)}")
    Logger().info("-" * 50)
    
    if file1_better:
        score_diffs = [item['score_difference'] for item in file1_better]
        Logger().info(f"Average score difference: {sum(score_diffs) / len(score_diffs):.4f}")
        Logger().info(f"Max score difference: {max(score_diffs):.4f}")
        Logger().info(f"Min score difference: {min(score_diffs):.4f}")
        
        # Show top 3 examples
        sorted_by_diff = sorted(file1_better, key=lambda x: x['score_difference'], reverse=True)
        Logger().info("\nTop 3 examples (largest score differences):")
        for i, item in enumerate(sorted_by_diff[:3], 1):
            Logger().info(f"\n{i}. Question ID: {item['question_id']}")
            Logger().info(f"   Question: {item['question'][:100]}...")
            Logger().info(f"   Expected: {item['expected_answers'][0]}")
            Logger().info(f"   File1 answer: {item['file1_answer']}")
            Logger().info(f"   File2 answer: {item['file2_answer']}")
            Logger().info(f"   Scores: File1={item['file1_score']:.4f}, File2={item['file2_score']:.4f}")
    
    Logger().info("\n" + "-" * 50)
    Logger().info(f"QUESTIONS WHERE FILE 2 PERFORMED BETTER: {len(file2_better)}")
    Logger().info("-" * 50)
    
    if file2_better:
        score_diffs = [item['score_difference'] for item in file2_better]
        Logger().info(f"Average score difference: {sum(score_diffs) / len(score_diffs):.4f}")
        Logger().info(f"Max score difference: {max(score_diffs):.4f}")
        Logger().info(f"Min score difference: {min(score_diffs):.4f}")
        
        # Show top 3 examples
        sorted_by_diff = sorted(file2_better, key=lambda x: x['score_difference'], reverse=True)
        Logger().info("\nTop 3 examples (largest score differences):")
        for i, item in enumerate(sorted_by_diff[:3], 1):
            Logger().info(f"\n{i}. Question ID: {item['question_id']}")
            Logger().info(f"   Question: {item['question'][:100]}...")
            Logger().info(f"   Expected: {item['expected_answers'][0]}")
            Logger().info(f"   File1 answer: {item['file1_answer']}")
            Logger().info(f"   File2 answer: {item['file2_answer']}")
            Logger().info(f"   Scores: File1={item['file1_score']:.4f}, File2={item['file2_score']:.4f}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for QA system comparison.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='qa-comparator',
        description='Compare two JSONL files with QA results from MuSiQue dataset'
    )

    parser.add_argument('-f1', '--file1', type=str, required=True,
                        help='Path to the first JSONL file (required)')

    parser.add_argument('-f2', '--file2', type=str, required=True,
                        help='Path to the second JSONL file (required)')

    parser.add_argument('-t', '--threshold', type=float, default=0.3,
                        help='ROUGE-1 threshold for considering answer correct (default: 0.3)')

    parser.add_argument('-o', '--output', type=str, default='qa_comparison',
                        help='Output filename prefix (default: qa_comparison)')

    return parser.parse_args()


def main():
    """
    Main function to process command line arguments and run QA comparison.
    """
    args = parse_args()

    file1_path = args.file1
    file2_path = args.file2
    threshold = args.threshold
    output_prefix = args.output

    # Validate input files
    if not os.path.exists(file1_path):
        Logger().error(f"Error: File 1 does not exist: {file1_path}")
        return

    if not os.path.exists(file2_path):
        Logger().error(f"Error: File 2 does not exist: {file2_path}")
        return

    if not file1_path.endswith('.jsonl'):
        Logger().error(f"Error: File 1 must be a JSONL file: {file1_path}")
        return

    if not file2_path.endswith('.jsonl'):
        Logger().error(f"Error: File 2 must be a JSONL file: {file2_path}")
        return

    # Extract file names for labeling
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)

    Logger().info(f"Comparing QA results from MuSiQue dataset")
    Logger().info(f"File 1: {file1_name}")
    Logger().info(f"File 2: {file2_name}")
    Logger().info(f"ROUGE-1 threshold: {threshold}")
    Logger().info(f"Output prefix: {output_prefix}")

    # Load MuSiQue dataset
    Logger().info("Loading MuSiQue dataset...")
    dataset_qa = load_musique_dataset()
    if not dataset_qa:
        Logger().error("Error: Could not load MuSiQue dataset")
        return

    # Process both files
    Logger().info("Processing first file...")
    results1 = process_jsonl_file(file1_path, dataset_qa)
    if not results1:
        Logger().error("Error: Could not process first file")
        return

    Logger().info("Processing second file...")
    results2 = process_jsonl_file(file2_path, dataset_qa)
    if not results2:
        Logger().error("Error: Could not process second file")
        return

    # Compare results
    Logger().info("Comparing results...")
    file1_better, file2_better = compare_results(results1, results2, threshold)

    # Save results
    Logger().info("Saving comparison results...")
    save_comparison_results(file1_better, file2_better, output_prefix, file1_name, file2_name)

    # Generate detailed report
    generate_detailed_report(file1_better, file2_better, threshold, file1_name, file2_name)

    Logger().info("QA comparison completed successfully!")


if __name__ == "__main__":
    Logger().info("Starting QA comparison")
    main()
    MainProcessLogger().shutdown()
