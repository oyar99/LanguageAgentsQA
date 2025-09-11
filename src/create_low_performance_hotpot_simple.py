#!/usr/bin/env python3
"""
HotpotQA Low Performance Dataset Creator

This script reads a JSONL file with QA system results, identifies questions where
ROUGE-1 < 0.9, and creates a new HotpotQA dataset containing only those poorly
performing questions in the original HotpotQA format.

Usage:
    python create_low_performance_hotpot.py -f <jsonl_file_path> [-t <threshold>] [-o <output_file>]
    
Examples:
    python create_low_performance_hotpot.py -f /path/to/results.jsonl
    python create_low_performance_hotpot.py -f /path/to/results.jsonl -t 0.85 -o low_performance_hotpot.json
"""

import argparse
import json
import os
from typing import Dict, List, Any
from data.hotpot.hotpot import Hotpot
from evaluator.rogue_evaluator import rouge_score


def calculate_rouge1_score(expected_answers, actual_answer):
    """
    Calculate ROUGE-1 F1 score using the exact same method as the evaluator.

    Args:
        expected_answers (list): List of possible correct answers
        actual_answer (str): The predicted answer

    Returns:
        float: ROUGE-1 F1 score
    """
    scores = rouge_score(expected_answers, actual_answer)
    return scores[0][0]  # Return ROUGE-1 F1 score


def load_hotpot_original_data():
    """
    Load the original HotpotQA dataset.
    
    Returns:
        dict: Dictionary mapping question IDs to full question objects
    """
    file_path = os.path.join("data", "hotpot", "hotpot_dev_distractor_v1.json")
    
    if not os.path.exists(file_path):
        print(f"ERROR: Original HotpotQA file not found at {file_path}")
        return {}
    
    question_map = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            question_map[sample['_id']] = sample
    
    print(f"Loaded {len(question_map)} questions from original HotpotQA dataset")
    return question_map


def load_dataset_qa():
    """
    Load dataset using the project's existing Dataset class.

    Returns:
        Dictionary mapping question IDs to QuestionAnswer objects
    """
    class MockArgs:
        def __init__(self):
            self.conversation = None
            self.questions = None
            self.category = None
            self.shuffle = False
            self.limit = None
            self.model = None

    args = MockArgs()
    dataset = Hotpot(args)
    samples = dataset.read()

    result = {}
    for sample in samples:
        for qa in sample['sample']['qa']:
            if qa:
                result[qa['question_id']] = qa

    print(f"Loaded {len(result)} questions from Hotpot dataset")
    return result


def analyze_jsonl_file(file_path: str, dataset_qa: Dict, threshold: float = 0.9) -> Dict[str, Any]:
    """
    Analyze JSONL file and identify questions with ROUGE-1 < threshold.

    Args:
        file_path: Path to the JSONL results file
        dataset_qa: Dictionary mapping question IDs to QuestionAnswer objects
        threshold: ROUGE-1 threshold (default: 0.9)

    Returns:
        Dictionary with analysis results and list of low-performing question IDs
    """
    low_performance_ids = []
    total_processed = 0
    total_matched = 0

    print(f"Analyzing JSONL file: {file_path}")
    print(f"Looking for questions with ROUGE-1 < {threshold}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line.strip())
            total_processed += 1

            # Extract question ID from custom_id field
            custom_id = data.get('custom_id', '')
            if not custom_id:
                print(f"WARNING: No custom_id found on line {line_num}")
                continue

            question_id = custom_id

            # Find corresponding entry in dataset
            if question_id not in dataset_qa:
                print(f"WARNING: Question ID '{question_id}' not found in dataset (line {line_num})")
                continue

            total_matched += 1
            qa_entry = dataset_qa[question_id]

            # Extract predicted answer
            try:
                choices = data.get('response', {}).get('body', {}).get('choices', [])
                if not choices:
                    print(f"WARNING: No choices found for question {question_id}")
                    continue

                predicted_answer = choices[0].get('message', {}).get('content', '').strip()
                if not predicted_answer:
                    print(f"WARNING: Empty predicted answer for question {question_id}")
                    continue

                # Calculate ROUGE-1 score
                expected_answers = qa_entry.get('answer', [])
                rouge1_score = calculate_rouge1_score(expected_answers, predicted_answer)

                # Check if below threshold
                if rouge1_score > threshold:
                    low_performance_ids.append(question_id)
                    print(f"Low performance: {question_id} - ROUGE-1: {rouge1_score:.3f}")

            except Exception as e:
                print(f"ERROR: Error processing question {question_id}: {e}")
                continue

    analysis_results = {
        'total_processed': total_processed,
        'total_matched': total_matched,
        'low_performance_count': len(low_performance_ids),
        'low_performance_ids': low_performance_ids,
        'threshold': threshold
    }

    print(f"Analysis complete:")
    print(f"  Total lines processed: {total_processed}")
    print(f"  Total questions matched: {total_matched}")
    print(f"  Questions with ROUGE-1 < {threshold}: {len(low_performance_ids)}")
    print(f"  Percentage below threshold: {(len(low_performance_ids) / total_matched * 100):.1f}%")

    return analysis_results


def create_low_performance_dataset(original_data: Dict, low_performance_ids: List[str], output_file: str):
    """
    Create a new HotpotQA dataset containing only low-performing questions.

    Args:
        original_data: Dictionary mapping question IDs to original HotpotQA question objects
        low_performance_ids: List of question IDs with low performance
        output_file: Path to save the new dataset
    """
    low_performance_questions = []

    for question_id in low_performance_ids:
        if question_id in original_data:
            low_performance_questions.append(original_data[question_id])
        else:
            print(f"WARNING: Question ID '{question_id}' not found in original data")

    print(f"Creating dataset with {len(low_performance_questions)} questions")

    # Save the new dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(low_performance_questions, f, indent=2, ensure_ascii=False)

    print(f"Low performance HotpotQA dataset saved to: {output_file}")

    # Print some statistics about the dataset
    question_types = {}
    difficulty_levels = {}
    
    for question in low_performance_questions:
        q_type = question.get('type', 'unknown')
        q_level = question.get('level', 'unknown')
        
        question_types[q_type] = question_types.get(q_type, 0) + 1
        difficulty_levels[q_level] = difficulty_levels.get(q_level, 0) + 1

    print("Dataset statistics:")
    print(f"  Question types: {question_types}")
    print(f"  Difficulty levels: {difficulty_levels}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create HotpotQA dataset with low-performing questions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '-f', '--file',
        required=True,
        help='Path to JSONL file with QA results'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.9,
        help='ROUGE-1 threshold (default: 0.9)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='low_performance_hotpot.json',
        help='Output file for low performance dataset (default: low_performance_hotpot.json)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.file):
        print(f"ERROR: Input file not found: {args.file}")
        return 1

    if args.threshold < 0 or args.threshold > 1:
        print(f"ERROR: Threshold must be between 0 and 1, got: {args.threshold}")
        return 1

    try:
        # Load datasets
        print("Loading original HotpotQA data...")
        original_data = load_hotpot_original_data()
        if not original_data:
            return 1

        print("Loading dataset for question mapping...")
        dataset_qa = load_dataset_qa()
        if not dataset_qa:
            return 1

        # Analyze JSONL file
        analysis_results = analyze_jsonl_file(args.file, dataset_qa, args.threshold)
        
        if analysis_results['low_performance_count'] == 0:
            print(f"No questions found with ROUGE-1 < {args.threshold}")
            return 0

        # Create new dataset
        create_low_performance_dataset(
            original_data, 
            analysis_results['low_performance_ids'], 
            args.output
        )

        print("Script completed successfully!")
        return 0

    except Exception as e:
        print(f"ERROR: Script failed with error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
