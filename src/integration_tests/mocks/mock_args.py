"""Mock arguments for integration tests."""
import argparse
from typing import Any, Dict


def create_agent_args(override_args: Dict[str, Any] = None) -> argparse.Namespace:
    """Create mock arguments for the CognitiveAgent integration test."""
    args = argparse.Namespace()

    if override_args is None:
        override_args = {}

    # Required arguments for prediction mode
    args.execution = 'predict'
    args.dataset = override_args.get('dataset', 'musique')
    args.model = 'gpt-4o-mini-2'
    args.agent = override_args.get('agent', 'cognitive')

    # Dataset configuration
    args.conversation = None
    args.questions = None
    args.category = None
    args.limit = 1  # Process only one question
    args.shuffle = False

    # Optional arguments
    args.k = None
    args.agent_args = None

    # Evaluation arguments (not used in predict mode)
    args.evaluation = None
    args.bert_eval = False
    args.judge_eval = False
    args.judge_eval_path = None
    args.retrieval = False
    args.retrieval_unranked = False
    args.metric = False
    args.eval_batch = False

    return args
