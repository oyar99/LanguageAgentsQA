"""Evaluating agent-based architectures for retrieval and answer generation tasks."""
import argparse
from dotenv import load_dotenv

from logger.logger import Logger, MainProcessLogger
from orchestrator.orchestrator import Orchestrator
from utils.argparse_utils import KeyValue


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='agent-eval-mem',
        description='Evaluate various agent-based architectures for retrieval and answer generation tasks'
    )

    parser.add_argument('-e', '--execution', choices=['eval', 'predict'], required=True,
                        help='mode of execution (required)')

    # Dataset processing arguments
    parser.add_argument('-d', '--dataset', choices=['locomo', 'hotpot', '2wiki',
                                                    'musique', 'hotpot2', 'musique2', '2wiki2'], required=True,
                        help='dataset to be processed (required)')
    parser.add_argument('-c', '--conversation', type=str,
                        help='conversation id to be extracted from the dataset - (optional)')
    parser.add_argument('-q', '--questions', type=int,
                        help='number of questions to be processed in each dataset sample (optional)')
    parser.add_argument('-ct', '--category', type=int,
                        help='category to be extracted from the dataset (optional)')
    parser.add_argument('-l', '--limit', type=int,
                        help='limit the number of samples to process. \
Ignored if conversation id is provided (optional)')
    parser.add_argument('-s', '--shuffle', action='store_true',
                        help='randomly shuffle dataset questions (optional)')

    # Predict mode arguments
    parser.add_argument('-m', '--model', choices=['gpt-4o-mini', 'gpt-4o-mini-2', 'o3-mini', 'gpt-4o-mini-batch',
                                                  'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct',
                                                  'google/gemma-3-12b-pt', 'phi-4-mini', 'gpt', 'gpt-2',
                                                  'gpt-4.1-mini'],
                        help='model deployment identifier (required in predict mode)')

    parser.add_argument('-a', '--agent', choices=['default', 'oracle', 'bm25', 'dense',
                                                  'colbertv2', 'colbertv2_rerank', 'hippo', 'react_custom',
                                                  'lexical_semantic', 'react_retriever', 'cognitive', 'react_hippo',
                                                  'dag_custom', 'dag_custom_v2'],
                        default='default', help='agent to be used (required in predict mode)')

    parser.add_argument('-ag', '--agent-args', type=str, nargs='*', action=KeyValue,
                        help='additional arguments for the agent in key=value format (optional)')

    # TODO: Remove this argument in favor of agent-args for agents that need k parameter
    parser.add_argument('-k', '--k', type=int,
                        help='number of documents to be retrieved for agents that support k argument (optional)')

    # Evaluation mode arguments
    parser.add_argument('-ev', '--evaluation', type=str,
                        help='evaluation file path (required in evaluation mode)')

    parser.add_argument('-bt', '--bert-eval', action='store_true',
                        help='run bert evaluation (optional)')

    parser.add_argument('-j', '--judge-eval', action='store_true',
                        help='run judge evaluation (optional). Other evaluations will be skipped in this mode')

    parser.add_argument('-jp', '--judge-eval-path', type=str,
                        help='path to the judge evaluation file (optional). \
If not provided, an evaluation file is generated')

    parser.add_argument('-r', '--retrieval', action='store_true',
                        help='run retrieval evaluation (optional)')

    parser.add_argument('-rr', '--retrieval-unranked', action='store_true',
                        help='run retrieval evaluation for unranked list of documents (optional)')

    parser.add_argument('-mt', '--metric', action='store_true',
                        help='run metric evaluation (optional)')

    parser.add_argument('-eb', '--eval-batch', action='store_true',
                        help='run batch evaluation (optional)')

    return parser.parse_args()


def main():
    """
    Entry point to evaluate agent-based architectures for retrieval and answer generation tasks.
    """
    Logger().info(
        f"Starting program with execution id: {Logger().get_run_id()}")

    args = parse_args()

    Orchestrator(args).run()

    Logger().info(
        f"Terminating program with execution id: {Logger().get_run_id()}")

    MainProcessLogger().shutdown()


if __name__ == "__main__":
    load_dotenv()
    main()
