"""Orchestrator module to run the predictor and evaluator."""

from typing import Type
from agents.bm25.bm25 import BM25
from agents.colbertv2.colbertv2 import ColbertV2
from agents.colbertv2_reranker.colbertv2_reranker import ColbertV2Reranker
from agents.default.default import Default
from agents.dense.dense import Dense
from agents.hippo_rag.hippo_rag import HippoRAG
from agents.oracle.oracle import Oracle
from agents.react_agent.react_agent import ReactAgent
from agents.react_agent_custom.react_agent_custom import ReactAgentCustom
from agents.lexical_semantic_agent.lexical_semantic_agent import LexicalSemanticAgent
from agents.react_retriever.react_retriever import ReactRetriever
from data.hotpot.hotpot import Hotpot
from data.locomo.locomo import Locomo
from data.musique.musique import MuSiQue
from data.twowikimultihopqa.two_wiki import TwoWiki
from evaluator.evaluator import evaluator
from logger.logger import Logger
from models.agent import Agent
from models.dataset import Dataset
from predictor.predictor import predictor

# pylint: disable-next=too-few-public-methods
class Orchestrator:
    """
    Orchestrator class to run the predictor and evaluator.
    It initializes the agent and dataset based on the provided arguments.
    """

    def __init__(self, args):
        self._config = args

        datasets: dict[str, Type[Dataset]] = {
            'locomo': Locomo,
            'hotpot': Hotpot,
            '2wiki': TwoWiki,
            'musique': MuSiQue,
        }

        if args.dataset not in datasets:
            Logger().error(f"Dataset {args.dataset} not supported")
            raise ValueError(f"Dataset {args.dataset} not supported")

        agents: dict[str, Type[Agent]] = {
            'default': Default,
            'oracle': Oracle,
            'bm25': BM25,
            'dense': Dense,
            'colbertv2': ColbertV2,
            'colbertv2_rerank': ColbertV2Reranker,
            'hippo': HippoRAG,
            'react': ReactAgent,
            'react_custom': ReactAgentCustom,
            'lexical_semantic': LexicalSemanticAgent,
            'react_retriever': ReactRetriever,
        }

        if args.agent not in agents:
            Logger().error(f"Agent {args.agent} not supported")
            raise ValueError(f"Agent {args.agent} not supported")

        self.agent = agents[args.agent](args)
        self.dataset = datasets[args.dataset](args)

    def run(self):
        """
        Generates predictions for the given dataset using the specified agent.

        Raises:
            ValueError: if the execution mode is not supported
        """
        if self._config.execution == 'predict':
            Logger().info("Running predictor")
            predictor(self._config, self.dataset, self.agent)
        elif self._config.execution == 'eval':
            Logger().info("Running predictor")
            evaluator(self._config, self.dataset)
        else:
            Logger().error(
                f"Execution mode {self._config.execution} not supported")
            raise ValueError(
                f"Execution mode {self._config.execution} not supported")
