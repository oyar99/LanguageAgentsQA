"""LexicalSemanticAgent for intelligent search using both lexical and semantic approaches.

This agent automatically decides between BM25 lexical search and ColBERT semantic search
based on the nature of the query, following ReAct framework with manual tool invocation.
"""
# pylint: disable=duplicate-code
import json
import os
from typing import Dict, List, Any, Optional, Set

from rank_bm25 import BM25Okapi as BM25Ranker
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult
from utils.tokenizer import PreprocessingMethod, tokenize
from utils.model_utils import supports_temperature_param


class LexicalSemanticAgent(Agent):
    """
    LexicalSemanticAgent that intelligently chooses between lexical and semantic search
    based on query characteristics using ReAct framework.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._bm25_index = None
        self._colbert_searcher = None
        self._args = args
        super().__init__(args)

        self.standalone = True

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for both lexical (BM25) and semantic (ColBERT) retrieval.
        """
        Logger().info("Indexing documents using LexicalSemanticAgent")
        corpus = dataset.read_corpus()

        # Index for BM25 lexical search
        Logger().info("Creating BM25 index for lexical search")
        self._index_bm25(corpus)

        # Index for ColBERT semantic search
        Logger().info("Creating ColBERT index for semantic search")
        self._index_colbert(dataset, corpus)

        self._corpus = corpus
        Logger().info("Successfully indexed documents for both lexical and semantic search")

    def _index_bm25(self, corpus: List[Dict[str, Any]]) -> None:
        """Create BM25 index for lexical search."""
        self._bm25_index = BM25Ranker(
            corpus,
            tokenizer=self._tokenize_doc,
            b=0.75,
            k1=0.5
        )

    def _tokenize_doc(self, doc: Dict[str, Any]) -> List[str]:
        """Tokenize a document for BM25 indexing."""
        return tokenize(
            doc['content'],
            ngrams=2,
            remove_stopwords=True,
            preprocessing_method=PreprocessingMethod.STEMMING
        )

    def _index_colbert(self, dataset: Dataset, corpus: List[Dict[str, Any]]) -> None:
        """Create ColBERT index for semantic search."""
        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        os.makedirs(colbert_dir, exist_ok=True)

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            config = ColBERTConfig(
                nbits=2,
            )
            indexer = Indexer('colbert-ir/colbertv2.0', config=config)
            indexer.index(
                name=dataset.name or 'index',
                collection=[doc['content'] for doc in corpus],
                overwrite='reuse'  # type: ignore
            )

        self._index = dataset.name or 'index'

        # Initialize searcher
        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            self._colbert_searcher = Searcher(index=self._index, collection=[
                doc['content'] for doc in corpus])

    def _search_lexical(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform BM25 lexical search.

        Args:
            query (str): The search query.
            k (int): Number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved documents with scores.
        """
        if not self._bm25_index or not self._corpus:
            raise ValueError("BM25 index not created. Please index the dataset first.")

        # Tokenize the query
        tokenized_query = tokenize(
            query,
            ngrams=2,
            remove_stopwords=True,
            preprocessing_method=PreprocessingMethod.STEMMING
        )

        # Get scores for the query
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top k documents with their scores
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        documents = []
        for idx, score in top_k:
            documents.append({
                'doc_id': self._corpus[idx]['doc_id'],
                'content': self._corpus[idx]['content'],
                'score': score,
                'original_id': idx
            })

        return documents

    def _search_semantic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform ColBERT semantic search.

        Args:
            query (str): The search query.
            k (int): Number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved documents with scores.
        """
        if not self._colbert_searcher:
            Logger().warn("ColBERT searcher not available, falling back to lexical search")
            return self._search_lexical(query, k)

        doc_ids, _ranking, scores = self._colbert_searcher.search(query, k=k)

        documents = []
        for doc_id, _, score in zip(doc_ids, _ranking, scores):
            documents.append({
                'doc_id': self._corpus[doc_id]['doc_id'],
                'content': self._corpus[doc_id]['content'],
                'score': score,
                'original_id': doc_id
            })

        return documents

    def _create_react_prompt(self, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Create a ReAct prompt for the agent.

        Args:
            conversation_history (List[Dict[str, str]], optional): Previous conversation turns.

        Returns:
            str: Formatted ReAct prompt.
        """
        base_prompt = LEXICAL_SEMANTIC_AGENT_PROMPT

        if conversation_history:
            history_text = "\n".join(json.dumps(turn, indent=2)
                                     for turn in conversation_history)
            base_prompt += f"\n\nCONVERSATION HISTORY:\n{history_text}"

        return base_prompt

    def _parse_structured_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse the structured JSON response from the model.

        Args:
            response_content (str): Raw response from the model.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON response or None if parsing fails.
        """
        try:
            # Try to extract JSON from the response
            if '```json' in response_content:
                json_start = response_content.find('```json') + 7
                json_end = response_content.find('```', json_start)
                json_str = response_content[json_start:json_end].strip()
            elif '{' in response_content and '}' in response_content:
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                json_str = response_content[json_start:json_end]
            else:
                # Fallback: try to parse the entire response
                json_str = response_content.strip()

            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            Logger().warn(f"Failed to parse structured response: {e}")
            Logger().debug(f"Raw response: {response_content}")
            return None

    # pylint: disable-next=too-many-branches,too-many-locals,too-many-statements
    def reason(self, question: str) -> NoteBook:  # type: ignore
        """
        Reason over the indexed dataset to answer the question using ReAct framework
        with intelligent lexical/semantic search selection.
        """
        Logger().debug(f"Starting reasoning for question: {question}")

        if not self._bm25_index or not self._colbert_searcher:
            raise ValueError(
                "Indexes not initialized. Please index the dataset first.")

        conversation_history = []
        sources: Set[int] = set()
        max_iterations = 8
        iteration = 0
        final_answer = None

        while iteration < max_iterations and final_answer is None:
            iteration += 1
            Logger().debug(f"ReAct iteration {iteration}")

            # Create prompt with conversation history
            prompt = self._create_react_prompt(conversation_history)

            # Call the model
            messages = [
                {"role": "system", "content": LEXICAL_SEMANTIC_AGENT_PROMPT},
                {"role": "user", "content": question},
                {"role": "system", "content": prompt}
            ]

            Logger().debug(f"Sending messages to model: {messages}")

            open_ai_request = {
                "custom_id": f"lexical_semantic_iteration_{iteration}",
                "model": self._args.model,
                "messages": messages,
                "temperature": default_job_args['temperature']
                if supports_temperature_param(self._args.model) else None,
                "frequency_penalty": default_job_args['frequency_penalty'],
                "presence_penalty": default_job_args['presence_penalty'],
                "max_completion_tokens": 1000,
            }

            result = chat_completions([open_ai_request])[0][0]
            response_content = result.choices[0].message.content.strip()

            Logger().debug(f"Model response: {response_content}")

            # Parse structured response
            parsed_response = self._parse_structured_response(response_content)

            if not parsed_response:
                # Fallback: treat as final answer
                final_answer = response_content
                break

            # Handle the structured response
            thought = parsed_response.get('thought', '')
            actions = parsed_response.get('actions', [])
            final_answer = parsed_response.get('final_answer', None)

            # Add to conversation history
            turn = {'thought': thought, 'actions': actions, 'observations': []}

            for action in actions:
                # Parse action string to extract function name and arguments
                action_name = None
                action_input = None

                if isinstance(action, str) and '(' in action and ')' in action:
                    # Extract function name and arguments from string like "search_lexical('query')"
                    paren_start = action.find('(')
                    paren_end = action.rfind(')')

                    action_name = action[:paren_start].strip()
                    args_str = action[paren_start + 1:paren_end].strip()

                    # Remove quotes from arguments if present
                    if args_str.startswith(("'", '"')) and args_str.endswith(("'", '"')):
                        action_input = args_str[1:-1]
                    else:
                        action_input = args_str

                # Handle search actions
                if action_name in ['search_lexical', 'search_semantic']:
                    if action_name == 'search_lexical':
                        documents = self._search_lexical(
                            action_input, k=self._args.k or 5)
                        Logger().debug(
                            f"Lexical search for '{action_input}': {len(documents)} results")
                    else:  # search_semantic
                        documents = self._search_semantic(
                            action_input, k=self._args.k or 5)
                        Logger().debug(
                            f"Semantic search for '{action_input}': {len(documents)} results")

                    # Track sources
                    sources.update(doc['original_id'] for doc in documents)

                    # Create observation
                    turn['observations'].append([doc['content'] for doc in documents])

            conversation_history.append(turn)

        if final_answer is None:
            final_answer = "N/A"

        Logger().info(f"Final answer: {final_answer}")

        # Create notebook with results
        notebook = NoteBook()
        notebook.update_sources([
            RetrievedResult(
                doc_id=self._corpus[doc_id]['doc_id'],
                content=self._corpus[doc_id]['content']
            )
            for doc_id in sources
        ])
        notebook.update_notes(final_answer)

        return notebook

    def batch_reason(self, _: List[QuestionAnswer]) -> List[NoteBook]:  # type: ignore
        """
        Batch reasoning is not implemented for the LexicalSemanticAgent.

        Raises:
            NotImplementedError: Batch reasoning is not implemented.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the LexicalSemanticAgent.")

    def multiprocessing_reason(self, questions: List[str]) -> List[NoteBook]:
        """
        Reason over the indexed dataset to answer multiple questions.
        """
        notebooks = []

        for question in questions:
            notebook = self.reason(question)
            notebooks.append(notebook)

        return notebooks


# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

LEXICAL_SEMANTIC_AGENT_PROMPT = '''You are an intelligent search agent that can choose between two search methods
to find relevant documents: lexical search and semantic search.

## AVAILABLE TOOLS:

1. **search_lexical(query)**: Performs exact keyword matching using BM25 algorithm. Best for:
   - Names of people, places, organizations, products
   - Specific technical terms or proper nouns
   - Exact phrases or quotes
   - When you need precise term matching

2. **search_semantic(query)**: Performs meaning-based search using ColBERT. Best for:
   - Conceptual queries about topics or themes
   - Questions requiring understanding of context
   - When the answer might use different words than the query
   - Complex reasoning or analytical questions

## SEARCH STRATEGY:

Choose your search method based on the query characteristics:
- If the query contains **named entities** (people, places, organizations, products, etc.), use `search_lexical`
- For **conceptual or analytical** queries, use `search_semantic`
- You can use both methods if needed to gather comprehensive information

## RESPONSE FORMAT:

Respond with exactly one JSON object per response. Use either the intermediate format or the final format, never both.

**Intermediate responses** (while searching):
```json
{
  "thought": "Your reasoning about what information you need and which search method to use",
  "actions": ["search_lexical('specific name or term')", "search_semantic('conceptual query')"]
}
```

**Final response** (when ready to answer):
```json
{
  "thought": "Your final reasoning based on the retrieved information",
  "final_answer": "Concise, exact answer to the question"
}
```

## INSTRUCTIONS:

1. **Think step by step** about what information you need
2. **Choose the appropriate search method** based on query type
3. **Use retrieved information** to provide accurate, concise answers
4. **Be precise** - if the answer is a single word, date, or name, provide just that
5. **If information is insufficient**, set final_answer to "N/A"

## EXAMPLE:

Question: "What nationality is Scott Derrickson?"

Iteration 1:
```json
{
  "thought": "I need to find information about Scott Derrickson's nationality. Since this is asking about a specific person's nationality, I should use lexical search to find exact mentions of his name.",
  "actions": ["search_lexical('Scott Derrickson nationality')"]
}
```

Iteration 2 (after receiving search results):
```json
{
  "thought": "The search results show that Scott Derrickson is an American film director. Based on this information, I can provide the answer.",
  "final_answer": "American"
}
```

Your goal is to find the most relevant information using the appropriate search method \
and provide accurate, concise answers.'''
