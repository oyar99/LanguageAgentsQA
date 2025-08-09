"""ReactAgentCustom for reasoning using custom instruction fine-tuned model with structured output schema.
"""
# pylint: disable=duplicate-code
import json
import os
from typing import Dict, List, Any, Optional
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult
from utils.model_utils import supports_temperature_param


class ReactAgentCustom(Agent):
    """
    ReactAgentCustom for reasoning over indexed documents using a custom instruction fine-tuned model
    with structured output schema following the ReAct prompting framework.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._searcher = None
        self._args = args
        super().__init__(args)

        self.standalone = True

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for retrieval using ColBERT.
        """
        Logger().info("Indexing documents using ColbertV2")
        corpus = dataset.read_corpus()

        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        os.makedirs(colbert_dir, exist_ok=True)

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            config = ColBERTConfig(
                nbits=2,
            )
            self._index = Indexer('colbert-ir/colbertv2.0', config=config)
            self._index.index(
                name=dataset.name or 'index',
                collection=[doc['content'] for doc in corpus],
                overwrite='reuse'  # type: ignore
            )

        self._index = dataset.name or 'index'
        self._corpus = corpus
        Logger().info("Successfully indexed documents")

        # Initialize searcher
        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            self._searcher = Searcher(index=self._index, collection=[
                doc['content'] for doc in self._corpus])

    def _search_documents(self, query: str) -> tuple[List[Dict[str, Any]], List[int]]:
        """
        Search for documents using the ColBERT retriever.

        Args:
            query (str): The search query.

        Returns:
            tuple[List[Dict[str, Any]], List[int]]: Tuple containing list of retrieved documents 
with content and list of doc_ids.
        """
        if not self._searcher:
            raise ValueError(
                "Searcher not initialized. Please index the dataset first.")

        doc_ids, _ranking, scores = self._searcher.search(
            query, k=self._args.k or 5)

        documents = []
        for doc_id, _, score in zip(doc_ids, _ranking, scores):
            documents.append({
                'doc_id': self._corpus[doc_id]['doc_id'],
                'content': self._corpus[doc_id]['content'],
                'score': score
            })

        Logger().debug(
            f"Search results for query '{query}': Found {len(documents)} documents")
        return documents, doc_ids

    def _create_react_prompt(self, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Create a ReAct prompt for the custom instruction fine-tuned model.

        Args:
            conversation_history (List[Dict[str, str]], optional): Previous conversation turns.

        Returns:
            str: Formatted ReAct prompt.
        """
        base_prompt = REACT_AGENT_CUSTOM_PROMPT

        if conversation_history:
            history_text = "\n".join([
                f"Thought: {turn['thought']}" if 'thought' in turn else "" +
                f"Action: {turn['action']}" if 'action' in turn else "" +
                f"Observation: {turn['observation']}" if 'observation' in turn else ""
                for turn in conversation_history
            ])
            base_prompt += f"\n\nConversation History:\n{history_text}"

        return base_prompt

    def _parse_structured_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse the structured JSON response from the custom model.

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
            Logger().warning(f"Failed to parse structured response: {e}")
            Logger().debug(f"Raw response: {response_content}")
            return None

    def reason(self, question: str) -> NoteBook:  # type: ignore
        """
        Reason over the indexed dataset to answer the question using ReAct framework
        with a custom instruction fine-tuned model.
        """
        Logger().debug(f"Starting reasoning for question: {question}")

        if not self._searcher:
            raise ValueError(
                "Searcher not initialized. Please index the dataset first.")

        conversation_history = []
        sources = set()
        max_iterations = 5
        iteration = 0
        final_answer = None

        while iteration < max_iterations and final_answer is None:
            iteration += 1
            Logger().debug(f"ReAct iteration {iteration}")

            # Create prompt with conversation history
            prompt = self._create_react_prompt(conversation_history)

            # Call the custom model
            messages = [
                {"role": "system", "content": REACT_AGENT_CUSTOM_PROMPT},
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ]

            open_ai_request = {
                "custom_id": f"react_iteration_{iteration}",
                "model": self._args.model,
                "messages": messages,
                "temperature": default_job_args['temperature']
                if supports_temperature_param(self._args.model) else None,
                "frequency_penalty": default_job_args['frequency_penalty'],
                "presence_penalty": default_job_args['presence_penalty'],
                "max_completion_tokens": 1000,
            }

            try:
                result = chat_completions([open_ai_request])[0][0]
                response_content = result.choices[0].message.content.strip()

                Logger().debug(f"Model response: {response_content}")

                # Parse structured response
                parsed_response = self._parse_structured_response(
                    response_content)

                if not parsed_response:
                    Logger().warning("Could not parse response, using fallback handling")
                    # Fallback: treat as final answer
                    final_answer = response_content
                    break

                # Handle the structured response
                thought = parsed_response.get('thought', '')
                action = parsed_response.get('action', '')
                action_input = parsed_response.get('action_input', '')
                final_answer = parsed_response.get('final_answer', '')

                # Add to conversation history
                turn = {'thought': thought}

                if action and action.lower() == 'search':
                    turn['action'] = f"search({action_input})"

                    # Perform search
                    documents, doc_ids = self._search_documents(action_input)

                    # Track sources
                    sources.update(doc_ids)

                    # Create observation
                    observation = "Retrieved documents:\n" + "\n".join([
                        f"Document {i+1}: {doc['content']}"
                        for i, doc in enumerate(documents)
                    ])

                    turn['observation'] = observation
                    conversation_history.append(turn)

                elif final_answer:
                    # Model provided final answer
                    Logger().debug(f"Final answer provided: {final_answer}")
                    break
                else:
                    # No clear action, treat as final answer
                    final_answer = thought or response_content
                    break

            except (json.JSONDecodeError, ValueError, RuntimeError) as e:
                Logger().error(f"Error in ReAct iteration {iteration}: {e}")
                final_answer = "N/A"
                break

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

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:  # type: ignore
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for the ReactAgentCustom.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the ReactAgentCustom.")

    def multiprocessing_reason(self, questions: list[str]) -> list[NoteBook]:
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

# Main instruction prompt for the custom model
REACT_AGENT_CUSTOM_PROMPT = '''You will be presented with a question, and you will need to search for relevant \
documents that support the answer to the question. You will then use these documents to provide an EXACT answer, \
using only words found in the documents when possible. UNDER no circumstances should you include any additional \
commentary, explanations, reasoning, or notes in your response. Your response MUST be concise and to the point. \
If the answer can be a single word (e.g., Yes, No, a date, or an object), please provide just that word.

You should decompose the question into multiple sub-questions if necessary, and search for relevant documents for each sub-question. \
Each query needs to be optimized to maximize the probability of retrieving the most relevant documents using a semantic retriever. \
As such, you can rephrase the question to make it more specific or to focus on a particular aspect of the question. \
You should then use the retrieved documents to answer the original question.

For example, consider the following question:

"Were Scott Derrickson and Ed Wood of the same nationality?"

You can decompose this question into two sub-questions, and then search for relevant documents for each sub-question.

1. {"thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.", "action": "search", "action_input": "Scott Derrickson nationality"}

Scott Derrickson is an American film director, producer, and screenwriter. He is known for his work in the horror genre, including \
films like "The Exorcism of Emily Rose" and "Doctor Strange".

2. {"thought": "Now I need to find Ed Wood's nationality to compare with Scott Derrickson's.", "action": "search", "action_input": "Ed Wood nationality"}

Ed Wood was an American filmmaker, actor, and writer, often regarded as one of the worst directors in film history. He is best known \
for his cult classic "Plan 9 from Outer Space".

{"thought": "Both Scott Derrickson and Ed Wood are American, so they are of the same nationality.", "action": null, "final_answer": "Yes"}

Consider another question:

"In which county is Kimbrough Memorial Stadium located?"

You can first search for the location of Kimbrough Memorial Stadium.

1. {"thought": "I need to find where Kimbrough Memorial Stadium is located.", "action": "search", "action_input": "Kimbrough Memorial Stadium location"}

Kimbrough Memorial Stadium is a stadium in Canyon, Texas. It is owned by Canyon Independent School District, and is primarily \
used for American football.

Then, you can search for the county of Canyon, Texas.

2. {"thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.", "action": "search", "action_input": "Canyon Texas county"}

Canyon is a city in, and the county seat of, Randall County, Texas, United States. The population was 13,303 at the 2010 census.

{"thought": "Kimbrough Memorial Stadium is in Canyon, Texas, and Canyon is in Randall County.", "action": null, "final_answer": "Randall County"}

Your response must be in JSON format with the following structure and your final answer MUST be formatted as a single line of text, \
containing ONLY the answer to the question following the aforementioned rules. If the answer cannot be inferred with the information \
found in the documents, you MUST then respond with "N/A".

{
    "thought": "Your reasoning process",
    "action": "search" (if you need to search) or null (if ready to answer),
    "action_input": "your search query" (only if action is search),
    "final_answer": "your final answer" (only when you have enough information)
}'''
