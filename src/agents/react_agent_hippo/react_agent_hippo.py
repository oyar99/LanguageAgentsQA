"""ReactAgentCustom for reasoning using custom instruction fine-tuned model with structured output schema.
"""
# pylint: disable=duplicate-code
import os
from typing import Dict, List
from logger.logger import Logger
from models.action import Action
from models.agent import SingleProcessIntelligentAgent
from models.dataset import Dataset

# pylint: disable-next=too-few-public-methods
class ReactAgentHippo(SingleProcessIntelligentAgent):
    """
    ReactAgentCustom for reasoning over indexed documents using a custom instruction fine-tuned model
    with structured output schema following the ReAct prompting framework.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._reverse_doc_map = None
        self._args = args
        actions = {
            "search": Action(
                "Search for relevant documents in the knowlege graph to answer the given query. \
The system works best when queries include specific entities and relationships \
extracted from the question",
                self._search_documents
            )
        }
        prompt_examples = PROMPT_EXAMPLES_TOOLS
        super().__init__(actions, prompt_examples, args)

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for retrieval using HippoRAG.
        """
        Logger().info("Indexing documents using HippoRAG")
        corpus = dataset.read_corpus()

        hipporag_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'hipporag' + os.sep + (dataset.name or ""))

        os.makedirs(hipporag_dir, exist_ok=True)

        embedding_model = 'facebook/contriever'

        # pylint: disable-next=import-outside-toplevel
        from hipporag import HippoRAG as HippoRAGModel

        force_remote_llm = os.getenv("REMOTE_LLM", None)

        hipporag = HippoRAGModel(
            save_dir=hipporag_dir,
            llm_model_name=self._args.model,
            embedding_model_name=embedding_model,
            llm_base_url='http://localhost:8000/v1',
            azure_endpoint=None if force_remote_llm != "1" else (
                os.getenv("AZURE_OPENAI_ENDPOINT", None)),
        )

        hipporag.index(docs=[doc['content'] for doc in corpus])

        self._index = hipporag
        self._corpus = corpus
        self._reverse_doc_map = {
            doc['content']: doc_idx for doc_idx, doc in enumerate(corpus)}

        Logger().info("Successfully indexed documents")

    def _search_documents(
            self,
            query: str,
    ) -> tuple[List[str], List[str], Dict[str, int]]:
        """
        Search for documents using the ColBERT retriever.

        Args:
            query (str): The search query.

        Returns:
            tuple[List[str], List[str], Dict[str, int]]: Tuple containing list of observations (retrieved documents)
, list of sources if any, and metrics if any.
        """
        results = self._index.retrieve(
            [query], num_to_retrieve=self._args.k or 5)

        documents = []
        for doc, score in zip(results[0].docs, results[0].doc_scores):
            documents.append({
                'doc_id': self._reverse_doc_map[doc],
                'content': doc,
                'score': score,
            })

        Logger().debug(
            f"Search results for query '{query}': Found {len(documents)} documents")

        return ([doc['content'] for doc in documents],
                [doc['doc_id'] for doc in documents],
                {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0})

PROMPT_EXAMPLES_TOOLS = '''### Example 1

Question: "Were Scott Derrickson and Ed Wood of the same nationality?"

Iteration 1:
```json
{
    "thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",
    "actions": ["search('Scott Derrickson's nationality')", "search('Ed Wood's nationality')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",
    "actions": ["search('Scott Derrickson's nationality')", "search('Ed Wood's nationality')"],
    "observations": [["Scott Derrickson is an American film director, producer, and screenwriter. He is known for his work in the horror genre, including \
films like 'The Exorcism of Emily Rose' and 'Doctor Strange'."], ["Ed Wood was an American filmmaker, actor, and writer, often regarded as one of the worst directors in film history. He is best known \
for his cult classic 'Plan 9 from Outer Space'."]]
}
```

Iteration 3:
```json
{
    "thought": "Both Scott Derrickson and Ed Wood are American based on the retrieved information, so they are of the same nationality.",
    "final_answer": "Yes"
}
```

### Example 2

Question: "In which county is Kimbrough Memorial Stadium located?"

Iteration 1:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search('Kimbrough Memorial Stadium location')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search('Kimbrough Memorial Stadium location')"],
    "observations": [["Kimbrough Memorial Stadium is a stadium in Canyon, Texas. It is owned by Canyon Independent School District, and is primarily \
used for American football."]]
}
```

Iteration 3:
```json
{
    "thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",
    "actions": ["search('Canyon Texas county')"]
}
```

Iteration 4:
```json
{
    "thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",
    "actions": ["search('Canyon Texas county')"],
    "observations": [["Canyon is a city in, and the county seat of, Randall County, Texas, United States. The population was 13,303 at the 2010 census."]]
}
```

Iteration 5:
```json
{
    "thought": "Kimbrough Memorial Stadium is in Canyon, Texas, and Canyon is in Randall County.",
    "final_answer": "Randall County"
}
```
'''
