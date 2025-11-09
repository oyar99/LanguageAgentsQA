"""2Wiki dataset module."""

import json
import os
from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.document import Document
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.hash_utils import get_content_hash
from utils.question_utils import filter_questions


class TwoWiki(Dataset):
    """2Wiki dataset class."""

    def __init__(self, args):
        super().__init__(args, name=args.dataset)

        # Override prompt dict entries
        # TODO: validate new few-shot examples that are more representative of the dataset do not impact
        # performance across other datasets. If not, reconcile dag execution prompt across datasets.
        self._prompt_dict['dag_execution_footer'] = DAG_EXECUTION_FOOTER

        Logger().info("Initialized an instance of the 2Wiki dataset")

    def read(self) -> list[DatasetSample]:
        """
        Reads the 2Wiki dataset.

        Returns:
            dataset (list[DatasetSample]): the dataset samples
        """
        Logger().info("Reading the 2WikiMultihopQA dataset")
        conversation_id = self._args.conversation

        file_name = "train.json" if self._args.dataset == "2wiki2" else "dev.json"

        # pylint: disable=duplicate-code
        file_path = os.path.join("data", "twowikimultihopqa", file_name)
        with open(file_path, encoding="utf-8") as two_wiki_dataset:
            dataset = [
                DatasetSample(
                    sample_id=sample['_id'],
                    sample=DatasetSampleInstance(
                        qa=filter_questions([QuestionAnswer(
                            docs=[Document(doc_id=get_content_hash(' '.join(doc[1])), content=' '.join(doc[1]))
                                  for doc in sample['context']
                                  if any(doc[0] == fact[0] for fact in sample['supporting_facts'])],
                            question_id=sample['_id'],
                            question=sample['question'],
                            answer=[str(sample['answer'])],
                            category=QuestionCategory.COMPARISON
                            if sample['type'] in ('comparison', 'bridge_comparison') else (
                                QuestionCategory.MULTI_HOP
                                if sample['type'] in ('inference', 'compositional') else QuestionCategory.NONE
                            )
                        )], self._args.questions, self._args.category)
                    )
                )
                for sample in json.load(two_wiki_dataset)
                if conversation_id is None or sample['_id'] == conversation_id
            ]
            dataset = super().process_dataset(dataset)
            Logger().info(
                f"2Wiki dataset read successfully. Total samples: {len(dataset)}")

            return dataset
        # pylint: enable=duplicate-code

    def read_corpus(self) -> list[Document]:
        """
        Reads the 2Wiki dataset and returns the corpus.

        Returns:
            corpus (list[Document]): the corpus
        """
        Logger().info("Reading the 2Wiki dataset corpus")
        file_name = "corpus_2.json" if self._args.dataset == "2wiki2" else "corpus.json"
        file_path = os.path.join("data", "twowikimultihopqa", file_name)
        with open(file_path, encoding="utf-8") as twowiki_corpus:
            corpus = json.load(twowiki_corpus)
            corpus = [
                Document(doc_id=get_content_hash(doc['text']), content=f'{doc["title"]}:{doc["text"]}')
                for doc in corpus
            ]
            super()._log_dataset_stats(corpus)

            return corpus

DAG_EXECUTION_FOOTER = '''### Example

[
    {
        "node_id": "node_1",
        "sub_question": "Who was John V, Prince Of Anhalt-Zerbst's father?",
        "dependencies": [],
        "result": "Ernest I",
        "context": "John V, Prince of Anhalt-Zerbst, was the second son of Ernest I, Prince of Anhalt-Dessau. Therefore, his father is Ernest I."
    },
    {
        "node_id": "node_2",
        "sub_question": "When did the person identified in node 1 die?",
        "dependencies": ["node_1"]
    }
]

Iteration 1:
```json
{
        "thought": "I have a DAG plan with 2 nodes where node_2 depends on node_1. Node_1 is already complete with result 'Ernest I'. Now I need to \
solve node_2 carefully using the result from node_1. I will invoke the answer action with a query that carefully replaces node references with actual values. \
That is, I need to ask 'When did Ernest I die?'. I must not include any node IDs in the query.",
    "actions": ["answer('When did Ernest I die?', 'node_2')"]
}
```

Iteration 2:
```json
{
        "thought": "I have a DAG plan with 2 nodes where node_2 depends on node_1. Node_1 is already complete with result 'Ernest I'. Now I need to \
solve node_2 carefully using the result from node_1. I will invoke the answer action with a query that carefully replaces node references with actual values. \
That is, I need to ask 'When did Ernest I die?'. I must not include any node IDs in the query.",
    "actions": ["answer('When did Ernest I die?', 'node_2')"],
    "observations": [["
    [{
        "node_id": "node_2",
        "sub_question": "When did the person identified in node 1 die?",
        "dependencies": ["node_1"],
        "result": "1516",
        "context": "Ernest I, Prince of Anhalt-Dessau died in 1516."
    }]
"]]
}
```

Iteration 3:
```json
{
    "thought": "All nodes are now complete.",
    "final_answer": "EXECUTION_COMPLETE"
}
```

### Example

[
    {
        "node_id": "node_1",
        "sub_question": "What is the nationality of director James Cameron?",
        "dependencies": [],
        "result": "Canadian-American",
        "context": "James Cameron is a Canadian-American film director, producer, and screenwriter."
    },
    {
        "node_id": "node_2",
        "sub_question": "What is the nationality of director Steven Spielberg?",
        "dependencies": [],
        "result": "American",
        "context": "Steven Spielberg is an American film director, producer, and screenwriter."
    },
    {
        "node_id": "node_3",
        "sub_question": "Do the directors from nodes 1 and 2 share the same nationality?",
        "dependencies": ["node_1", "node_2"]
    }
]

Iteration 1:
```json
{
    "thought": "I have a DAG plan with 3 nodes. Nodes 1 and 2 are already complete. Node_3 depends on both and asks if they share the same nationality. \
James Cameron is Canadian-American and Steven Spielberg is American. Since both have American nationality, they do share the same nationality. \
I can update the node directly as the answer can be deduced from the current DAG state.",
    "actions": ["update_node('Yes', 'node_3')"]
}
```

Iteration 2:
```json
{
    "thought": "I have a DAG plan with 3 nodes. Nodes 1 and 2 are already complete. Node_3 depends on both and asks if they share the same nationality. \
James Cameron is Canadian-American and Steven Spielberg is American. Since both have American nationality, they do share the same nationality. \
I can update the node directly as the answer can be deduced from the current DAG state.",
    "actions": ["update_node('Yes', 'node_3')"],
    "observations": [["
    [{
        "node_id": "node_3",
        "sub_question": "Do the directors from nodes 1 and 2 share the same nationality?",
        "dependencies": ["node_1", "node_2"],
        "result": "Yes"
    }]
"]]
}
```

Iteration 3:
```json
{
    "thought": "All nodes are now complete.",
    "final_answer": "EXECUTION_COMPLETE"
}
```

### Example

[
    {
        "node_id": "node_1",
        "sub_question": "Who was Lambert I, Count Of Louvain's father?",
        "dependencies": [],
        "result": "Reginar III",
        "context": "I found that Lambert I, Count of Louvain, was the son of Reginar III, Count of Hainaut and Adela d' Eguisheim. Therefore, the answer is Reginar III."
    },
    {
        "node_id": "node_2",
        "sub_question": "Where did the person identified in node 1 die?",
        "dependencies": ["node_1"]
    }
]

Iteration 1:
```json
{
    "thought": "I have a DAG plan with 2 nodes where node_2 depends on node_1. Node_1 is already complete with result 'Reginar III'. Now I need to \
solve node_2 carefully using the result from node_1. The sub_question asks where the person identified in node 1 died, but I need to use the actual result. \
The person identified in node 1 in the result field is Reginar III (Lambert I's father). I will invoke the answer action with a query that uses the actual name 'Where did Reginar III die?'. \
I must not include any node IDs in the query.",
    "actions": ["answer('Where did Reginar III die?', 'node_2')"]
}
```

Iteration 2:
```json
{
    "thought": "I have a DAG plan with 2 nodes where node_2 depends on node_1. Node_1 is already complete with result 'Reginar III'. Now I need to \
solve node_2 carefully using the result from node_1. The sub_question asks where the person identified in node 1 died, but I need to use the actual result. \
The person identified in node 1 in the result field is Reginar III (Lambert I's father). I will invoke the answer action with a query that uses the actual name 'Where did Reginar III die?'. \
I must not include any node IDs in the query.",
    "actions": ["answer('Where did Reginar III die?', 'node_2')"],
    "observations": [["
    [{
        "node_id": "node_2",
        "sub_question": "Where did the person identified in node 1 die?",
        "dependencies": ["node_1"],
        "result": "Florennes",
        "context": "I found that Reginar III, Count of Hainaut, died in Florennes, County of Namur, Holy Roman Empire in 973."
    }]
"]]
}
```

Iteration 3:
```json
{
    "thought": "All nodes are now complete.",
    "final_answer": "EXECUTION_COMPLETE"
}
```

**CRITICAL**: When solving dependent nodes, you MUST replace any references like "the person identified in node \
X" with the actual values from the "result" field of dependency nodes - never use vague node references in your \
queries, always use the specific names/values from the result fields. Your responses MUST NEVER contain any observations \
under absolutely no circumstances. Observations are returned to you after each action and are not part of your response. You must \
only respond with the JSON structure containing your thoughts and actions, or the final answer when ready.
'''