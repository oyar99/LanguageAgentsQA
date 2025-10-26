"""Locomo dataset module."""

import json
import os
import random
import re
from typing import Optional

from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.document import Document
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.hash_utils import get_content_hash
from utils.question_utils import filter_questions


def session_id(doc_id: str) -> str:
    """
    Extracts the session id from the doc_id.

    Args:
        doc_id (str): the document id

    Returns:
        session_id (str): the session id
    """
    return f"session_{doc_id.split(':')[0][1:]}"


def dia_idx(doc_id: str) -> int:
    """
    Extracts the index of the dialogue from the doc_id.

    Args:
        doc_id (str): the document id

    Returns:
        dia-idx (int): the index of the dialogue
    """
    return int(doc_id.split(':')[1]) - 1


def format_content(date: str, message: int, speaker: str, text: str, alt_text: Optional[str] = None) -> str:
    """
    Formats the content of a message.

    Args:
        date (str): the date of the message
        message (int): the message number
        speaker (str): the speaker of the message
        text (str): the text of the message
        alt_text (str): the alt text of attached images if any
    """
    return f"At around {date}, during message {message}, {speaker} said: {text}" if alt_text is None else \
        f"At around {date}, during message {message}, {speaker} said: {text} - Attached image: {alt_text}"


class Locomo(Dataset):
    """Locomo dataset class."""

    def __init__(self, args):
        super().__init__(args, name="locomo")
        # Override the default prompts
        self._prompt_dict = {
            'qa_rel': QA_PROMPT_RELEVANT,
            'qa_all': QA_PROMPT_ALL,
            'dag_footer': DAG_DECOMPOSITION_FOOTER,
            'react_footer': REACT_FOOTER,
            'react_footer_2': REACT_FOOTER,
            'dag_execution_footer': DAG_EXECUTION_FOOTER,
            'dag_synthesis_footer': DAG_SYNTHESIS_FOOTER,
            'solve_command_example': SOLVE_COMMAND_EXAMPLE,
            'final_answer_command_example': FINAL_ANSWER_COMMAND_EXAMPLE,
            'alternative_answer_command_example': ALTERNATIVE_ANSWER_COMMAND_EXAMPLE,
        }
        Logger().info("Initialized an instance of the Locomo dataset")

    # @override
    def read(self) -> list[DatasetSample]:
        """
        Reads the Locomo dataset.

        Returns:
            dataset (list[DatasetSample]): the Locomo dataset
        """
        Logger().info("Reading Locomo dataset")
        conversation_id = self._args.conversation
        file_path = os.path.join("data", "locomo", "locomo10.json")

        with open(file_path, "r", encoding="utf-8") as locomo_dataset:
            data = json.load(locomo_dataset)

            if self._args.shuffle:
                random.shuffle(data)
                Logger().info("Questions shuffled randomly")

            dataset = []
            for cs in data:
                if conversation_id is None or (cs['sample_id'] == 'conv-26'): # or cs['sample_id'] == 'conv-30'):
                    # Shuffle questions within each conversation if shuffle is enabled
                    qa_list = cs['qa']
                    if self._args.shuffle:
                        qa_list = qa_list.copy()
                        random.shuffle(qa_list)

                    dataset.append(DatasetSample(
                        sample_id=cs['sample_id'],
                        sample=DatasetSampleInstance(
                            qa=filter_questions([QuestionAnswer(
                                docs=[Document(
                                    doc_id=f'{ev}:{cs["sample_id"]}',
                                    folder_id=cs['sample_id'],
                                    content=format_content(
                                        date=cs['conversation'][f'{session_id(ev)}_date_time'],
                                        message=dia_idx(ev) + 1,
                                        speaker=cs['conversation'][session_id(
                                            ev)][dia_idx(ev)]['speaker'],
                                        text=cs['conversation'][session_id(
                                            ev)][dia_idx(ev)]['text'],
                                        alt_text=cs['conversation'][session_id(ev)][dia_idx(ev)].get('blip_caption'))
                                )
                                    for ev in qa['evidence']
                                    if session_id(ev) in cs['conversation'] and dia_idx(ev) <
                                    len(cs['conversation'][session_id(ev)])],
                                question_id=f'{cs["sample_id"]}-{get_content_hash(qa["question"])}',
                                question=qa['question'],
                                answer=[str(qa.get('answer')) or str(qa.get(
                                    'adversarial_answer'))],
                                category=QuestionCategory(qa['category'])
                            ) for _, qa in enumerate(qa_list)], self._args.questions, self._args.category)
                        )
                    ))
            dataset = super().process_dataset(dataset)

            Logger().info(
                f"Locomo dataset read successfully. Total samples: {len(dataset)}")

            return dataset

    # @override
    def read_corpus(self) -> list[Document]:
        """
        Reads the LoCoMo dataset corpus.

        Returns:
            corpus (list[str]): list of docs (messages) from the corpus
        """
        Logger().info("Reading the LoCoMo dataset corpus")
        file_path = os.path.join("data", "locomo", "locomo10.json")
        with open(file_path, encoding="utf-8") as locomo_corpus:
            corpus = json.load(locomo_corpus)

            pattern = re.compile(r"^session_\d+$")

            corpus = [
                Document(
                    doc_id=f'{str(message["dia_id"])}:{conversation_sample["sample_id"]}',
                    folder_id=conversation_sample['sample_id'],
                    content=format_content(
                        date=conversation_sample['conversation'][f'{key}_date_time'],
                        message=dia_idx(message['dia_id']) + 1,
                        speaker=message['speaker'],
                        text=message['text'],
                        alt_text=message.get('blip_caption'),
                    ),
                )
                for conversation_sample in corpus
                for key, session in conversation_sample['conversation'].items() if pattern.match(key)
                for message in session
            ]
            super()._log_dataset_stats(corpus)

            return corpus

    # @override
    def get_question(self, question_id: str) -> Optional[QuestionAnswer]:
        """
        Gets a question from the dataset.

        Args:
            question_id (str): the id of the question to be retrieved

        Raises:
            ValueError: if the dataset is not read or the question id is not found in the dataset

        Returns:
            question (Optional[QuestionAnswer]): the question if found, None otherwise
        """
        if not self._dataset and not self._dataset_map:
            Logger().error("Dataset not read. Please read the dataset before getting questions.")
            raise ValueError(
                "Dataset not read. Please read the dataset before getting questions.")

        match = re.match(r'^(conv-.\d+)-(.*)$',
                         question_id)

        if not match:
            Logger().error(
                f"Invalid question id format: {question_id}. Expected format: conv-<sample_id>-<message_id>")
            raise ValueError(
                f"Invalid question id format: {question_id}. Expected format: conv-<sample_id>-<message_id>"
            )

        sample_id = str(match.group(1))
        message_id = str(match.group(2))

        if sample_id not in self._dataset_map:  # type: ignore
            Logger().error(
                f"Sample id {sample_id} not found in the dataset.")
            raise ValueError(
                f"Sample id {sample_id} not found in the dataset.")

        return next((
            qa for qa in self._dataset_map[sample_id]['qa']  # type: ignore
            if get_content_hash(qa['question']) == message_id), None
        )


QA_PROMPT_RELEVANT = '''You are a helpful Question Answering assistant. You will be presented with snippets from a \
conversation between two users, followed by a question. Your task is to provide an EXACT and short answer, using words \
found in the conversations when possible. If the answer can be a single word (e.g., Yes, No, or an entity), please \
answer with just that word. For dates, always answer with ABSOLUTE dates such as "5 July 2023" or "week before 5 June" instead \
of relative answers such as "Yesterday" or "last week" since your answers should not depend on the current date.

For example, given the following conversation:

"At around 1:50 pm on 17 August, 2023, during message 15, Caroline said: I'm always here for you, Mel! We had a blast last year \
at the Pride fest. Those supportive friends definitely make everything worth it!"

And given the following question:

Q: "When did Caroline and Melanie go to a pride festival together?"

Your answer should be: 

"2022"

The conversation takes place over multiple days and the date of each conversation is added at the beginning of each message.

Below are the relevant messages in the conversation.

{context}
'''

QA_PROMPT_ALL = '''You are a helpful Question Answering assistant. You will be presented with multiple conversations \
between two users, followed by a list of questions. Your task is to provide an EXACT and short answer to each question\
, using words found in the conversations when possible. If the answer can be a single word (e.g., Yes, No, or an entity)\
, please answer with just that word. For dates, always answer with "ABSOLUTE" dates such as "5 July 2023" or "week before \
5 June" instead of relative answers such as "Yesterday", "last week" or "5 years ago" since your answers SHOULD NOT depend \
on today's date.

For example, given the following conversation:

"At around 1:50 pm on 17 August, 2023, during message 15, Caroline said: I'm always here for you, Mel! We had a blast last year \
at the Pride fest. Those supportive friends definitely make everything worth it!"

And given the following question:

Q (<question_id>): "When did Caroline and Melanie go to a pride festival together?"

Your answer should be:

"2022"

Each conversation takes place over multiple days and the date of each conversation is added at the beginning of each message.

Format your answer as a JSON object where each question is answered exactly once. Your response should also honor the given question \
order and question ids.

Below are the conversations in the dataset. (The conversations may be truncated due to length constraints, but all relevant \
details are guaranteed to be included.)

{context}
'''

REACT_FOOTER = '''## EXAMPLES

### Example 1

Question: "When did Jolene`s mother pass away?"

Iteration 1:
```json
{
    "thought": "I need to find information about Jolene's mother.",
    "actions": ["search('Jolenes's mother')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find information about Jolene's mother.",
    "actions": ["search('Jolenes's mother')"],
    "observations": [[""At around 4:06 pm on 23 January, 2023, during message 6, Jolene said: Sorry about your loss. My mother also passed away \
last year. This is my room in her house, I also have many memories there. Is there anything special about it you remember? - Attached image: a \
photo of a room with a bench and a window"]]
}
```

Iteration 3:
```json
{
    "thought": "Jolene mentioned on 23 January, 2023 that her mother passed away last year. She didn't mention a specific date. Therefore, we can
only infer she died in the year 2022.",
    "final_answer": "2022"
}
```

### Example 2

Question: "How many weeks passed between Maria adopting Coco and Shadow?"

Iteration 1:
```json
{
    "thought": "I need to find out when Maria adopted Coco and when she adopted Shadow.",
    "actions": ["search('Maria adopted Coco')", "search('Maria adopted Shadow')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find out when Maria adopted Coco and when she adopted Shadow.",
    "actions": ["search('Maria adopted Coco')", "search('Maria adopted Shadow')"],
    "observations": [["At around 12:10 am on 11 August, 2023, during message 1, Maria said: Hey John! Long time no talk! \
Guess what - I got a puppy two weeks ago! Her name's Coco and she's adorable. - Attached image: a photo of a white dog laying in the grass with a baseball"], ["At \
around 3:14 pm on 13 August, 2023, during message 2, Maria said: Wow, John! You're doing great things. Kudos for helping kids learn. On another note, I \
just adopted this cute pup from a shelter. She brings so much joy! I feel blessed to be able to give her a home."]]
}
```

Iteration 3:
```json
{
    "thought": "Maria mentioned she got Coco two weeks ago from 11 August, 2023. I did not find information about Shadow, but I found that she \
adopted another dog on 13 August, 2023. I need to search for more messages around that date to find out if that dog is named Shadow.",
    "actions": ["search('Maria and John messages 3 and 4 on 13 August, 2023 at around 3:14 pm about the adopted dog')"]
}
```

Iteration 4:
```json
{
    "thought": "Maria mentioned she got Coco two weeks ago from 11 August, 2023. I did not find information about Shadow, but I found that she \
adopted another dog on 13 August, 2023. I need to search for more messages around that date to find out if that dog is named Shadow.",
    "actions": ["search('Maria message 3 and 4 on 13 August, 2023 at around 3:14 pm about the adopted dog')"]
    "observations": [["At around 3:14 pm on 13 August, 2023, during message 3, John said: Wow, Maria, how cute! What's her name? Do she and your other dog \
get along?", "At around 3:14 pm on 13 August, 2023, during message 4, Maria said: Her name is Shadow! She's full of energy and always puts a smile on my \
face. She's been a great addition to my life. They get along great."]]
}
```

Iteration 5:
```json
{
    "thought": "Maria mentioned she adopted a dog on 13 August, 2023. She got Coco two weeks ago from 11 August, 2023. \
This means there were approximately two weeks between Coco and Shadow adoption.",
    "final_answer": "two weeks"
}
```
'''

DAG_DECOMPOSITION_FOOTER = '''The questions are based on a dataset of conversations between two users. \
Each conversation consists of multiple messages exchanged over several days. Each message includes a timestamp indicating when it was sent.
If a question structure is simple and does not require multiple steps to answer, you must create a DAG with a single node as per the examples below.

## EXAMPLES

Question: "When did Caroline go to the adoption meeting?"

Response:
{
    "reasoning": "To find out when Caroline went to the adoption meeting, I need to determine the date of the meeting.",
    "dag_plan": [
        {
            "node_id": "node_1",
            "sub_question": "When did Caroline go to the adoption meeting?",
            "dependencies": []
        }
    ]
}

Question: "How many weeks passed between Maria adopting Coco and Shadow?"

Response:
{
    "reasoning": "I need to first find when Maria adopted Coco and then when she adopted Shadow. After that, I can calculate the number of weeks between \
those two dates.",
    "dag_plan": [
        {
            "node_id": "node_1",
            "sub_question": "What is the date Maria adopted Coco?",
            "dependencies": []
        },
        {
            "node_id": "node_2",
            "sub_question": "What is the date Maria adopted Shadow?",
            "dependencies": []
        },
        {
            "node_id": "node_3",
            "sub_question": "How many weeks passed between the adoption of Coco and Shadow?",
            "dependencies": ["node_1", "node_2"]
        }
    ]
}
'''

DAG_EXECUTION_FOOTER = '''### Example

[
    {
        "node_id": "node_1",
        "sub_question": "What is the date Maria adopted Coco?",
        "dependencies": []
    },
    {
        "node_id": "node_2", 
        "sub_question": "What is the date Maria adopted Shadow?",
        "dependencies": []
    },
    {
        "node_id": "node_3",
        "sub_question": "How many weeks passed between the adoption of Coco and Shadow?",
        "dependencies": ["node_1", "node_2"]
    }
]

Iteration 1:
```json
{
    "thought": "I have a DAG plan with 3 nodes where node_3 depends on node_1 and node_2. I should start by solving node_1 first.",
    "actions": ["answer('Maria adoption date Coco', 'node_1')"]
}
```

Iteration 2:
```json
{
    "thought": "I have a DAG plan with 3 nodes where node_3 depends on node_1 and node_2. I should start by solving node_1 first.",
    "actions": ["answer('Maria adoption date Coco', 'node_1')"]
    "observations": [["
    [{
        "node_id": "node_1",
        "sub_question": "What is the date Maria adopted Coco?",
        "dependencies": [],
        "result": "2 weeks before 11 August, 2023",
        "context": "Maria mentioned she got a puppy named Coco two weeks before 11 August, 2023 when she sent a message on that date.",
        "sources": ["At around 12:10 am on 11 August, 2023, during message 1, Maria said: Hey John! Long time no talk! Guess what - I got a puppy two \
weeks ago! Her name's Coco and she's adorable"]
    }]
"]]
}
```

Iteration 3:
```json
{
    "thought": "Node_1 is complete and shows that Maria adopted Coco 2 weeks before 11 August, 2023. Now I need to solve node_2.",
    "actions": ["answer('Maria adoption date Shadow', 'node_2')"]
}
```

Iteration 4:
```json
{
    "thought": "Node_1 is complete and shows that Maria adopted Coco 2 weeks before 11 August, 2023. Now I need to solve node_2.",
    "actions": ["answer('Maria adoption date Shadow', 'node_2')"]
    "observations": [["
    [{
        "node_id": "node_2",
        "sub_question": "What is the date Maria adopted Shadow?",
        "dependencies": [],
        "result": "Maria adopted Shadow on August 13, 2023",
        "context": "Maria sent a message on 13 August, 2023 mentioning she just adopted a dog. In a later message on the same day, she revealed the dog's name is Shadow.",
        "sources": ["At around 3:14 pm on 13 August, 2023, during message 2, Maria said: Wow, John! You're doing great things. Kudos for helping kids learn. On \
another note, I just adopted this cute pup from a shelter. She brings so much joy! I feel blessed to be able to give her a home.", "At around 3:14 pm on 13 \
August, 2023, during message 4, Maria said: Her name is Shadow! She's full of energy and always puts a smile on my face. She's been a great addition to my life. They \
get along great."]
    }]
"]]
}
```

Iteration 5:
```json
{
    "thought": "node_2 is complete and shows that Maria adopted Shadow on August 13, 2023. Now that I have the results for node_1 and node_2, I can answer node_3 \
directly. Since Maria adopted Coco 2 weeks before 11 August, 2023, and she adopted Shadow two days later on 13 August, 2023, it means that 2 weeks passed between the two adoptions.",
    "actions": ["update_node('2 weeks', 'node_3')"]
}
```

Iteration 6:
```json
{
    "thought": "node_2 is complete and shows that Maria adopted Shadow on August 13, 2023. Now that I have the results for node_1 and node_2, I can answer node_3 \
directly. Since Maria adopted Coco 2 weeks before 11 August, 2023, and she adopted Shadow two days later on 13 August, 2023, it means that 2 weeks passed between the two adoptions.",
    "actions": ["update_node('2 weeks', 'node_3')"]
    "observations": [["All dependent nodes are complete or no more nodes can be solved. DAG execution may be finished."]]
}
```

Iteration 7:
```json
{
    "thought": "All nodes are now complete.",
    "final_answer": "EXECUTION_COMPLETE"
}
```

**IMPORTANT**: Your responses MUST NEVER contain any observations under absolutely no circumstances. Observations are returned to you after each action \
and are not part of your response. You must only respond with the JSON structure containing your thoughts and actions, or the final answer when ready.
'''

DAG_SYNTHESIS_FOOTER = '''### Example

## DAG State

[
  {
    "node_id": "node_1",
    "sub_question": "What is the date Maria adopted Coco?",
    "dependencies": [],
    "result": "2 weeks before 11 August, 2023",
    "context": "I found that Maria mentioned she got a puppy named Coco two weeks before 11 August, 2023 when she sent a message on that date."
  },
  {
    "node_id": "node_2",
    "sub_question": "What is the date Maria adopted Shadow?",
    "dependencies": [],
    "result": "August 13, 2023",
    "context": "I found that Maria sent a message on 13 August, 2023 mentioning she just adopted a dog. In a later message on the same day, she \
revealed the dog's name is Shadow."
  },
  {
    "node_id": "node_3",
    "sub_question": "How many weeks passed between the adoption of Coco and Shadow?",
    "dependencies": [
      "node_1",
      "node_2"
    ],
    "result": "2 weeks",
  }
]

Question: "How many weeks passed between Maria adopting Coco and Shadow?"

Output:
{
    "thought": "Using the results from node_1 and node_2, Maria adopted Coco 2 weeks before 11 August, 2023, and she adopted Shadow on August 13, 2023. \
This means that 2 weeks passed between the two adoptions.",
    "answer": "2 weeks"
}
'''

SOLVE_COMMAND_EXAMPLE = '''### Example

DAG Plan:
[
    {
        "node_id": "node_1",
        "sub_question": "When did Caroline go to the adoption meeting?",
        "dependencies": []
    }
]

SOLVE(node_id="node_1")

Output:

{
    "thought": "This is a simple single-node question that doesn't depend on any other nodes. I need to search for information about Caroline's adoption meeting.",
    "answer": "",
    "tool": "SEARCH",
    "query": "When did Caroline go to the adoption meeting?"
}

### Example

DAG Plan:
[
    {
        "node_id": "node_1",
        "sub_question": "What is the date Maria adopted Coco?",
        "dependencies": [],
        "result": "2 weeks before 11 August, 2023"
    },
    {
        "node_id": "node_2",
        "sub_question": "What is the date Maria adopted Shadow?",
        "dependencies": [],
        "result": "August 13, 2023"
    },
    {
        "node_id": "node_3",
        "sub_question": "How many weeks passed between the adoption of Coco and Shadow?",
        "dependencies": ["node_1", "node_2"]
    }
]

SOLVE(node_id="node_3")

Output:
{
    "thought": "Node_3 depends on node_1 and node_2, which found that Maria adopted Coco 2 weeks before 11 August, 2023 (so around July 28, 2023) \
and she adopted Shadow on August 13, 2023. Based on the conversation context, the time difference between these two adoptions is 2 weeks.",
    "answer": "2 weeks"
}
'''

FINAL_ANSWER_COMMAND_EXAMPLE = '''### Example

DAG Plan:
[
  {
    "node_id": "node_1",
    "sub_question": "What is the date Maria adopted Coco?",
    "dependencies": [],
    "result": "2 weeks before 11 August, 2023",
    "context": "I found that Maria mentioned she got a puppy named Coco two weeks before 11 August, 2023 when she sent a message on that date."
  },
  {
    "node_id": "node_2",
    "sub_question": "What is the date Maria adopted Shadow?",
    "dependencies": [],
    "result": "August 13, 2023",
    "context": "I found that Maria sent a message on 13 August, 2023 mentioning she just adopted a dog. In a later message on the same day, she revealed the dog's name is Shadow."
  },
  {
    "node_id": "node_3",
    "sub_question": "How many weeks passed between the adoption of Coco and Shadow?",
    "dependencies": [
      "node_1",
      "node_2"
    ],
    "result": "2 weeks"
  }
]

SOLVE(question="How many weeks passed between Maria adopting Coco and Shadow?")

Output:
{
    "thought": "Using the results from node_1 and node_2, Maria adopted Coco 2 weeks before 11 August, 2023, and she adopted Shadow on August 13, 2023. \
This means that 2 weeks passed between the two adoptions.",
    "answer": "2 weeks"
}

### Example

DAG Plan:
[
  {
    "node_id": "node_1",
    "sub_question": "When did Caroline go to the adoption meeting?",
    "dependencies": [],
    "result": "March 15, 2023",
    "context": "Caroline mentioned in a conversation that she went to an adoption meeting on March 15, 2023."
  }
]

SOLVE(question="When did Caroline go to the adoption meeting?")

Output:
{
    "thought": "Based on node_1, Caroline went to the adoption meeting on March 15, 2023.",
    "answer": "March 15, 2023"
}
'''

ALTERNATIVE_ANSWER_COMMAND_EXAMPLE = '''### Example

[
    {
        "node_id": "node_1",
        "sub_question": "Where did Caroline move from 4 years ago?",
        "dependencies": [],
        "result": "Sweden",
        "sources": [
            "At around 11:30 am on 9 June, 2023, during message 22, Caroline said: I moved here from Sweden 4 years ago, and it was quite an adjustment.",
            "At around 2:45 pm on 15 July, 2023, during message 25, Caroline said: Back in Stockholm, I had a different life. Moving to this country changed everything for me."
        ]
    }
]

ALTERNATIVE_ANSWER(node_id="node_1", exclude=["Sweden"])

Output:

{
    "thought": "Analyzing the sources more carefully, Caroline mentions she moved from Sweden but also specifically references \
'Stockholm' in one of the messages. Stockholm is the capital city of Sweden, so I can provide 'Stockholm' as an \
alternative, more specific answer to where she moved from.",
    "answer": "Stockholm"
}
'''
