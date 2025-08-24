# Evaluating Cognitive Language Agent Architectures for Question Answering

_Cognitive Language Agents_ is a framework for designing **intelligent** language agents that integrate **LLMs** for reasoning and communication, using language as their primary means of interacting with their environment. These agents consist of three key components: **memory, action, and decision-making**.

In this work, we present a **systematic evaluation** of three agent-based architectures designed for **Question Answering (QA)** and **Multi-Hop Question Answering (MHQA)**. Our goal is to assess how well these architectures perform on **general-purpose tasks** and how effectively their **planning, collaboration, and decision-making capabilities** can be leveraged for both **retrieval** and **answering**.

## Datasets

We evaluate the **three systems** against well-known benchmark datasets for **QA and MHQA**:

- **_LoCoMo_**: A dataset consisting of **10 very long-term conversations** between two users, annotated for the QA task. The dataset has been **forked into this repository** under the `src/datasets/locomo` directory. See [LoCoMo](https://github.com/snap-research/locomo) for details on dataset generation and statistics.

- **_HotpotQA_**: A QA dataset featuring **natural, multi-hop questions**. This dataset has been **forked into this repository** under the `src/datasets/hotpot` directory, with instructions on how to initialize it correctly.

- **_2WikiMultihopQA_**: A QA dataset to evaluate Multi-Hop questions that contains comprehensive information of reasoning paths required to arrive at the correct answer. This dataset can be found under `src/datasets/twowikimultihopqa` directory. See [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop?tab=readme-ov-file) for more details on the dataset.

- **_MuSiQue_**: A Multi-Hop QA dataset with 2-4 hop questions constructeed via single-hop question composition. This dataset can be found under `src/datasets/musique` directory. See [MuSiQue](https://github.com/stonybrooknlp/musique) for more details on the dataset.

## Questions

Each dataset includes a subset of **five different types of questions**, with a particular focus on **Multi-Hop questions**:

1. **Multi-Hop (1)**: The model must **perform multiple reasoning steps** across different parts of the conversation to derive the correct answer.
2. **Temporal (2)**: The model must answer a question that requires **understanding dates and times** within the conversation.
3. **Open-Domain (3)**: General broad questions about the conversation that require **deep comprehension**.
4. **Single-Hop (4)**: The model must **extract a single piece of information** from the conversation to answer the question.
5. **Adversarial (5)**: The model must determine whether the answer is **(a) not mentioned** or **(b) explicitly stated** in the conversation.

## Requirements

We recommend using **Python 3.13** and creating a virtual environment.

Verify the installed Python version:

```sh
python --version
```

Create virtual environment.

```bash
python -m venv env
```

Activate the virtual environment.

```bash
.\env\Scripts\activate
```

```bash
source env/bin/activate
```

Install required packages.

```bash
pip install -r requirements.txt
```

Install `faiss_gpu-1.7.3`.

```bash
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## Closed-Source Models

The script supports any **closed-source models** that allow **batch deployments** via the Azure Open AI API and **open-source models** that are available via VLLM.

Sample Models

| Model Name                  | Context Length   | Max Outputh Length |
|-----------------------------|------------------|--------------------|
| o3-mini                     | 200,000 tokens   | 100,000 tokens     |
| GPT-4o-mini                 | 128,000 tokens   | 16,384 tokens      |
| Qwen2.5-14B-Instruct        | 32,000 tokens    | 8,192 tokens       |
| Qwen2.5-1.5B-Instruct       | 32,768 tokens    | 8,192 tokens       |
| Gemma 3-27B                 | 128,000 tokens   | 8,192 tokens       |

## VLLM

To start a `VLLM` HTTP server, the following command can be used.

```sh
vllm serve Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 2 --dtype float16 --gpu-memory-utilization 0.95 --max-model-len 32000 --max-num-seqs 128
```

**Explanation:**

```sh
Qwen/Qwen2.5-14B-Instruct # Specifies the LLM model to use
--tensore-parallel-size # Specifies the number of GPUs to use using tensor parallelism
--dtype float16 # Loads the model using 16-bit floating point precision (fp16)
--gpu-memory-utilization # Specifies how much memory to use for each GPU
--max-model-len # Sets the maximum number of tokens per input sequence to 32,000
--max-nums-seqs # Maximum number of concurrent sequences (i.e., requests) that can be processed in parallel.
```

## How to run

The script supports two execution modes:

- `predict`: Generates answers for a given dataset.
- `eval`: Runs evaluation metrics (`Exact Match (EM)`, `F1 Score`, `R1 Score`, `L1 Score`) against ground-truth answers.

### Running Predictions

Below are examples of using the script in `predict` mode.

#### Example 1: Single-Hop Questions (LoCoMo Dataset)

To generate predictions for **up to 20 single-hop questions** from a single conversation in the _LoCoMo_ dataset using `gpt-4o-mini`, run:

```sh
python .\index.py -e predict -m gpt-4o-mini -c conv-26 -q 20 -ct 4 -d locomo
```

**Explanation:**

```sh
-e predict    # Runs the script in prediction mode.
-m gpt-4o-mini    # Specifies GPT-4o-mini as the model.
-c conv-26    # Identifies the conversation ID to process. Change "conv-26" to target a different conversation or omit.
-q 20    # Limits the number of questions to at most 20.
-ct 4    # Filters only single-hop questions.
-d locomo    # Specifies the dataset (LoCoMo) to use.
```

#### Example 2: Multi-Hop Questions (HotpotQA Dataset)

To generate predictions for all multi-hop questions from up to 10 conversations in the _hotpotQA_ dataset using gpt-4o-mini, you can run the following command:

```sh
python .\index.py -e predict -m gpt-4o-mini -l 10 -ct 1 -d hotpot
```

**Explanation:**

```sh
-l 10    # Limits the number of conversations/samples to at most 10.
-ct 1    # Filters only multi-hop questions.
-d hotpot    # Specifies the dataset (hotpotQA) to use.
```

### Running Evaluation

To evalaute the generated predictions against ground truth using **Exact Match (EM)** and **F1 Score**, run:

```sh
python .\index.py -e "eval" -ev "predictions.jsonl" -d hotpot
```

**Explanation:**

```sh
-e eval    # Runs the script in evaluation mode.
-ev predictions.jsonl    # Path to the batch output containing the generated answers.
```

The metrics will be logged.

### Long-Running tasks

Some processes may take long to complete if processing full datasets. We recommend using [nohup](https://www.man7.org/linux/man-pages/man1/nohup.1.html) for Linux.

```sh
nohup python index.py -e predict -m gpt-4o-mini -d musique -a react_custom &
```

To check the progress, either review the log file or run

```sh
tail -f nohup.out
```

### Getting Help

For more details on available command-line arguments, run:

```sh
python .\index.py --help
```
