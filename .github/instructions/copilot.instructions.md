# GitHub Copilot Instructions

## General Coding Practices

### Type Annotations
- **Always provide complete type annotations** for all functions, methods, variables, and class attributes
- Use proper generic types from `typing` module when needed
- Include return type annotations even for simple functions
- Type annotate class attributes in `__init__` methods

### Code Quality
- **Disable pylint rules only when strictly necessary** and provide clear justification
- When disabling pylint, use specific rule codes (e.g., `# pylint: disable=too-many-arguments`) rather than broad disables
- Prefer refactoring code to meet pylint standards over disabling rules

## Repository Overview

This is a **Question Answering evaluation framework** that implements both **agentic** and **non-agentic** solutions for QA tasks. The system evaluates cognitive language agent architectures for multi-hop question answering across benchmark datasets.

## Architecture Components

### Entry Point
- **`src/index.py`**: Main entry point that parses command-line arguments and delegates to orchestrator

### Core Components

#### 1. Orchestrator (`src/orchestrator/`)
- **`orchestrator.py`**: Central coordinator that:
  - Initializes datasets and agents based on command-line arguments
  - Delegates execution to predictor or evaluator based on execution mode
  - Maps string names to concrete dataset and agent classes

#### 2. Agents (`src/agents/`)
All agents inherit from the abstract `Agent` base class in `src/models/agent.py` and must implement:
- `index(dataset)`: Index documents for retrieval
- `reason(question)`: Generate reasoning notebook for a single question
- `batch_reason(questions)`: Process multiple questions (not implemented in most agents)

**Example Agents (more may be available):**
- **`bm25/`**: BM25-based sparse retrieval agent
- **`colbertv2/`**: Dense retrieval using ColBERT v2
- **`colbertv2_reranker/`**: ColBERT with reranking capabilities  
- **`dense/`**: Generic dense embedding retrieval
- **`hippo_rag/`**: HippoRAG implementation
- **`oracle/`**: Perfect retrieval (uses ground truth supporting facts)
- **`react_agent/`**: ReAct framework using OpenAI native tools
- **`react_agent_custom/`**: Custom ReAct using instruction-tuned models with structured JSON output
- **`default/`**: Simple baseline agent

**Agent Properties:**
- `standalone`: Boolean indicating if agent produces final answers independently
- `support_batch`: Boolean indicating if agent supports batch processing

#### 3. Datasets (`src/data/`)
All datasets inherit from abstract `Dataset` class in `src/models/dataset.py` and implement:
- `read_corpus()`: Return list of documents for indexing
- `read_qa()`: Return list of QuestionAnswer objects
- Dataset filtering by conversation ID, question count, question type

**Example Datasets (more may be available):**
- **`locomo/`**: Long conversation QA dataset (10 conversations)
- **`hotpot/`**: Multi-hop QA with Wikipedia articles
- **`twowikimultihopqa/`**: 2-4 hop reasoning questions
- **`musique/`**: Multi-hop questions via composition

#### 4. Models (`src/models/`)
Core data structures:
- **`agent.py`**: Abstract Agent base class and NoteBook class for storing reasoning traces
- **`dataset.py`**: Abstract Dataset base class
- **`document.py`**: Document representation with content and metadata
- **`question_answer.py`**: QA pair with supporting facts and question type
- **`retrieved_result.py`**: Retrieved document with relevance score

#### 5. Predictor (`src/predictor/`)
- **`predictor.py`**: Orchestrates prediction generation:
  - Indexes dataset using selected agent
  - Processes questions (single or batch mode)
  - Handles multiprocessing for parallel execution
  - Saves results to output files

#### 6. Evaluator (`src/evaluator/`)
Evaluation metrics implementation:
- **`exact_match_evaluator.py`**: Exact string match
- **`f1_evaluator.py`**: Token-level F1 score
- **`rogue_evaluator.py`**: ROUGE metrics
- **`bert_evaluator.py`**: BERT-based semantic similarity
- **`judge_evaluator.py`**: LLM-based evaluation

#### 7. Azure OpenAI Integration (`src/azure_open_ai/`)
- **`openai_client.py`**: Singleton client for Azure OpenAI
- **`chat_completions.py`**: Synchronous chat completion handling
- **`batch.py`**: Batch processing for multiple requests
- **`batch_evaluation.py`**: Batch evaluation workflows

#### 8. Utilities (`src/utils/`)
Helper modules:
- **`model_utils.py`**: Model capability checking
- **`token_utils.py`**: Token counting and management
- **`question_utils.py`**: Question processing utilities
- **`hash_utils.py`**: Content hashing for caching
- **`singleton.py`**: Singleton pattern implementation

### Execution Modes
- **Prediction Mode** (`predict`): Generates answers for given datasets
- **Evaluation Mode** (`eval`): Runs evaluation metrics against ground truth

### Development Guidelines

#### Adding New Agents
1. Create new directory under `src/agents/`
2. Inherit from `Agent` base class
3. Implement required abstract methods
4. Add to orchestrator's agent mapping
5. Follow existing naming conventions

#### Adding New Datasets  
1. Create new directory under `src/data/`
2. Inherit from `Dataset` base class
3. Implement corpus and QA reading methods
4. Add to orchestrator's dataset mapping
5. Include download scripts and README

### Configuration
- Models support both closed-source (Azure OpenAI) and open-source (VLLM) backends
- Environment variables for API keys and endpoints
- Command-line arguments for execution parameters

