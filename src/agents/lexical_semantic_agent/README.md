# LexicalSemanticAgent

An intelligent search agent that automatically chooses between lexical (BM25) and semantic (ColBERT) search based on query characteristics.

## Overview

The LexicalSemanticAgent implements a ReAct (Reasoning + Acting) framework that intelligently selects the appropriate search method:

- **Lexical Search (BM25)**: Used for queries containing named entities (people, places, organizations, products)
- **Semantic Search (ColBERT)**: Used for conceptual queries requiring meaning-based understanding

## Usage

```bash
# Using the agent with the orchestrator
python src/index.py --agent lexical_semantic --dataset <dataset_name> --execution predict
```

## Architecture

### Core Components

1. **Named Entity Detection**
   - Primary: spaCy NER for accurate entity recognition
   - Fallback: Simple heuristics based on capitalization patterns
   
2. **Search Implementations**
   - **BM25 Lexical**: Self-contained implementation using `rank_bm25`
   - **ColBERT Semantic**: Integration with ColBERT v2 (graceful fallback to lexical if unavailable)

3. **ReAct Framework**
   - Custom prompting with clear tool descriptions
   - Manual tool invocation via JSON parsing
   - Structured conversation history tracking

### Decision Logic

```python
if contains_named_entities(query):
    use_search_lexical(query)
else:
    use_search_semantic(query)
```

## Examples

### Named Entity Queries (→ Lexical Search)
- "What is Scott Derrickson's nationality?"
- "Where is Microsoft headquartered?" 
- "Tell me about Albert Einstein"

### Conceptual Queries (→ Semantic Search)
- "How does photosynthesis work?"
- "What are the benefits of exercise?"
- "Explain machine learning algorithms"

## Features

- ✅ **Standalone Operation**: Produces final answers independently
- ✅ **Model Agnostic**: Works with any chat_completions compatible model
- ✅ **Robust Fallbacks**: Handles missing dependencies gracefully
- ✅ **Self-Contained**: No shared code dependencies with other agents
- ✅ **Production Ready**: Comprehensive error handling and logging

## Dependencies

### Required
- `rank_bm25`: For lexical search functionality
- `azure_open_ai`: For LLM interaction via chat_completions

### Optional
- `spacy` + `en_core_web_sm`: For advanced named entity recognition
- `colbert-ai`: For semantic search (falls back to lexical if unavailable)

## Configuration

The agent inherits configuration from the standard agent framework:
- `args.model`: LLM model to use for reasoning
- `args.k`: Number of documents to retrieve per search (default: 5)