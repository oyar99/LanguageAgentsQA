#!/usr/bin/env python3
"""
Interactive Search Tool

This script provides an interactive search interface using either BM25 (lexical) 
or ColBERT (semantic) retriever. It initializes the specified dataset, creates 
the appropriate index, and allows users to perform searches interactively with 
real-time results.

Usage:
    python interactive_search.py -d <dataset_name> -m <search_method> [options]
    
Examples:
    python interactive_search.py -d musique -m bm25
    python interactive_search.py -d locomo -m colbert
    python interactive_search.py -d hotpot -m bm25 -k 10
    python interactive_search.py -d twowiki -m colbert -k 3
"""

import argparse
import os
import sys
import threading
from typing import List, Dict, Any

# ColBERT imports
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

# BM25 imports
from rank_bm25 import BM25Okapi as BM25Ranker

# Project imports
from logger.logger import Logger
from data.musique.musique import MuSiQue
from data.locomo.locomo import Locomo
from data.hotpot.hotpot import Hotpot
from data.twowikimultihopqa.two_wiki import TwoWiki
from models.dataset import Dataset
from utils.tokenizer import PreprocessingMethod, tokenize


class InteractiveSearchEngine:
    """
    Interactive search engine using either BM25 or ColBERT for document retrieval.
    """
    
    def __init__(self, dataset_name: str, search_method: str = 'colbert'):
        self.dataset_name = dataset_name
        self.search_method = search_method.lower()
        self.dataset = None
        self.corpus = None
        self.searcher = None
        self.bm25_content_index = None
        self.bm25_title_index = None
        self.index_name = None
        self.lock = threading.Lock()
        
        # Initialize logger
        Logger()
        
        # Validate search method
        if self.search_method not in ['bm25', 'colbert']:
            raise ValueError(f"Unsupported search method: {search_method}. "
                           f"Supported methods: ['bm25', 'colbert']")
        
    def _create_mock_args(self):
        """Create mock arguments for dataset initialization."""
        class MockArgs:
            def __init__(self):
                self.limit = None
                self.conversation = None
                self.shuffle = False
                self.questions = None
                self.category = None
                self.model = None
        
        return MockArgs()
    
    def _initialize_dataset(self):
        """Initialize the specified dataset."""
        Logger().info(f"Initializing {self.dataset_name} dataset...")
        
        # Map dataset names to classes
        dataset_classes = {
            'musique': MuSiQue,
            'locomo': Locomo,
            'hotpot': Hotpot,
            'twowiki': TwoWiki
        }
        
        if self.dataset_name.lower() not in dataset_classes:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. "
                           f"Supported datasets: {list(dataset_classes.keys())}")
        
        dataset_class = dataset_classes[self.dataset_name.lower()]
        args = self._create_mock_args()
        
        self.dataset = dataset_class(args)
        Logger().info(f"Successfully initialized {self.dataset.name} dataset")
    
    def _read_corpus(self):
        """Read and prepare the corpus for indexing."""
        Logger().info("Reading corpus...")
        self.corpus = self.dataset.read_corpus()
        Logger().info(f"Loaded {len(self.corpus)} documents from corpus")
        
        # Log corpus statistics
        total_chars = sum(len(doc['content']) for doc in self.corpus)
        avg_chars = total_chars / len(self.corpus) if self.corpus else 0
        Logger().info(f"Average document length: {avg_chars:.2f} characters")
    
    def _tokenize_doc(self, doc: Dict[str, Any]) -> List[str]:
        """Tokenize a document for BM25 indexing."""
        return tokenize(
            doc['content'],
            ngrams=2,
            remove_stopwords=True,
            preprocessing_method=PreprocessingMethod.STEMMING
        )
    
    def _tokenize_title(self, doc: Dict[str, Any]) -> List[str]:
        """Tokenize document title for BM25 indexing."""
        # Extract title from content (format is usually "Title:Content")
        title = ""
        if 'title' in doc and doc['title']:
            title = doc['title']
        elif ':' in doc['content']:
            title = doc['content'].split(':', 1)[0].strip()
        
        # Handle empty or None titles
        if not title:
            return []
        
        return tokenize(
            title,
            ngrams=1,  # Use unigrams for titles for better exact matching
            remove_stopwords=False,  # Keep all words in titles
            preprocessing_method=PreprocessingMethod.NONE
        )
    
    def _create_bm25_index(self):
        """Create BM25 index for lexical search with separate content and title indexing."""
        Logger().info("Creating BM25 index for lexical search...")
        
        # Prepare tokenized corpus for BM25 (content)
        tokenized_content = [self._tokenize_doc(doc) for doc in self.corpus]
        
        # Prepare tokenized titles for BM25 (titles)
        tokenized_titles = [self._tokenize_title(doc) for doc in self.corpus]
        
        # Create separate BM25 indices for content and titles
        self.bm25_content_index = BM25Ranker(
            tokenized_content,
            b=0.75,
            k1=1.5
        )
        
        self.bm25_title_index = BM25Ranker(
            tokenized_titles,
            b=0.75,
            k1=1.5
        )
        
        Logger().info("Successfully created BM25 indices for content and titles")
    
    def _create_index(self):
        """Create index based on the selected search method."""
        if self.search_method == 'bm25':
            self._create_bm25_index()
        elif self.search_method == 'colbert':
            self._create_colbert_index()
    
    def _create_colbert_index(self):
        """Create ColBERT index for the corpus."""
        Logger().info("Creating ColBERT index...")
        
        # Set up ColBERT directory
        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')
        os.makedirs(colbert_dir, exist_ok=True)
        
        # Create index
        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            config = ColBERTConfig(nbits=2)
            indexer = Indexer('colbert-ir/colbertv2.0', config=config)
            
            self.index_name = self.dataset.name or 'index'
            indexer.index(
                name=self.index_name,
                collection=[doc['content'] for doc in self.corpus],
                overwrite='reuse'
            )
        
        Logger().info("Successfully created ColBERT index")
    
    def _initialize_searcher(self):
        """Initialize the searcher based on the search method."""
        if self.search_method == 'colbert':
            self._initialize_colbert_searcher()
        # BM25 doesn't need separate searcher initialization
    
    def _initialize_colbert_searcher(self):
        """Initialize the ColBERT searcher."""
        Logger().info("Initializing ColBERT searcher...")
        
        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')
        
        with self.lock:
            with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
                self.searcher = Searcher(
                    index=self.index_name,
                    collection=[doc['content'] for doc in self.corpus],
                    verbose=1
                )
        
        Logger().info("Successfully initialized ColBERT searcher")
    
    def setup(self):
        """Set up the search engine by initializing dataset, corpus, index, and searcher."""
        try:
            self._initialize_dataset()
            self._read_corpus()
            self._create_index()
            self._initialize_searcher()
            Logger().info("Search engine setup completed successfully!")
            return True
        except Exception as e:
            Logger().error(f"Failed to set up search engine: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using the query.
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores and metadata
        """
        try:
            if self.search_method == 'bm25':
                return self._search_bm25(query, top_k)
            elif self.search_method == 'colbert':
                return self._search_colbert(query, top_k)
        except Exception as e:
            Logger().error(f"Search failed: {str(e)}")
            return []
    
    def _search_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 lexical search combining content and title scores."""
        if not self.bm25_content_index or not self.bm25_title_index:
            raise RuntimeError("BM25 indices not initialized. Call setup() first.")
        
        # Tokenize the query for content search
        tokenized_query_content = tokenize(
            query,
            ngrams=2,
            remove_stopwords=True,
            preprocessing_method=PreprocessingMethod.STEMMING
        )
        
        # Tokenize the query for title search
        tokenized_query_title = tokenize(
            query,
            ngrams=1,  # Use unigrams for title matching
            remove_stopwords=False,
            preprocessing_method=PreprocessingMethod.NONE
        )
        
        # Get scores from both indices
        content_scores = self.bm25_content_index.get_scores(tokenized_query_content)
        title_scores = self.bm25_title_index.get_scores(tokenized_query_title)
        
        # Combine scores with title weight (title gets 2x weight)
        title_weight = 1.5
        content_weight = 1.0
        
        combined_scores = []
        for i in range(len(self.corpus)):
            combined_score = (content_weight * content_scores[i] + 
                            title_weight * title_scores[i])
            combined_scores.append(combined_score)
        
        # Get top k documents with their combined scores
        top_k_indices = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for i, (idx, score) in enumerate(top_k_indices):
            # Extract title for display
            title = ""
            if 'title' in self.corpus[idx] and self.corpus[idx]['title']:
                title = self.corpus[idx]['title']
            elif ':' in self.corpus[idx]['content']:
                title = self.corpus[idx]['content'].split(':', 1)[0].strip()
            
            result = {
                'rank': i + 1,
                'doc_id': self.corpus[idx]['doc_id'],
                'content': self.corpus[idx]['content'],
                'title': title,
                'score': score,
                'content_score': content_scores[idx],
                'title_score': title_scores[idx],
                'original_index': idx
            }
            results.append(result)
        
        return results
    
    def _search_colbert(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform ColBERT semantic search."""
        if not self.searcher:
            raise RuntimeError("ColBERT searcher not initialized. Call setup() first.")
        
        # Perform search
        doc_ids, rankings, scores = self.searcher.search(query, k=top_k)
        
        # Prepare results
        results = []
        for i, (doc_id, ranking, score) in enumerate(zip(doc_ids, rankings, scores)):
            result = {
                'rank': i + 1,
                'doc_id': self.corpus[doc_id]['doc_id'],
                'content': self.corpus[doc_id]['content'],
                'score': score,
                'original_index': doc_id
            }
            results.append(result)
        
        return results
    
    def print_results(self, query: str, results: List[Dict[str, Any]]):
        """Print search results in a formatted way."""
        if not results:
            print("No results found.")
            return
        
        search_method_display = "BM25 (Lexical)" if self.search_method == 'bm25' else "ColBERT (Semantic)"
        
        print(f"\n{'='*80}")
        print(f"Search Query: '{query}' | Method: {search_method_display}")
        print(f"Found {len(results)} results:")
        print("="*80)
        
        for result in results:
            print(f"\nRank {result['rank']} (Score: {result['score']:.4f})")
            
            # Show detailed scores for BM25
            if self.search_method == 'bm25' and 'content_score' in result and 'title_score' in result:
                print(f"  Content Score: {result['content_score']:.4f} | Title Score: {result['title_score']:.4f}")
            
            print(f"Doc ID: {result['doc_id']}")
            
            # Show title if available
            if 'title' in result and result['title']:
                print(f"Title: {result['title']}")
            
            print(f"Content: {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
            print("-" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='interactive-search',
        description='Interactive search tool using BM25 or ColBERT retriever for document search'
    )
    
    parser.add_argument('-d', '--dataset', 
                        choices=['musique', 'locomo', 'hotpot', 'twowiki'],
                        required=True,
                        help='Name of the dataset to search (required)')
    
    parser.add_argument('-m', '--method',
                        choices=['bm25', 'colbert'],
                        default='colbert',
                        help='Search method to use: bm25 (lexical) or colbert (semantic) (default: colbert)')
    
    parser.add_argument('-k', '--top-k',
                        type=int,
                        default=5,
                        help='Number of top results to return (default: 5)')
    
    return parser.parse_args()


def main():
    """Main function to run the interactive search tool."""
    # Parse arguments
    args = parse_args()
    
    # Initialize search engine
    search_engine = InteractiveSearchEngine(args.dataset, args.method)
    
    search_method_display = "BM25 (Lexical)" if args.method == 'bm25' else "ColBERT (Semantic)"
    
    print(f"Setting up search engine for {args.dataset} dataset using {search_method_display}...")
    if args.method == 'colbert':
        print("This may take a few minutes for ColBERT indexing...")
    else:
        print("BM25 indexing is usually faster...")
    
    # Set up the search engine
    if not search_engine.setup():
        print("Failed to set up search engine. Exiting.")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("🔍 Interactive Search Tool Ready!")
    print(f"Dataset: {args.dataset}")
    print(f"Search Method: {search_method_display}")
    print(f"Corpus size: {len(search_engine.corpus)} documents")
    print(f"Top-K results: {args.top_k}")
    print("="*80)
    print("\nEnter your search queries below.")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("Type 'help' for usage information.")
    print("-" * 80)
    
    # Interactive search loop
    try:
        while True:
            try:
                # Get user input
                query = input("\n🔍 Enter search query: ").strip()
                
                # Handle special commands
                if query.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\nUsage:")
                    print("- Enter any search query to find relevant documents")
                    print(f"- Current search method: {search_method_display}")
                    print("- Type 'quit' or 'exit' to exit")
                    print("- Type 'help' to see this message")
                    print("- Press Ctrl+C to exit")
                    continue
                
                if not query:
                    print("Please enter a valid search query.")
                    continue
                
                # Perform search
                print("Searching...")
                results = search_engine.search(query, args.top_k)
                
                # Display results
                search_engine.print_results(query, results)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error during search: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
