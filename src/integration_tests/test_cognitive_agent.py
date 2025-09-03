"""
Integration tests for the CognitiveAgent through the Orchestrator.

This test module verifies the complete end-to-end behavior of the CognitiveAgent
when integrated with the Orchestrator, including proper OpenAI API interactions
and file output validation.
"""
import os
import unittest
import json
from unittest.mock import MagicMock, patch
from integration_tests.mocks.mock_args import create_agent_args
from integration_tests.mocks.multiprocessing_mocks import MockPool
from integration_tests.mocks.openai_mocks import MockOpenAIClient
from integration_tests.mocks.file_mocks import mock_open, mock_qa_output_path, mock_retrieval_output_path
from integration_tests.mocks.colbert_mocks import create_mock_indexer, create_mock_searcher, create_mock_run, create_mock_run_config, create_mock_colbert_config
from integration_tests.mocks.worker_mocks import create_mock_worker
from integration_tests.integration_test import IntegrationTest
from orchestrator.orchestrator import Orchestrator


class TestCognitiveAgentIntegration(IntegrationTest):
    """Integration tests for CognitiveAgent through Orchestrator."""

    @patch('models.agent.Lock', new=MagicMock)
    @patch('models.agent.Pool', new=MockPool)
    @patch('spacy.load', new=MagicMock)
    @patch('builtins.open', side_effect=mock_open)
    @patch('predictor.predictor.get_qa_output_path', side_effect=mock_qa_output_path)
    @patch('predictor.predictor.get_retrieval_output_path', side_effect=mock_retrieval_output_path)
    @patch('agents.cognitive_agent.cognitive_agent.worker', return_value=create_mock_worker())
    @patch('agents.cognitive_agent.cognitive_agent.Run', new=create_mock_run)
    @patch('agents.cognitive_agent.cognitive_agent.RunConfig', new=create_mock_run_config)
    @patch('agents.cognitive_agent.cognitive_agent.ColBERTConfig', new=create_mock_colbert_config)
    @patch('agents.cognitive_agent.cognitive_agent.Indexer', new=create_mock_indexer)
    @patch('agents.cognitive_agent.cognitive_agent.Searcher', new=create_mock_searcher)
    @patch('azure_open_ai.openai_client.OpenAIClient.get_client', new=MockOpenAIClient)
    # pylint: disable-next=unused-argument
    def test_cognitive_agent(self, mock_worker, *rest):
        """Test ReAct pattern with two iterations: thought+actions -> thought+final_answer."""

        # Configure OpenAI mock for ReAct pattern: first intermediate, then final response
        MockOpenAIClient.configure_responses([
            # First response: thought + actions (intermediate step)
            {
                "thought": "I need to search for information about who performed the album Green.",
                "actions": ["search('Green album performer')"]
            },
            # Second response: thought + final_answer (final step)
            {
                "thought": "Based on the search results, I can see that Steve Hillage performed the \
album Green. Now I need to find his spouse.",
                "final_answer": "Miquette Giraudy"
            }
        ])

        # Configure worker search results
        mock_worker.searcher.search.return_value = ([0, 1], [1, 2], [0.9, 0.8])

        # Create test arguments
        args = create_agent_args(override_args=None)

        # Run the full orchestrator workflow
        orchestrator = Orchestrator(args)
        orchestrator.run()

        # Assert that worker.searcher.search was called with the expected query
        mock_worker.searcher.search.assert_called_with(
            'Green album performer', 
            k=5
        )

        # Assert that openAI was called exactly twice
        self.assertEqual(MockOpenAIClient.get_call_count(), 2)

        # Assert that output files were created
        self._assert_output_files_exist()

        # Assert that output files contain expected content
        self._assert_output_files_content()

    def _assert_output_files_exist(self):
        """Assert that the expected output files were created."""
        qa_output_file = mock_qa_output_path()
        retrieval_output_file = mock_retrieval_output_path()

        self.assertTrue(os.path.exists(qa_output_file),
                        f"QA output file should exist at {qa_output_file}")
        self.assertTrue(os.path.exists(retrieval_output_file),
                        f"Retrieval output file should exist at {retrieval_output_file}")

    def _assert_output_files_content(self):
        """Assert that the output files contain expected content."""
        # Check QA output file content
        qa_output_file = mock_qa_output_path()
        with open(qa_output_file, 'r', encoding='utf-8') as f:
            qa_lines = f.readlines()

        self.assertEqual(len(qa_lines), 1,
                         "QA output should contain exactly one line")

        qa_result = json.loads(qa_lines[0])
        self.assertIn('custom_id', qa_result)
        self.assertIn('response', qa_result)
        self.assertIn('body', qa_result['response'])
        self.assertIn('choices', qa_result['response']['body'])
        self.assertEqual(len(qa_result['response']['body']['choices']), 1)

        # Check that the answer contains expected content
        answer_content = qa_result['response']['body']['choices'][0]['message']['content']
        self.assertIn('Miquette Giraudy', answer_content,
                      "Answer should contain the expected spouse name")

        # Check retrieval output file content
        retrieval_output_file = mock_retrieval_output_path()
        with open(retrieval_output_file, 'r', encoding='utf-8') as f:
            retrieval_lines = f.readlines()

        self.assertEqual(len(retrieval_lines), 1,
                         "Retrieval output should contain exactly one line")

        retrieval_result = json.loads(retrieval_lines[0])
        self.assertIn('custom_id', retrieval_result)
        self.assertIn('question', retrieval_result)
        self.assertIn('result', retrieval_result)
        self.assertEqual(
            retrieval_result['question'], "Who is the spouse of the Green performer?")
        self.assertEqual(retrieval_result['custom_id'], "2hop__460946_294723")


if __name__ == '__main__':
    unittest.main(verbosity=2)
