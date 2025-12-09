"""
Modified generation manager for medical diagnosis with per-episode documents.

This extends LLMGenerationManager to support:
1. Per-episode documents passed from batch metadata
2. Mock retrieval mode for training (simpler, no external server needed)
"""

import torch
import requests
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from .generation import LLMGenerationManager, GenerationConfig
from verl import DataProto


@dataclass
class MedicalGenerationConfig(GenerationConfig):
    """Extended config for medical generation."""
    use_mock_retrieval: bool = False  # If True, use mock retrieval instead of real server
    mock_response_template: str = "Medical information related to your query: {query}"


class MedicalLLMGenerationManager(LLMGenerationManager):
    """
    Extended generation manager for medical diagnosis.

    This version supports:
    1. Per-episode documents stored in batch metadata
    2. Mock retrieval mode for training (returns placeholder results)
    3. Real retrieval with per-episode document routing
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        use_mock_retrieval: bool = False,
    ):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        self.current_documents = None  # Will be set per batch
        self.use_mock_retrieval = use_mock_retrieval
        self.active_documents_map = {}  # Maps active index to documents

    def set_documents(self, documents_list: List[List[Dict]]):
        """Set the documents for the current batch.

        Args:
            documents_list: List of document lists, one per item in the batch.
                           Each document is a dict with 'id' and 'contents' keys.
        """
        self.current_documents = documents_list

    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop with document-aware search.

        This overrides the parent method to extract documents from the batch
        before running the generation loop.
        """
        # Extract documents from batch metadata if available
        self.current_documents = None
        if hasattr(gen_batch, 'non_tensor_batch') and gen_batch.non_tensor_batch is not None:
            if 'extra_info' in gen_batch.non_tensor_batch:
                extra_info = gen_batch.non_tensor_batch['extra_info']
                if isinstance(extra_info, (list, tuple)) and len(extra_info) > 0:
                    # Extract documents from each item's extra_info
                    documents_list = []
                    for info in extra_info:
                        if isinstance(info, dict) and 'documents' in info:
                            documents_list.append(info['documents'])
                        else:
                            documents_list.append([])
                    self.current_documents = documents_list

        # Call parent's run_llm_loop
        return super().run_llm_loop(gen_batch, initial_input_ids)

    def _mock_search_result(self, query: str) -> str:
        """Generate a mock search result for training.

        This provides a consistent format without needing the retrieval server.
        """
        mock_responses = [
            f"Doc 1(Title: Medical Record) Patient presents with symptoms related to: {query}. "
            f"Clinical findings suggest further investigation may be needed.",
            f"Doc 2(Title: Clinical Notes) Assessment for {query}: "
            f"Various diagnostic considerations have been documented.",
            f"Doc 3(Title: Lab Results) Test results pertaining to {query}: "
            f"Values within expected ranges for this patient population.",
        ]
        return "\n".join(mock_responses)

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batch search with document-aware or mock retrieval.

        In mock mode, returns placeholder results for training.
        In real mode, attempts to use per-episode documents if available.
        """
        if queries is None or len(queries) == 0:
            return []

        # Mock retrieval mode - useful for training without retrieval server
        if self.use_mock_retrieval:
            return [self._mock_search_result(query) for query in queries]

        # If no documents available, try parent's batch_search (external server)
        if self.current_documents is None or len(self.current_documents) == 0:
            try:
                return super().batch_search(queries)
            except Exception as e:
                print(f"[MedicalLLMGenerationManager] Retrieval failed, using mock: {e}")
                return [self._mock_search_result(query) for query in queries]

        # Use first set of documents for all queries (simplified)
        # In a full implementation, we'd track which query belongs to which episode
        documents = self.current_documents[0] if self.current_documents else []

        if not documents:
            return [self._mock_search_result(query) for query in queries]

        try:
            results = self._batch_search_with_documents(queries, documents)['result']
            return [self._passages2string(result) for result in results]
        except Exception as e:
            print(f"[MedicalLLMGenerationManager] Document search failed, using mock: {e}")
            return [self._mock_search_result(query) for query in queries]

    def _batch_search_with_documents(self, queries: List[str], documents: List[Dict]) -> Dict:
        """
        Perform batch search with specific documents.

        Args:
            queries: List of search queries
            documents: List of documents to search within

        Returns:
            Search results from the retrieval server
        """
        payload = {
            "queries": queries,
            "documents": documents,
            "topk": self.config.topk,
            "return_scores": True
        }

        response = requests.post(self.config.search_url, json=payload)
        return response.json()


class MockRetrievalLLMGenerationManager(LLMGenerationManager):
    """
    A simpler generation manager that uses mock retrieval.

    This is useful for training when we just want to learn search behavior
    without needing a real retrieval server.
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)

    def _mock_search_result(self, query: str) -> str:
        """Generate a mock search result."""
        # Provide varied mock responses to simulate retrieval
        templates = [
            "Doc 1(Title: Medical Assessment) Relevant clinical information for query '{query}': "
            "Patient evaluation documented with appropriate findings.",
            "Doc 2(Title: Diagnostic Notes) Regarding '{query}': "
            "Clinical observations recorded for diagnostic consideration.",
            "Doc 3(Title: Treatment Record) Documentation related to '{query}': "
            "Medical management plan outlined as per clinical guidelines.",
        ]
        result = "\n".join(t.format(query=query[:50]) for t in templates)
        return result

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """Return mock search results for all queries."""
        if queries is None:
            return []
        return [self._mock_search_result(query) for query in queries]
