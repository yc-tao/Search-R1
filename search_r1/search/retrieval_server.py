import json
import argparse
from typing import List, Dict, Optional, Union

from rank_bm25 import BM25Okapi
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and lowercasing."""
    return text.lower().split()


class Document(BaseModel):
    """Represents a document with an ID and contents."""
    id: str
    contents: str


class QueryRequest(BaseModel):
    """Request model for the /retrieve endpoint."""
    queries: List[str]
    # Accept either:
    # 1) A single shared document list for all queries
    # 2) A list of document lists, one per query (per-query documents)
    documents: Union[List[Document], List[List[Document]]]
    topk: Optional[int] = 3
    return_scores: bool = False


class DynamicBM25Retriever:
    """
    A dynamic BM25 retriever that indexes documents on-the-fly for each query.
    Suitable for small document sets (1-100 documents).
    """

    def __init__(self):
        pass

    def search(
        self,
        query: str,
        documents: List[Document],
        topk: int = 3,
        return_score: bool = False
    ):
        """
        Search for relevant documents given a query.

        Args:
            query: The search query string
            documents: List of documents to search within
            topk: Number of top results to return
            return_score: Whether to return BM25 scores

        Returns:
            If return_score=True: (results, scores)
            If return_score=False: results
        """
        if not documents:
            if return_score:
                return [], []
            return []

        # Tokenize all documents
        doc_texts = [doc.contents for doc in documents]
        tokenized_docs = [tokenize(text) for text in doc_texts]

        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)

        # Tokenize query and get scores
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:topk]

        # Format results
        results = []
        result_scores = []

        for idx in top_indices:
            doc = documents[idx]
            # Parse contents to extract title and text
            content = doc.contents
            lines = content.split("\n")
            title = lines[0].strip('"') if lines else ""
            text = "\n".join(lines[1:]) if len(lines) > 1 else ""

            result_doc = {
                "id": doc.id,
                "title": title,
                "text": text,
                "contents": content
            }
            results.append(result_doc)
            result_scores.append(float(scores[idx]))

        if return_score:
            return results, result_scores
        return results

    def batch_search(
        self,
        query_list: List[str],
        documents: Union[List[Document], List[List[Document]]],
        topk: int = 3,
        return_score: bool = False
    ):
        """
        Batch search for multiple queries with either a shared document set
        or per-query document sets.

        Args:
            query_list: List of query strings
            documents: Shared document list or list of document lists (one per query)
            topk: Number of top results to return per query
            return_score: Whether to return BM25 scores

        Returns:
            If return_score=True: (results, scores)
            If return_score=False: results
        """
        if not query_list:
            return ([], []) if return_score else []

        # Normalize documents so each query has an associated doc list
        use_per_query_docs = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list)
        if use_per_query_docs:
            if len(documents) != len(query_list):
                raise ValueError("Number of document lists must match number of queries when providing per-query documents.")
            doc_sets = documents
        else:
            doc_sets = [documents for _ in query_list]

        all_results = []
        all_scores = []

        for query, doc_set in zip(query_list, doc_sets):
            results, scores = self.search(
                query=query,
                documents=doc_set,
                topk=topk,
                return_score=True
            )
            all_results.append(results)
            all_scores.append(scores)

        if return_score:
            return all_results, all_scores
        return all_results


class Config:
    """Configuration class for the retrieval server."""
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 56321,
        topk: int = 3
    ):
        self.host = host
        self.port = port
        self.topk = topk


# Initialize FastAPI app and retriever
app = FastAPI()
retriever = DynamicBM25Retriever()

# Initialize default config (will be overridden if run as main)
config = Config()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and documents, performs BM25 retrieval.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "documents": [
        {"id": "0", "contents": "\"Python\"\nPython is a programming language..."},
        {"id": "1", "contents": "\"Neural Networks\"\nNeural networks are..."}
      ],
      "topk": 3,
      "return_scores": true
    }
    # Or per-query documents:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "documents": [
        [
          {"id": "0", "contents": "\"Python\"\nPython is a programming language..."}
        ],
        [
          {"id": "1", "contents": "\"Neural Networks\"\nNeural networks are..."}
        ]
      ]
    }

    Output format:
    {
      "result": [
        [
          {"document": {"id": "0", "title": "Python", "text": "...", "contents": "..."}, "score": 2.5},
          {"document": {"id": "1", "title": "...", "text": "...", "contents": "..."}, "score": 1.2}
        ],
        [...]
      ]
    }
    """
    doc_payload = request.documents
    use_per_query_docs = isinstance(doc_payload, list) and len(doc_payload) > 0 and isinstance(doc_payload[0], list)
    if use_per_query_docs:
        doc_count = sum(len(docs) for docs in doc_payload)
    else:
        doc_count = len(doc_payload) if isinstance(doc_payload, list) else 0

    # Validate document count
    if doc_count > 5000:
        raise HTTPException(
            status_code=400,
            detail="Too many documents. Maximum allowed is 5000 documents per request."
        )

    if doc_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents provided. Please include at least one document."
        )

    # Use provided topk or default from config
    topk = request.topk if request.topk else config.topk

    # Normalize documents: allow either a shared doc list or per-query doc lists
    documents = doc_payload
    if use_per_query_docs:
        if len(documents) != len(request.queries):
            raise HTTPException(
                status_code=400,
                detail="When providing per-query documents, the number of document lists must match the number of queries."
            )
    else:
        documents = [documents for _ in request.queries]

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        documents=documents,
        topk=topk,
        return_score=request.return_scores
    )

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)

    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the dynamic BM25 retrieval server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=56321,
        help="Port number to bind the server to."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Default number of retrieved passages for one query."
    )

    args = parser.parse_args()

    # Create global config
    config = Config(
        host=args.host,
        port=args.port,
        topk=args.topk
    )

    # Launch the server
    print(f"Starting BM25 retrieval server on {config.host}:{config.port}")
    print(f"Default top-k: {config.topk}")
    uvicorn.run(app, host=config.host, port=config.port)
