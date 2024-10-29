from typing import List, Dict, Any
import uuid
from .core import Node
import chromadb
from chromadb.utils import embedding_functions

class TextVectorStore(Node):
    def __init__(
        self,
        name: str,
        index_name: str,
        chroma_db_path: str,
        embedding_model: str,
        top_k_results: int,
    ):
        super().__init__(name)
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.create_collection(
            name=index_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedder,
            get_or_create=True,
        )
        self.top_k_results = top_k_results

    def __call__(self, documents: List[Dict[str, str]]) -> None:
        self.add(documents)

    def add(self, documents: List[Dict[str, str]]):
        ids = [str(uuid.uuid4()) for _ in documents]
        self.collection.add(
            documents=[doc["content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
            ids=ids,
        )
        print(f"Added {len(documents)} documents to the vector store")

    def search(self, query: str, *args, **kwargs) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query], n_results=self.top_k_results, *args, **kwargs
        )
        print(f"Retrieved {len(results['documents'][0])} documents from vector store")
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]