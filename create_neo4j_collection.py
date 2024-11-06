import chromadb
from chromadb.utils import embedding_functions
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import torch
import json


if __name__ == "__main__":
    COLLECTION = "neo4j_financebench"
    chroma_client = chromadb.PersistentClient("cache/chromadb")

    try:
        chroma_client.delete_collection(COLLECTION)
    except ValueError:
        pass

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    collection = chroma_client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedder,
    )

    JSON_PATH = Path("data", "financebench_neo4j.json")
    with open(JSON_PATH, "rt", encoding="utf-8") as f:
        neo4j_items = json.load(f)

    for id, item in tqdm(enumerate(neo4j_items), total=len(neo4j_items)):
        triples_str = "\n".join(f"({x}, {r}, {y})" for x, r, y in item["triples"])
        
        collection.add(
            documents=item["text"],
            metadatas=[{"triples": triples_str, "document": item["document"]}],
            ids=[f"{item['document']}-{id}"],
        )
