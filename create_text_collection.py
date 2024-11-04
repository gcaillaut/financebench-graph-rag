import chromadb
from chromadb.utils import embedding_functions
from torch.utils.data import DataLoader
from pdf import extract_chunks
from pathlib import Path
from tqdm import tqdm
import torch


if __name__ == "__main__":
    COLLECTION = "text_financebench"
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
        name=COLLECTION, metadata={"hnsw:space": "cosine"},
        embedding_function=embedder,
    )

    DATA_DIR = Path("data", "financebench")
    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    
    def _collator(batch):
        res = []
        for path in batch:
            try:
                chunks = extract_chunks(path)
                res.append({
                    "document": path.stem,
                    "path": path,
                    "chunks": chunks,
                })
            except:
                pass
        return res
    
    loader = DataLoader(pdf_files, collate_fn=_collator, batch_size=1, num_workers=6, shuffle=False)
    for batch in tqdm(loader):
        for x in batch:
            collection.add(
                documents=x["chunks"],
                metadatas=[{"path": str(x["path"]), "document": x["document"]}] * len(x["chunks"]),
                ids=[
                    f"{x['document']}-{chunkid}"
                    for chunkid in range(len(x["chunks"]))
                ],
            )
