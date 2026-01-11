import json
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer

from gemmaqa.utils import get_logger

logger = get_logger(__name__)


class GemmaQARetriever:
    """
    Retriever class using SentenceTransformers and FAISS.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        """
        Initialize the retriever.

        Args:
            model_name: Name of the sentence-transformer model.
            device: Device to use for encoding ('cuda' or 'cpu').
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.encoder = SentenceTransformer(model_name, device=device)
        
        self.index = None
        self.corpus = []

    def index_corpus(
        self,
        corpus_path: str | Path,
        output_dir: str | Path,
        batch_size: int = 64,
    ):
        """
        Builds FAISS index from corpus.json and saves it.

        Args:
            corpus_path: Path to corpus.json (list of dicts with 'text' field).
            output_dir: Directory to save index and corpus cache.
            batch_size: Batch size for encoding.
        """
        corpus_path = Path(corpus_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        texts = [doc["text"] for doc in self.corpus]
        logger.info(f"Encoding {len(texts)} documents...")

        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True # Normalize for Inner Product (Cosine Similarity)
        )

        d = embeddings.shape[1]
        logger.info(f"Building FAISS index (dim={d})...")
        
        # Inner Product with normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(d) 
        self.index.add(embeddings)

        index_path = output_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Index saved to {index_path}")

    def load_index(self, index_path: str | Path, corpus_path: str | Path):
        """
        Load a pre-built FAISS index and the corresponding corpus.

        Args:
            index_path: Path to faiss_index.bin
            corpus_path: Path to corpus.json
        """
        logger.info(f"Loading index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        
        logger.info(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
            
        if self.index.ntotal != len(self.corpus):
            logger.warning(
                f"Index size ({self.index.ntotal}) != Corpus size ({len(self.corpus)}). "
                "Retrieval results might be misaligned!"
            )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Retrieve top-k contexts for a query.

        Args:
            query: The query text.
            k: Number of results to return.

        Returns:
            List of dicts containing 'text', 'title', 'score'.
        """
        if self.index is None or not self.corpus:
            raise ValueError("Index or corpus not loaded. Call load_index() first.")

        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.corpus):
                doc = self.corpus[idx]
                results.append({
                    "text": doc["text"],
                    "title": doc.get("title", ""),
                    "score": float(score)
                })
        
        return results
