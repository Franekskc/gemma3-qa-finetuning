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

    def _load_and_format_corpus(self, corpus_path: str | Path) -> list[dict]:
        """
        Load and format corpus. If items have 'context', 'question', 'answers',
        format them into a single text block.
        """
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        formatted_corpus = []
        for item in raw_data:
            # Check if it's a SQuAD-style item
            if "context" in item and "question" in item and "answers" in item:
                # Extract first answer
                answers = item["answers"]
                if isinstance(answers, dict) and "text" in answers:
                    first_answer = answers["text"][0] if answers["text"] else ""
                elif isinstance(answers, list) and len(answers) > 0:
                     # Fallback if answers is just a list of strings
                    first_answer = answers[0]
                else:
                    first_answer = ""
                
                # Format:
                # Context: ...
                # Question: ...
                # Answer: ...
                text_block = (
                    f"Context: {item['context']}\n"
                    f"Question: {item['question']}\n"
                    f"Answer: {first_answer}"
                )
                
                # Create a new item ensuring 'text' is the formatted block
                new_item = item.copy()
                new_item["text"] = text_block
                formatted_corpus.append(new_item)
            
            elif "text" in item:
                formatted_corpus.append(item)
            else:
                logger.warning(f"Item missing required fields (context/question/answers OR text): {item.keys()}")
        
        return formatted_corpus

    def index_corpus(
        self,
        corpus_path: str | Path,
        output_dir: str | Path,
        batch_size: int = 64,
    ):
        """
        Builds FAISS index from corpus JSON and saves it.

        Args:
            corpus_path: Path to dataset JSON (e.g. train_subset.json).
            output_dir: Directory to save index.
            batch_size: Batch size for encoding.
        """
        corpus_path = Path(corpus_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading and formatting corpus from {corpus_path}...")
        self.corpus = self._load_and_format_corpus(corpus_path)

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
            corpus_path: Path to dataset JSON (e.g. train_subset.json)
        """
        logger.info(f"Loading index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        
        logger.info(f"Loading corpus from {corpus_path}...")
        self.corpus = self._load_and_format_corpus(corpus_path)
            
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
