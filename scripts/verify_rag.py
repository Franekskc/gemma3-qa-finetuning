
import sys
from pathlib import Path
import json

# Add src to path just in case
sys.path.append(str(Path.cwd() / "src"))

from gemmaqa.rag.retriever import GemmaQARetriever

def verify_retrieval():
    print("Verifying Retrieval...")
    
    # Ensure data exists (mock if needed or check real data)
    corpus_path = Path("data/corpus.json")
    if not corpus_path.exists():
        print(f"Propably data extraction has not run yet. {corpus_path} not found.")
        return False
        
    index_path = Path("data/faiss_index.bin")
    
    retriever = GemmaQARetriever(device="cpu") # Force CPU for simple verification
    
    if not index_path.exists():
        print("Index not found, creating it...")
        retriever.index_corpus(corpus_path, "data")
    else:
        print("Loading existing index...")
        retriever.load_index(index_path, corpus_path)
        
    # Test query
    query = "What is the capital of France?"
    # We might not have this in corpus, but let's see if it runs
    results = retriever.retrieve(query, k=3)
    
    print(f"Query: {query}")
    for i, res in enumerate(results):
        print(f"Result {i+1} (Score: {res['score']:.4f}): {res['text'][:100]}...")
        
    print("Retrieval verification successful!")
    return True

if __name__ == "__main__":
    verify_retrieval()
