from datasets import load_dataset, DatasetDict
from langchain_core.documents import Document


def load_squad_dataset() -> DatasetDict:
    dataset: DatasetDict = load_dataset("squad")
    return dataset


def to_langchain_dataset(dataset: DatasetDict) -> list[Document]:
    # 1. Create a dictionary to track unique contexts while keeping metadata
    # Format: { "context_string": "title_string" }
    unique_contexts = {}

    print("Processing dataset to remove duplicates...")
    for row in dataset["train"]:
        context = row["context"]
        # If we haven't seen this context yet, save it along with its title
        if context not in unique_contexts:
            unique_contexts[context] = row["title"]

    # 2. Transform to Langchain Document format
    langchain_documents: list[Document] = []

    for context, title in unique_contexts.items():
        doc = Document(
            page_content=context,
            metadata={
                "title": title,
                "source": "squad_train"
            }
        )
        langchain_documents.append(doc)

    print(f"Loaded {len(langchain_documents)} unique contexts (reduced from {len(dataset['train'])} rows).")

    return langchain_documents