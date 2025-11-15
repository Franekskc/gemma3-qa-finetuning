from datasets import load_dataset, DatasetDict
from langchain_core.documents import Document


def load_squad_dataset() -> DatasetDict:
    dataset: DatasetDict = load_dataset("squad")
    example = dataset["train"][0]

    print("--- CONTEXT ---")
    print(example["context"])
    print("\n--- QUESTION ---")
    print(example["question"])
    print("\n--- ANSWERS ---")
    print(example["answers"])

    return dataset

def to_langchain_dataset(dataset: DatasetDict) -> list[Document]:
    # load unique contexts - the same contexts can be used in multiple questions
    all_unique_contexts = set(dataset["train"]["context"])

    # transform to Langchain Document format
    langchain_documents: list[Document] = [
        Document(page_content=context) for context in all_unique_contexts
    ]

    print(f"Loaded {len(langchain_documents)} unique contexts.")

    return langchain_documents