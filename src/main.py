import asyncio
from pprint import pprint

import dotenv
from datasets import DatasetDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag import load_squad_dataset, to_langchain_dataset, create_database_instance, create_rag_agent


dotenv.load_dotenv()

async def test_rag():
    print("Loading dataset")
    data_set: DatasetDict = load_squad_dataset()

    ## parse to langchain format, we don't use text_splitter because the questions are small enough to fit the context size
    train_subset = DatasetDict({
        "train": data_set["train"].select(range(100))
    })

    langchain_documents: list[Document] = to_langchain_dataset(train_subset)
    print("Loaded Langchain documents")

    doc: Document = langchain_documents[0]
    print("Sample Document:")
    pprint(doc.page_content[:200] + "...")
    pprint(doc.metadata)

    ## create embeddings_model
    embeddings_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    print("Created embeddings model")

    ## instantiate database with embeddings
    db: FAISS = create_database_instance(embeddings_model, langchain_documents)
    print("Created database instance")

    print("\n--- Running Retrieval Sanity Check ---")

    # 1. Grab a "Ground Truth" example from the raw SQuAD dataset
    sample_row = data_set["train"][0]
    test_question = sample_row["question"]
    expected_context_snippet = sample_row["context"][:100]  # First 100 chars

    print(f"❓ Test Question: '{test_question}'")

    # 2. Perform a raw vector search
    results = db.similarity_search(test_question, k=1)

    # 3. Validation
    if not results:
        print("❌ FAILURE: No documents retrieved.")
    else:
        retrieved_doc = results[0]
        retrieved_text = retrieved_doc.page_content

        # Check if the retrieved text matches what SQuAD says is the right context
        if expected_context_snippet in retrieved_text:
            print(f"✅ SUCCESS: Database retrieved the correct context!")
            print(f"   Context found: '{retrieved_text[:60]}...'")
        else:
            print(f"⚠️ MISMATCH: Database retrieved a different document.")
            print(f"   Expected: {expected_context_snippet}...")
            print(f"   Got:      {retrieved_text[:60]}...")

    print("------------------------------------------\n")

    ## create retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


    agent = create_rag_agent(retriever)
    print("Created agent")

    # ask questions
    while True:
        try:
            user_input_raw = await asyncio.to_thread(input, "You: ")
            user_input = user_input_raw.strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            response = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
            )
            print(f"AI: {response["messages"][-1]}")
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    try:
        asyncio.run(test_rag())
    except KeyboardInterrupt:
        print("\nExiting...")