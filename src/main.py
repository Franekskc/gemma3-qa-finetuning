import asyncio
from pprint import pprint

import dotenv
from datasets import DatasetDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag import load_squad_dataset, to_langchain_dataset, create_database_instance
from rag.rag import crate_rag_agent


dotenv.load_dotenv()

async def test_rag():
    ## load dataset
    data_set: DatasetDict = load_squad_dataset()

    ## parse to langchain format, we dont use text_splitter because the questions are small enough to fit the context size
    langchain_documents: list[Document] = to_langchain_dataset(data_set)
    print("Loaded Langchain documents")

    ## create embeddings_model
    embeddings_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    print("Created embeddings model")

    ## instantiate database with embeddings
    db: FAISS = create_database_instance(embeddings_model, langchain_documents[:100])
    print("Created database instance")

    ## create retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


    agent = crate_rag_agent(retriever)
    print("Created agent")

    # ask questions
    while True:
        # Use asyncio.to_thread to run the blocking input() in a separate thread
        # This prevents it from blocking the entire event loop.
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
            pprint(f"AI: {response}")
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    try:
        asyncio.run(test_rag())
    except KeyboardInterrupt:
        print("\nExiting...")