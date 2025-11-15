from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import create_retriever_tool, Tool


def create_database_instance(embeddings_model: Embeddings, langchain_documents: list[Document],
                             save_path="faiss_squad_index") -> FAISS:
    vector_store = FAISS.from_documents(langchain_documents, embeddings_model)
    vector_store.save_local(save_path)

    return vector_store


def crate_rag_agent(retriever):
    retriever_tool = get_retriever_tool(retriever)
    agent = create_agent(model="openai:gpt-4o-mini", system_prompt="You are an assistant for question-answering tasks. You MUST use the 'retriever' tool to find context for EVERY user question. DO NOT answer any question from your own internal knowledge. If the retriever finds no relevant information, just say 'I do not have that information.'", tools=[retriever_tool])
    return agent


def get_retriever_tool(retriever)-> Tool:
    return create_retriever_tool(
        retriever,
        "retrieve_squad_context",
        "Search and return information context relevant to the question.",
    )
