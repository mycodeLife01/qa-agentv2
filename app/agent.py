import chromadb
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, ToolRuntime
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank


load_dotenv()

_model = ChatGoogleGenerativeAI(
    model=os.getenv("AGENT_MODEL"), temperature=1.0, thinking_level="minimal"
)
_llama_llm = GoogleGenAI(model=os.getenv("AGENT_MODEL"))

SYSTEM_PROMPT = """
You are a helpful assistant with a essential tool called 'search_vdb', You MUST stick to the following rules:
1. ALWAYS use the tool 'search_vdb' to get relevant context before responding to any user query.
2. ALWAYS put the **original user query** as the argument into the tool.
3. DO NOT make up any context if receiving any error message from the tool and clarify it to the user.
"""


# init chromadb during module initialization, app would be terminated if any error occurred
def _init_vector_database():
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST"), port=os.getenv("CHROMA_PORT")
    )
    collection = chroma_client.get_or_create_collection(
        os.getenv("CHROMA_COLLECTION_NAME")
    )
    return ChromaVectorStore(chroma_collection=collection)


_vector_store = _init_vector_database()

_index = None
_response_synthesizer = None


@tool
def search_vdb(query: str, runtime: ToolRuntime) -> str:
    """A search tool to get relevant context from vector database
    Args:
        query (str): The query to search for
    Returns:
        str: The relevant context
    """
    global _index, _response_synthesizer

    try:
        # Lazy initialization: cache index and response_synthesizer
        if _index is None:
            _index = VectorStoreIndex.from_vector_store(
                vector_store=_vector_store,
                embed_model=GoogleGenAIEmbedding(
                    model_name=os.getenv("EMBEDDING_MODEL"),
                    embed_batch_size=100,
                ),
            )

        if _response_synthesizer is None:
            _response_synthesizer = get_response_synthesizer(llm=_llama_llm)

        # Dynamic filters (cannot be cached)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="doc_content_hash",
                    value=runtime.context.doc_content_hash,
                )
            ]
        )

        # Two-stage retrieval: fetch more candidates, then rerank
        retriever = VectorIndexRetriever(
            index=_index, similarity_top_k=10, filters=filters
        )

        reranker = CohereRerank(
            api_key=os.getenv("COHERE_API_KEY"), top_n=4, model="rerank-v4.0-fast"
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=_response_synthesizer,
            node_postprocessors=[reranker],
        )

        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        print(f"[ERROR]: search_vdb, {e}")
        return "System Instruction: Tool calling failed"


@dataclass
class Context:
    doc_content_hash: str


agent = create_agent(
    model=_model,
    tools=[search_vdb],
    system_prompt=SYSTEM_PROMPT,
    context_schema=Context,
)
