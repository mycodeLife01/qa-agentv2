import chromadb
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model=os.getenv("AGENT_MODEL"), temperature=1.0, thinking_level="minimal"
)

SYSTEM_PROMPT = """
You are a helpful assistant with a essential tool called 'search_vdb', ALWAYS use this tool to get relevant context before responding to any user query
"""


def init_vector_database():
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST"), port=os.getenv("CHROMA_PORT")
    )
    collection = chroma_client.get_or_create_collection(
        os.getenv("CHROMA_COLLECTION_NAME")
    )
    return ChromaVectorStore(chroma_collection=collection)


vector_store = init_vector_database()


@tool
def search_vdb(query: str) -> str:
    """A search tool to get relevant context from vector database
    Args:
        query (str): The query to search for
    Returns:
        str: The relevant context
    """
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=GoogleGenAIEmbedding(
                model_name=os.getenv("EMBEDDING_MODEL"),
                embed_batch_size=100,
            ),
        )
        query_engine = index.as_query_engine(
            llm=GoogleGenAI(model=os.getenv("LLAMA_RESPONSE_MODEL"))
        )
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        print(f"[ERROR]: search_vdb, {e}")
        return "System Instruction: Tool calling failed"


agent = create_agent(
    model=model,
    tools=[search_vdb],
    system_prompt=SYSTEM_PROMPT,
)
