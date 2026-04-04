import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# --- CONFIGURATION ---
CHROMA_DB_PATH = "./data/chroma_db"
EMBED_MODEL = "nomic-embed-text" # Ensure you have run `ollama pull nomic-embed-text`

def get_local_embedding():
    """Returns your robust embedding model."""
    return OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )

def get_index(collection_name: str):
    """Gets or creates a LlamaIndex vector store from local ChromaDB."""
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get the Chroma collection
    chroma_collection = db.get_or_create_collection(collection_name)
    
    # Connect LlamaIndex to this collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Return the index (loaded from disk)
    return VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context, 
        embed_model=get_local_embedding()
    )

def create_tools(bidder_name: str):
    """Creates TWO tools: one for Tender rules, one for Bidder evidence."""
    
    # 1. Tender Docs Tool
    tender_index = get_index("tender-docs")
    tender_engine = tender_index.as_query_engine(similarity_top_k=5)
    
    tender_tool = QueryEngineTool(
        query_engine=tender_engine,
        metadata=ToolMetadata(
            name="search_tender_requirements",
            description="Search the official tender requirements and rules. Use this to understand what is required."
        )
    )

    # 2. Bidder Docs Tool
    bidder_index = get_index(bidder_name)
    bidder_engine = bidder_index.as_query_engine(similarity_top_k=7)
    
    bidder_tool = QueryEngineTool(
        query_engine=bidder_engine,
        metadata=ToolMetadata(
            name="search_bidder_documents",
            description=f"Search inside {bidder_name}'s submitted documents. Use this to find evidence like certificates, financial stats, or past experience."
        )
    )

    # Convert to LangChain tools so LangGraph can use them
    return [tender_tool.to_langchain_tool(), bidder_tool.to_langchain_tool()]
