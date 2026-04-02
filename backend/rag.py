from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import time


class RobustOllamaEmbedding(OllamaEmbedding):
    def _get_text_embeddings(self, texts):
        results = []
        for text in texts:
            for attempt in range(3):
                try:
                    result = super()._get_text_embeddings([text])
                    results.extend(result)
                    time.sleep(0.1)
                    break
                except Exception as e:
                    print(f"Embedding failed (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to embed text after 3 attempts: {text[:100]}")
        return results


def setup_settings(llm):
    """Configure llama_index global Settings. Call once before querying."""
    Settings.embed_model = RobustOllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"request_timeout": 1000.0},
        embed_batch_size=1
    )
    Settings.llm = llm


def query_collection(collection_name: str, query: str, chroma_db_path: str) -> str:
    """Query a ChromaDB collection and return the response as a string."""
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(similarity_top_k=10)
    response = query_engine.query(query)
    return str(response)
