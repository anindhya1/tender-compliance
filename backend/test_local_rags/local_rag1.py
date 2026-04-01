from llama_index.core import VectorStoreIndex, Settings, PromptTemplate, Document, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from docling.document_converter import DocumentConverter
import chromadb
import time

# 1. Embeddings
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

Settings.embed_model = RobustOllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"request_timeout": 1000.0},
    embed_batch_size=1
)

# 2. LLM
Settings.llm = Ollama(model="mistral", request_timeout=1000.0)

# 3. Chunk size
Settings.node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=25)

# 4. Load + Index
print("Loading documents...")
converter = DocumentConverter()
result = converter.convert("../data/dummy/Knowledge_transfer.pdf")
text = result.document.export_to_markdown()
documents = [Document(text=text)]
print(f"Loaded document with {len(text)} characters")

chroma_client = chromadb.PersistentClient(path="../data/chroma_db")
chroma_collection = chroma_client.get_or_create_collection("knowledge-transfer")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if chroma_collection.count() > 0:
    print("Loading existing index from ChromaDB...")
    index = VectorStoreIndex.from_vector_store(vector_store)
else:
    print("Embedding documents (this may take time)...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

# 5. Query
print("Querying...")
qa_prompt = PromptTemplate(
    "You are a helpful assistant. Use ONLY the context below to answer the question. "
    "Do not say you cannot read the file.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

query_engine = index.as_query_engine(
    text_qa_template=qa_prompt,
    similarity_top_k=10
)
response = query_engine.query("What is does tool_3_geographic do?")
print(response)