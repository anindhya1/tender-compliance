from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# 1. Use Ollama for Embeddings (Batch size 1 prevents crashes)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"request_timeout": 360.0}, # 6 minutes
    embed_batch_size=1 # CRITICAL FIX: Process 1 chunk at a time
)

# 2. Use Ollama for the LLM
Settings.llm = Ollama(
    model="mistral", 
    request_timeout=360.0
)

# 3. Load Documents
print("Loading documents...")
documents = SimpleDirectoryReader("./data/dummy").load_data()

# 4. Create Index (This will be slower but won't crash)
print("Embedding documents (this may take time)...")
index = VectorStoreIndex.from_documents(
    documents, 
    show_progress=True # Shows a progress bar so you know it's working
)

# 5. Query
print("Querying...")
query_engine = index.as_query_engine()
response = query_engine.query("What does this document say?")
print(response)
