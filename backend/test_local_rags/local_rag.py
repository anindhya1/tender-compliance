import os
# Must be at the very top
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Local LLM via Ollama
Settings.llm = Ollama(model="mistral", request_timeout=120.0)

# 2. Local Embedding with explicit cache folder
# If you run this once while online, it saves to this folder forever.
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_folder="./model_cache" 
)

# 3. Load structured folders
documents = SimpleDirectoryReader("./data/dummy", recursive=True).load_data()

# 4. Create and Query
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("Based on my folders, what are the tools used in this project?")
print(response)
