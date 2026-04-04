import os
import json
import time
import chromadb
from pathlib import Path
from openpyxl import load_workbook
from pydantic import BaseModel, Field
from llama_index.llms.ollama import Ollama

# --- YOUR IMPORTS ---
import read_checklist as checklist
import convert_to_md as markdown
import rag

# --- CONFIGURATION ---
CHECKLIST_PATH = "data/checklist/bqc-checklist.xlsx"
EVIDENCE_ROOT = "./data/markdown/01_md/bidder-docs"
TENDER_DOCS_ROOT = "./data/markdown/01_md/tender-docs"
OUTPUT_DIR = "./data/reports"
PROMPT_PATH = "./prompt.md"

# AI Settings
LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 1000

with open(PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()


# --- PYDANTIC MODELS ---
class AuditVerdict(BaseModel):
    status: str = Field(description="Pass, Fail, or N/A")
    reasoning: str = Field(description="Brief reasoning")
    quote: str = Field(description="Exact quote from the snippet")

# --- EMBEDDING MODEL ---
class OllamaChromaEmbeddingFunction:
    """ChromaDB-compatible wrapper around RobustOllamaEmbedding (defined in rag.py)."""
    def __init__(self):
        self.model = rag.RobustOllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"request_timeout": 1000.0},
            embed_batch_size=1
        )

    def __call__(self, input):
        return self.model.get_text_embedding_batch(input)

# --- 1. MEMORY ENGINE (ChromaDB) ---
CHROMA_DB_PATH = "./data/chroma_db"

class LocalMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embed_fn = OllamaChromaEmbeddingFunction()

    def index_text(self, project_name: str, full_text: str):
        """Splits text and saves to vector DB. Skips if collection already exists."""
        existing = [c.name for c in self.client.list_collections()]
        if project_name in existing:
            print(f"    Loading existing index for '{project_name}'...")
            return self.client.get_collection(project_name)

        print(f"    Indexing '{project_name}' for the first time...")
        collection = self.client.create_collection(
            name=project_name,
            embedding_function=self.embed_fn
        )

        paragraphs = full_text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) < CHUNK_SIZE:
                current_chunk += p + "\n\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        if current_chunk: chunks.append(current_chunk.strip())

        if chunks:
            collection.add(
                documents=chunks,
                ids=[f"id_{i}" for i in range(len(chunks))]
            )
        return collection

# --- 2. AI AUDITOR ---
def audit_single_requirement(llm: Ollama, bidder_name: str, req_id: str, req_text: str) -> AuditVerdict:
    # Retrieve context directly via rag.py
    tender_snippets = rag.query_collection("tender-docs", req_text, CHROMA_DB_PATH)
    bidder_snippets = rag.query_collection(bidder_name, req_text, CHROMA_DB_PATH)

    prompt = f"""
    {SYSTEM_PROMPT}

    ---

    REQUIREMENT ({req_id}): "{req_text}"

    TENDER CONTEXT:
    {tender_snippets}

    BIDDER DOCUMENTS:
    {bidder_snippets}
    """

    try:
        response = llm.complete(prompt)
        raw = response.text.strip()

        print(f"\n[RAW LLM RESPONSE - {req_id}]\n{raw}\n[END]\n")

        # JSON Cleanup
        if "```" in raw:
            start = raw.find("{", raw.find("```"))
            end = raw.rfind("}") + 1
            raw = raw[start:end]
        elif "{" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            raw = raw[start:end]

        data = json.loads(raw)
        return AuditVerdict(**data)
    except Exception as e:
        return AuditVerdict(status="Error", reasoning=str(e), quote="")

# --- MAIN EXECUTION ---
def main():
    # 1. Load Rules (Source of Truth)
    # We use your script to get the valid list of IDs and Requirements
    try:
        rubric_dict = checklist.extract_rubric_to_dict(CHECKLIST_PATH)
        print(f" Loaded {len(rubric_dict)} requirements from dictionary.")
    except Exception as e:
        print(f"CRITICAL: Helper script failed. {e}")
        return

    # 2. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    memory = LocalMemory()
    llm = Ollama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, request_timeout=1000.0)
    rag.setup_settings(llm)
    evidence_root = Path(EVIDENCE_ROOT)

    # 3. Index tender docs into a single collection
    tender_root = Path(TENDER_DOCS_ROOT)
    if tender_root.exists():
        print("\n--- Indexing Tender Docs ---")
        tender_text = markdown.get_folder_context(tender_root)
        if tender_text:
            try:
                memory.index_text("tender-docs", tender_text)
                print("Tender docs indexed.")
            except Exception as e:
                print(f"   [Error] Tender docs indexing failed: {e}")
        else:
            print("   [Skip] No markdown content found in tender-docs.")
    else:
        print(f"   [Warning] Tender docs folder not found: {TENDER_DOCS_ROOT}")

    # 5. Iterate Projects
    all_results = {}
    for folder in [f for f in evidence_root.iterdir() if f.is_dir()]:
        print(f"\n--- Processing: {folder.name} ---")

        # A. Index Text
        context_text = markdown.get_folder_context(folder)
        if not context_text: continue

        try:
            memory.index_text(folder.name, context_text)
        except Exception as e:
            print(f"   [Error] Indexing failed: {e}")
            continue

        print(f"   Auditing requirements...", end="", flush=True)

        # B. Audit Loop — iterate rubric directly
        results = {}
        for req_id, req_text in rubric_dict.items():
            verdict = audit_single_requirement(llm, folder.name, req_id, req_text)
            results[req_id] = {
                "requirement": req_text,
                "status": verdict.status,
                "reasoning": verdict.reasoning,
                "quote": verdict.quote
            }
            print(".", end="", flush=True)

        # C. Save results to JSON immediately (one file per bidder)
        json_path = os.path.join(OUTPUT_DIR, f"{folder.name}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n   Results saved: {json_path}")

        all_results[folder.name] = results

    # 8. Save combined JSON for all bidders
    combined_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Combined results saved: {combined_path}")

if __name__ == "__main__":
    main()
