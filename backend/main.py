import os
import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from openpyxl import load_workbook
from pydantic import BaseModel, Field
from llama_index.llms.ollama import Ollama

# --- YOUR IMPORTS ---
import read_checklist as checklist
import convert_to_md as markdown

# --- CONFIGURATION ---
CHECKLIST_PATH = "data/checklist/bqc-checklist.xlsx"
EVIDENCE_ROOT = "./data/markdown/01_md/bidder-docs"
TENDER_DOCS_ROOT = "./data/markdown/01_md/tender-docs"
OUTPUT_DIR = "./data/reports"

# AI Settings
LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 1000       
RETRIEVAL_COUNT = 25     

# Excel Column Definitions
COL_ID = 1; COL_REQ = 2; COL_STATUS = 3; COL_EVIDENCE = 4

# --- PYDANTIC MODELS ---
class AuditVerdict(BaseModel):
    status: str = Field(description="Pass, Fail, or N/A")
    reasoning: str = Field(description="Brief reasoning")
    quote: str = Field(description="Exact quote from the snippet")

# --- 1. MEMORY ENGINE (ChromaDB) ---
CHROMA_DB_PATH = "./data/chroma_db"

class LocalMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

    def index_text(self, project_name: str, full_text: str):
        """Splits text and saves to vector DB. Skips if collection already exists."""
        existing = [c.name for c in self.client.list_collections()]
        if project_name in existing:
            print(f"    Loading existing index for '{project_name}'...")
            return self.client.get_collection(project_name, embedding_function=self.embed_fn)

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
def audit_single_requirement(llm: Ollama, bidder_collection, tender_collection, req_id: str, req_text: str) -> AuditVerdict:
    # Retrieve tender context (what the requirement means)
    tender_results = tender_collection.query(query_texts=[req_text], n_results=RETRIEVAL_COUNT)
    tender_snippets = "\n---\n".join(tender_results['documents'][0])

    # Retrieve bidder evidence
    bidder_results = bidder_collection.query(query_texts=[req_text], n_results=RETRIEVAL_COUNT)
    bidder_snippets = "\n---\n".join(bidder_results['documents'][0])

    # Prompt
    prompt = f"""
    You are a strict Compliance Auditor.
    TASK: Check if the BIDDER DOCUMENTS satisfy the REQUIREMENT.

    REQUIREMENT ({req_id}): "{req_text}"

    TENDER CONTEXT (what this requirement means):
    {tender_snippets}

    BIDDER DOCUMENTS (evidence to evaluate):
    {bidder_snippets}

    INSTRUCTIONS:
    1. Use the TENDER CONTEXT to understand what the requirement is asking for.
    2. Evaluate the BIDDER DOCUMENTS strictly against that requirement.
    3. If the bidder's documents satisfy the requirement, Status is 'Pass'.
    4. If the bidder's documents contradict or fail the requirement, Status is 'Fail'.
    5. If there is no relevant evidence in the bidder's documents, Status is 'N/A'.
    6. Respond ONLY with valid JSON.
    {{
        "status": "Pass" | "Fail" | "N/A",
        "reasoning": "brief explanation",
        "quote": "exact text from bidder documents"
    }}
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

    # 5. Prepare combined workbook
    out_path = os.path.join(OUTPUT_DIR, "Report_All.xlsx")
    wb = load_workbook(CHECKLIST_PATH)
    template_ws = wb.active
    template_ws.title = "_template"

    # 6. Load tender-docs collection for all audits
    try:
        tender_collection = memory.client.get_collection("tender-docs", embedding_function=memory.embed_fn)
    except Exception as e:
        print(f"CRITICAL: Could not load tender-docs collection: {e}")
        return

    # 7. Iterate Projects
    for folder in [f for f in evidence_root.iterdir() if f.is_dir()]:
        print(f"\n--- Processing: {folder.name} ---")

        # A. Index Text
        context_text = markdown.get_folder_context(folder)
        if not context_text: continue

        try:
            bidder_collection = memory.index_text(folder.name, context_text)
        except Exception as e:
            print(f"   [Error] Indexing failed: {e}")
            continue

        # B. Add a sheet for this project (copy of template)
        ws = wb.copy_worksheet(template_ws)
        ws.title = folder.name[:31]  # Excel sheet names max 31 chars

        print(f"   Auditing requirements...", end="", flush=True)

        # C. The Hybrid Loop
        # We iterate Excel rows to find WHERE to write,
        # but we use rubric_dict to decide WHAT to write.
        for row in ws.iter_rows(min_row=2):
            cell_id_val = row[COL_ID - 1].value

            if cell_id_val:
                req_id = str(cell_id_val)

                # CHECK: Is this ID in our valid dictionary?
                if req_id in rubric_dict:
                    # Use text from Dictionary (Source of Truth)
                    req_text = rubric_dict[req_id]

                    # Run Audit
                    verdict = audit_single_requirement(llm, bidder_collection, tender_collection, req_id, req_text)

                    # Write to Excel
                    row[COL_STATUS - 1].value = verdict.status
                    row[COL_EVIDENCE - 1].value = f"{verdict.reasoning}\n\nRef: \"{verdict.quote}\""

                    print(".", end="", flush=True)

        print(f"\nSheet added: {folder.name}")

    # 8. Remove template sheet and save combined report
    wb.remove(template_ws)
    wb.save(out_path)
    print(f"\n Combined report saved: {out_path}")

if __name__ == "__main__":
    main()
