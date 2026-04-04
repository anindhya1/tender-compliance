import os
import json
import time
from pathlib import Path
from llama_index.core import Document

# --- YOUR IMPORTS ---
import read_checklist as checklist
import convert_to_md as markdown

# --- NEW IMPORTS ---
import local_rag   # The file we created for Chroma/LlamaIndex
import graph_agent # The file we created for LangGraph

# --- CONFIGURATION ---
CHECKLIST_PATH = "data/checklist/bqc-checklist.xlsx"
EVIDENCE_ROOT = "./data/markdown/01_md/bidder-docs"
TENDER_DOCS_ROOT = "./data/markdown/01_md/tender-docs"
OUTPUT_DIR = "./data/reports"

# --- HELPER: Indexing Logic ---
def ensure_index(collection_name: str, text_content: str):
    """
    Ensures the text is indexed in ChromaDB via LlamaIndex.
    If the index is empty, it inserts the document.
    """
    if not text_content:
        print(f"   [Skip] No content to index for {collection_name}")
        return

    print(f"   [Index] Checking index for '{collection_name}'...")
    index = local_rag.get_index(collection_name)
    
    # Simple check: If retrieval returns nothing for a generic query, we assume it needs indexing.
    # (For production, you might want more robust state management)
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("summary")
    
    if not nodes:
        print(f"   [Index] Empty index detected. Inserting documents...")
        doc = Document(text=text_content, metadata={"source": collection_name})
        index.insert(doc)
        print(f"   [Index] Successfully indexed {len(text_content)} chars.")
    else:
        print(f"   [Index] Index already exists. Skipping insertion.")

# --- MAIN EXECUTION ---
def main():
    # 1. Load Rules (Source of Truth)
    try:
        rubric_dict = checklist.extract_rubric_to_dict(CHECKLIST_PATH)
        print(f" Loaded {len(rubric_dict)} requirements from checklist.")
    except Exception as e:
        print(f"CRITICAL: Helper script failed. {e}")
        return

    # 2. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    evidence_root = Path(EVIDENCE_ROOT)

    # 3. Index Tender Docs (Global Context)
    tender_root = Path(TENDER_DOCS_ROOT)
    if tender_root.exists():
        print("\n--- Processing Tender Docs ---")
        tender_text = markdown.get_folder_context(tender_root)
        ensure_index("tender-docs", tender_text)
    else:
        print(f" [Warning] Tender docs folder not found: {TENDER_DOCS_ROOT}")

    # 4. Iterate Projects (Bidders)
    all_results = {}
    for folder in [f for f in evidence_root.iterdir() if f.is_dir()]:
        bidder_name = folder.name
        print(f"\n--- Processing Bidder: {bidder_name} ---")

        # A. Index Bidder Text
        context_text = markdown.get_folder_context(folder)
        if not context_text:
            print("   [Skip] No markdown content found.")
            continue
            
        ensure_index(bidder_name, context_text)

        # B. Create Tools for this specific bidder
        # This connects the Agent to the two specific Chroma collections
        tools = local_rag.create_tools(bidder_name)

        print(f"   Auditing requirements...", end="", flush=True)

        # C. Audit Loop
        results = {}
        for req_id, req_text in rubric_dict.items():
            
            # Run the LangGraph Agent
            try:
                # The agent returns a string that contains JSON
                raw_response = graph_agent.run_audit(req_id, req_text, bidder_name, tools)
                
                # Robust JSON cleanup (Local LLMs sometimes add chatter)
                json_str = raw_response.strip()
                if "```json" in json_str:
                    # Extract content between ```json and ```
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                     json_str = json_str.split("```")[1]
                
                # Remove any leading text before the first '{'
                if "{" in json_str:
                    json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]

                verdict = json.loads(json_str)
                
            except json.JSONDecodeError:
                verdict = {
                    "status": "Error", 
                    "reasoning": "LLM produced invalid JSON.", 
                    "quote": raw_response[:200] # Log raw output for debugging
                }
            except Exception as e:
                verdict = {
                    "status": "Error", 
                    "reasoning": f"Agent failure: {str(e)}", 
                    "quote": ""
                }

            # Store result
            results[req_id] = {
                "requirement": req_text,
                "status": verdict.get("status", "N/A"),
                "reasoning": verdict.get("reasoning", ""),
                "quote": verdict.get("quote", "")
            }
            print(".", end="", flush=True)

        # D. Save Results (One file per bidder)
        json_path = os.path.join(OUTPUT_DIR, f"{bidder_name}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n    Results saved: {json_path}")

        all_results[bidder_name] = results

    # 5. Save Combined Results
    combined_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n All audits complete. Combined results: {combined_path}")

if __name__ == "__main__":
    main()
