# Tender Compliance Auditor

An automated system for evaluating bidder documents against tender qualification criteria using local LLMs and RAG (Retrieval-Augmented Generation).

---

## What It Does

Given a set of bidder submissions and a tender specification, the system automatically audits each bidder's documents against a BQC (Bid Qualification Criteria) checklist and produces an Excel report with Pass / Fail / N/A verdicts for every requirement.

---

## Project Structure

```
tender/
└── backend/
    ├── main.py               # Primary pipeline (ChromaDB RAG)
    ├── test_agent.py         # Experimental pipeline (LlamaIndex ReActAgent)
    ├── convert_to_md.py      # Converts PDFs to markdown + get_folder_context()
    ├── read_checklist.py     # Reads BQC checklist Excel into a requirements dict
    ├── prompt.md             # System prompt for the compliance auditor LLM
    ├── pyproject.toml        # Project dependencies (managed with uv)
    ├── data/
    │   ├── raw/01/           # Raw PDF input files
    │   ├── markdown/
    │   │   └── 01_md/
    │   │       ├── tender-docs/    # Tender specification markdown files
    │   │       └── bidder-docs/    # Per-bidder markdown files
    │   │           ├── Braithwate/
    │   │           ├── Gourika/
    │   │           ├── Modern/
    │   │           ├── Texmaco/
    │   │           └── Titagarh/
    │   ├── checklist/
    │   │   └── bqc-checklist.xlsx  # BQC requirements checklist
    │   ├── chroma_db/        # Persistent ChromaDB vector store (auto-generated)
    │   ├── agent_indices/    # Persistent LlamaIndex vector indices (auto-generated)
    │   └── reports/          # Output Excel reports (auto-generated)
    └── model_cache/          # Cached HuggingFace embedding models
```

---

## How It Works

### Pipeline (`main.py`)

1. **Load checklist** — `read_checklist.py` reads `bqc-checklist.xlsx` and returns a `{req_id: req_text}` dictionary
2. **Convert PDFs** — `convert_to_md.py` converts raw PDFs to markdown (run once)
3. **Index documents** — Tender docs and each bidder's docs are chunked and stored in ChromaDB as persistent vector collections
4. **Audit** — For each requirement, the top 25 most relevant chunks are retrieved from both the tender collection (for context) and the bidder collection (for evidence), and passed to Mistral 7B which returns a structured verdict
5. **Report** — Verdicts are written into a copy of the checklist Excel, one sheet per bidder, saved as `Report_All.xlsx`

### Agent Pipeline (`test_agent.py`)

An experimental alternative using LlamaIndex's `ReActAgent`. Mistral is given two tools — `tender_context` and `bidder_docs` — and autonomously decides how to query them before forming a verdict. Outputs to `Report_Agent.xlsx`.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Mistral 7B via [Ollama](https://ollama.com) |
| Embeddings | Ollama (`nomic-embed-text`) / ChromaDB default |
| Vector Store | [ChromaDB](https://www.trychroma.com) (main) / LlamaIndex (agent) |
| RAG Framework | Manual (main) / [LlamaIndex](https://www.llamaindex.ai) (agent) |
| PDF Conversion | [Docling](https://github.com/DS4SD/docling) |
| Excel I/O | openpyxl |
| Package Manager | [uv](https://github.com/astral-sh/uv) |

---

## Setup

### Prerequisites
- [Ollama](https://ollama.com) installed and running
- Required models pulled:
  ```
  ollama pull mistral
  ollama pull nomic-embed-text
  ```
- [uv](https://github.com/astral-sh/uv) installed

### Install Dependencies
```bash
cd backend
uv sync
```

### Prepare Data
1. Place raw PDFs in `data/raw/01/` with subfolders matching the structure above
2. Run the PDF converter:
   ```bash
   uv run python convert_to_md.py
   ```

### Run the Audit
```bash
uv run python main.py
```

For the agent-based pipeline:
```bash
uv run python test_agent.py
```

---

## Output

Reports are saved to `data/reports/`:
- `Report_All.xlsx` — one sheet per bidder, from `main.py`
- `Report_Agent.xlsx` — one sheet per bidder, from `test_agent.py`

Each sheet is a copy of the BQC checklist with two columns filled in per requirement:
- **Status** — Pass / Fail / N/A / Error
- **Evidence** — Mistral's reasoning and an exact quote from the bidder's documents
