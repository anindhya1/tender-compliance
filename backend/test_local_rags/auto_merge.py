"""
Auto-Merging Retrieval with LlamaIndex and TruLens Evaluation
=============================================================
Implements hierarchical node parsing and auto-merging retrieval
over a PDF document, with optional TruLens evaluation.
"""

import os
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PDF_PATH = "Knowledge transfer.pdf"   # update if needed
EVAL_QUESTIONS_PATH = "generated_questions.text"      # update if needed
EMBED_MODEL = "local:BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1

# ── Imports ───────────────────────────────────────────────────────────────────

import openai
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    SimpleDirectoryReader,
)
from llama_index.core.settings import Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI


# ── Helper functions ──────────────────────────────────────────────────────────

def load_documents(pdf_path: str) -> Document:
    """Load a PDF and merge all pages into a single Document."""
    raw_docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    print(f"Loaded {len(raw_docs)} pages from '{pdf_path}'.")
    return Document(text="\n\n".join(doc.text for doc in raw_docs))


def build_automerging_index(
    documents,
    llm,
    embed_model: str = EMBED_MODEL,
    save_dir: str = "merging_index",
    chunk_sizes: list = None,
) -> VectorStoreIndex:
    """
    Parse documents into a hierarchy of nodes and build (or reload)
    a VectorStoreIndex over the leaf nodes.

    chunk_sizes controls the hierarchy levels, e.g.:
      [2048, 512]       → two-layer  (parent 2048, leaf 512)
      [2048, 512, 128]  → three-layer (parent 2048, mid 512, leaf 128)
    """
    chunk_sizes = chunk_sizes or [2048, 512, 128]

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(
        documents if isinstance(documents, list) else [documents]
    )
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"Total nodes: {len(nodes)} | Leaf nodes: {len(leaf_nodes)}")

    if not os.path.exists(save_dir):
        print(f"Building index → '{save_dir}' ...")
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        index.storage_context.persist(persist_dir=save_dir)
        print("Index saved.")
    else:
        print(f"Loading existing index from '{save_dir}' ...")
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            embed_model=embed_model,
        )

    return index


def get_automerging_query_engine(
    index: VectorStoreIndex,
    similarity_top_k: int = 12,
    rerank_top_n: int = 6,
) -> RetrieverQueryEngine:
    """
    Wrap the index with AutoMergingRetriever and a SentenceTransformer
    re-ranker, then return a RetrieverQueryEngine.
    """
    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever,
        index.storage_context,
        verbose=True,
    )
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=RERANKER_MODEL)
    return RetrieverQueryEngine.from_args(retriever, node_postprocessors=[rerank])


# ── TruLens helpers ───────────────────────────────────────────────────────────

def get_trulens_recorder(query_engine, app_id: str):
    """Build a TruLens recorder with Answer Relevance, Context Relevance, and Groundedness."""
    import nest_asyncio
    nest_asyncio.apply()

    from trulens_eval import Feedback, TruLlama, OpenAI as TruOpenAI
    from trulens_eval.feedback import Groundedness
    import numpy as np

    openai_provider = TruOpenAI()
    grounded = Groundedness(groundedness_provider=openai_provider)

    qa_relevance = (
        Feedback(openai_provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )
    qs_relevance = (
        Feedback(openai_provider.relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )
    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    return TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=[qa_relevance, qs_relevance, groundedness],
    )


def run_evals(eval_questions: list, tru_recorder, query_engine):
    """Run every evaluation question through the recorder."""
    for question in eval_questions:
        with tru_recorder as recording:
            query_engine.query(question)


def load_eval_questions(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Export it before running:\n  export OPENAI_API_KEY=sk-..."
        )

    llm = OpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    # LlamaIndex now uses Settings instead of ServiceContext.
    # Set the default LLM once so query synthesis uses the intended model.
    Settings.llm = llm
    document = load_documents(PDF_PATH)

    # ── Two-layer index ───────────────────────────────────────────────────────
    print("\n=== Building two-layer index (2048 / 512) ===")
    index_0 = build_automerging_index(
        [document], llm=llm, save_dir="merging_index_0", chunk_sizes=[2048, 512]
    )
    engine_0 = get_automerging_query_engine(index_0, similarity_top_k=12, rerank_top_n=6)

    # Quick demo query
    print("\n--- Demo query (two-layer) ---")
    resp = engine_0.query("What is the importance of tool_2 in the app?")
    print("Response:", resp)

    # ── Three-layer index ─────────────────────────────────────────────────────
    print("\n=== Building three-layer index (2048 / 512 / 128) ===")
    index_1 = build_automerging_index(
        [document], llm=llm, save_dir="merging_index_1", chunk_sizes=[2048, 512, 128]
    )
    engine_1 = get_automerging_query_engine(index_1, similarity_top_k=12, rerank_top_n=6)

    print("\n--- Demo query (three-layer) ---")
    resp = engine_1.query("What is the importance of tool_3 in the app?")
    print("Response:", resp)

    # ── Optional TruLens evaluation ───────────────────────────────────────────
    if os.path.exists(EVAL_QUESTIONS_PATH):
        print("\n=== Running TruLens evaluation ===")
        try:
            from trulens_eval import Tru
            Tru().reset_database()

            eval_questions = load_eval_questions(EVAL_QUESTIONS_PATH)
            print(f"Loaded {len(eval_questions)} evaluation questions.")

            print("\n--- Evaluating two-layer (app_0) ---")
            recorder_0 = get_trulens_recorder(engine_0, app_id="app_0")
            run_evals(eval_questions, recorder_0, engine_0)

            print("\n--- Evaluating three-layer (app_1) ---")
            recorder_1 = get_trulens_recorder(engine_1, app_id="app_1")
            run_evals(eval_questions, recorder_1, engine_1)

            print("\n=== Leaderboard ===")
            print(Tru().get_leaderboard(app_ids=[]))

            print("\nLaunching TruLens dashboard (Ctrl-C to stop)...")
            Tru().run_dashboard()

        except ImportError:
            print("trulens_eval not installed — skipping evaluation.")
    else:
        print(f"\nNo eval questions file found at '{EVAL_QUESTIONS_PATH}'. Skipping TruLens evaluation.")


if __name__ == "__main__":
    main()