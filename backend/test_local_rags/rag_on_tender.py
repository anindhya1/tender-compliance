"""
Sentence Window Retrieval with LlamaIndex and TruLens Evaluation
================================================================
Implements sentence-level node parsing with surrounding context windows
for improved RAG retrieval, with optional TruLens evaluation.
"""

import os
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

PDF_PATH = "Knowledge transfer.pdf"   # update if needed
EVAL_QUESTIONS_PATH = "generated_questions.text"      # update if needed
EMBED_MODEL = "local:BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.1

# ── Imports ───────────────────────────────────────────────────────────────────

# import numpy as np  # used by TruLens only

from docling.document_converter import DocumentConverter

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings


# ── Helper functions ──────────────────────────────────────────────────────────

def load_documents(pdf_path: str) -> Document:
    """Load a PDF via Docling and return as a single Document."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    text = result.document.export_to_markdown()
    print(f"Loaded '{pdf_path}' via Docling.")
    return Document(text=text)


def build_sentence_window_index(
    documents,
    llm,
    embed_model: str = EMBED_MODEL,
    sentence_window_size: int = 3,
    save_dir: str = "sentence_index",
) -> VectorStoreIndex:
    """
    Parse documents into sentence-level nodes, each carrying a metadata
    'window' of surrounding sentences, then build (or reload) a
    VectorStoreIndex over those nodes.

    sentence_window_size controls how many sentences on either side of the
    retrieved sentence are included as context when answering:
      1 → one neighbour on each side  (narrow context)
      3 → three neighbours on each side (broader context)
    """
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    if not os.path.exists(save_dir):
        print(f"Building index → '{save_dir}' (window_size={sentence_window_size}) ...")
        index = VectorStoreIndex.from_documents(
            documents if isinstance(documents, list) else [documents],
        )
        index.storage_context.persist(persist_dir=save_dir)
        print("Index saved.")
    else:
        print(f"Loading existing index from '{save_dir}' ...")
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
        )

    return index


def get_sentence_window_query_engine(
    sentence_index: VectorStoreIndex,
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
):
    """
    Build a query engine that:
    1. Retrieves the top-K most similar sentence nodes.
    2. Replaces each node's text with its surrounding window (MetadataReplacement).
    3. Re-ranks the expanded nodes and returns the top-N to the LLM.
    """
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=RERANKER_MODEL)

    return sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
    )


# ── TruLens helpers ───────────────────────────────────────────────────────────

# def get_trulens_recorder(query_engine, app_id: str):
#     """Build a TruLens recorder with Answer Relevance, Context Relevance, and Groundedness."""
#     import nest_asyncio
#     nest_asyncio.apply()
#
#     from trulens_eval import Feedback, TruLlama, OpenAI as TruOpenAI
#     from trulens_eval.feedback import Groundedness
#
#     openai_provider = TruOpenAI()
#     grounded = Groundedness(groundedness_provider=openai_provider)
#
#     qa_relevance = (
#         Feedback(openai_provider.relevance_with_cot_reasons, name="Answer Relevance")
#         .on_input_output()
#     )
#     qs_relevance = (
#         Feedback(openai_provider.relevance_with_cot_reasons, name="Context Relevance")
#         .on_input()
#         .on(TruLlama.select_source_nodes().node.text)
#         .aggregate(np.mean)
#     )
#     groundedness = (
#         Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
#         .on(TruLlama.select_source_nodes().node.text)
#         .on_output()
#         .aggregate(grounded.grounded_statements_aggregator)
#     )
#
#     from trulens_eval import TruLlama
#     return TruLlama(
#         query_engine,
#         app_id=app_id,
#         feedbacks=[qa_relevance, qs_relevance, groundedness],
#     )


# def run_evals(eval_questions: list, tru_recorder, query_engine):
#     """Run every evaluation question through the recorder."""
#     for question in eval_questions:
#         with tru_recorder as recording:
#             query_engine.query(question)


# def load_eval_questions(path: str) -> list:
#     with open(path, "r") as f:
#         return [line.strip() for line in f if line.strip()]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    llm = Ollama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, request_timeout=300.0)
    document = load_documents(PDF_PATH)

    # ── Window size = 4 ───────────────────────────────────────────────────────
    print("\n=== Building sentence window index (window_size=4) ===")
    index_1 = build_sentence_window_index(
        [document], llm=llm, sentence_window_size=4, save_dir="sentence_index_4"
    )
    engine_1 = get_sentence_window_query_engine(index_1, similarity_top_k=20, rerank_top_n=5)

    # ── Window size = 6 ───────────────────────────────────────────────────────
    print("\n=== Building sentence window index (window_size=6) ===")
    index_3 = build_sentence_window_index(
        [document], llm=llm, sentence_window_size=6, save_dir="sentence_index_6"
    )
    engine_3 = get_sentence_window_query_engine(index_3, similarity_top_k=20, rerank_top_n=5)

    # ── Interactive query loop ────────────────────────────────────────────────
    print("\n=== Interactive Query Mode ===")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        print("\n--- Answer (window_size=4) ---")
        resp_1 = engine_1.query(question)
        print(str(resp_1))

        print("\n--- Answer (window_size=6) ---")
        resp_3 = engine_3.query(question)
        print(str(resp_3))
        print()

    # ── Optional TruLens evaluation ───────────────────────────────────────────
    # if os.path.exists(EVAL_QUESTIONS_PATH):
    #     print("\n=== Running TruLens evaluation ===")
    #     try:
    #         from trulens_eval import Tru
    #         Tru().reset_database()
    #
    #         eval_questions = load_eval_questions(EVAL_QUESTIONS_PATH)
    #         print(f"Loaded {len(eval_questions)} evaluation questions.")
    #
    #         print("\n--- Evaluating window_size=1 (sentence window engine 1) ---")
    #         recorder_1 = get_trulens_recorder(engine_1, app_id="sentence window engine 1")
    #         run_evals(eval_questions, recorder_1, engine_1)
    #
    #         print("\n--- Evaluating window_size=3 (sentence window engine 3) ---")
    #         recorder_3 = get_trulens_recorder(engine_3, app_id="sentence window engine 3")
    #         run_evals(eval_questions, recorder_3, engine_3)
    #
    #         print("\n=== Leaderboard ===")
    #         print(Tru().get_leaderboard(app_ids=[]))
    #
    #         print("\nLaunching TruLens dashboard (Ctrl-C to stop)...")
    #         Tru().run_dashboard()
    #
    #     except ImportError:
    #         print("trulens_eval not installed — skipping evaluation.")
    # else:
    #     print(f"\nNo eval questions file found at '{EVAL_QUESTIONS_PATH}'. Skipping TruLens evaluation.")


if __name__ == "__main__":
    main()