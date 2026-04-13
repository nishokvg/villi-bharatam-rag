"""
graph.py — Villi Bharatam LangGraph RAG (HF Spaces version)
============================================================
Adapted from langraph-bharatham-rag/graph.py for Hugging Face Spaces.
Changes vs local version:
  - No dotenv (HF Spaces injects secrets as env vars directly)
  - Paths relative to app root (not ../prod-bharatham-rag/)
  - Cache disabled (HF free tier has ephemeral disk — don't persist cache)
"""

from __future__ import annotations

import os, json, time, pickle
from typing import Literal, TypedDict

import numpy as np
import chromadb
import cohere
from langdetect import detect, LangDetectException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma as LangChroma
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR      = os.path.join(BASE_DIR, "chroma-db")
BM25_PATH       = os.path.join(BASE_DIR, "bm25.pkl")
COLLECTION_NAME = "bharatham_prod"

TOP_K_RETRIEVAL = 10
TOP_K_RERANK    = 4
MAX_GRAPH_LOOPS = 2

OPENROUTER_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
COHERE_KEY      = os.environ.get("COHERE_API_KEY", "")
OR_EMBED_MODEL  = "openai/text-embedding-3-small"
OR_CHAT_MODEL   = "openai/gpt-4o-mini"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

PARVA_NAMES = {
    "1": "ஆதி பருவம்",
    "2": "சபா பருவம்",
    "3": "ஆரணிய பருவம்",
    "4": "விராட பருவம்",
    "5": "உத்தியோக பருவம்",
    "6": "வீட்டும பருவம்",
    "8": "கன்ன பருவம்",
    "9": "சல்லிய-சௌப்திக பருவம்",
}

SYSTEM_PROMPT = """You are a scholarly assistant for வில்லி பாரதம் (Villi Bharatam),
a 15th-century Tamil retelling of the Mahabharata.

Rules you MUST follow:
1. Answer using ONLY information from the provided passages. No outside knowledge.
2. For every fact in your answer, include a citation with:
   - quote: copy the EXACT sentence or phrase from the passage (not a paraphrase)
   - parva: the parva name shown in the passage header
   - page: the page number shown in the passage header
3. If the answer is not in the passages, say so clearly. Do not guess or invent.
4. Respond in the same language as the question (Tamil question → Tamil answer).
"""


# ── Schemas ───────────────────────────────────────────────────────────────────

class Evidence(BaseModel):
    quote: str
    parva: str
    page:  int

class BharatamAnswer(BaseModel):
    answer:     str
    evidence:   list[Evidence]
    confidence: Literal["high", "medium", "low", "not_found"]


# ── LangGraph State ───────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question:      str
    parva_filter:  int | None
    is_tamil:      bool
    documents:     list[Document]
    reranked_docs: list[Document]
    context:       str
    answer:        BharatamAnswer | None
    confidence:    str
    loop_count:    int
    error:         str | None


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index():
    with open(BM25_PATH, "rb") as f:
        saved = pickle.load(f)
    bm25_obj = saved["bm25"]
    chunks   = saved["chunks"]
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    return collection, bm25_obj, chunks


# ── Retrievers ────────────────────────────────────────────────────────────────

def build_retrievers(collection, bm25_obj, chunks, parva_filter: int | None):
    embed_fn = OpenAIEmbeddings(
        model           = OR_EMBED_MODEL,
        openai_api_key  = OPENROUTER_KEY,
        openai_api_base = OPENROUTER_BASE,
    )
    search_kwargs: dict = {"k": TOP_K_RETRIEVAL}
    if parva_filter:
        search_kwargs["filter"] = {"parva_num": str(parva_filter).zfill(2)}

    vectorstore = LangChroma(
        client             = chromadb.PersistentClient(path=CHROMA_DIR),
        collection_name    = COLLECTION_NAME,
        embedding_function = embed_fn,
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    docs = []
    for c in chunks:
        if parva_filter and c["parva_num"] != str(parva_filter).zfill(2):
            continue
        docs.append(Document(
            page_content = c["text"],
            metadata     = {
                "chunk_id":   c["chunk_id"],
                "parva_name": c["parva_name"],
                "parva_num":  c["parva_num"],
                "page_start": c["page_start"],
                "page_end":   c["page_end"],
            }
        ))
    bm25_retriever = BM25Retriever.from_documents(docs, k=TOP_K_RETRIEVAL)
    ensemble = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble, chroma_retriever


# ── Cohere rerank ─────────────────────────────────────────────────────────────

def cohere_rerank(question: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return docs
    co = cohere.Client(COHERE_KEY)
    response = co.rerank(
        model     = "rerank-multilingual-v3.0",
        query     = question,
        documents = [d.page_content for d in docs],
        top_n     = min(TOP_K_RERANK, len(docs)),
    )
    reranked = [docs[r.index] for r in response.results]
    for i, r in enumerate(response.results):
        reranked[i].metadata["cohere_score"] = round(r.relevance_score, 4)
    return reranked


def cohere_score_to_confidence(score: float) -> str:
    if score >= 0.75: return "high"
    if score >= 0.45: return "medium"
    if score >= 0.15: return "low"
    return "not_found"


# ── Graph nodes ───────────────────────────────────────────────────────────────

def node_detect_language(state: RAGState) -> RAGState:
    try:
        tamil = detect(state["question"]) == "ta"
    except LangDetectException:
        tamil = False
    return {**state, "is_tamil": tamil}


def node_retrieve(state: RAGState, ensemble_ret, chroma_ret) -> RAGState:
    retriever = ensemble_ret if state["is_tamil"] else chroma_ret
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


def node_rerank(state: RAGState) -> RAGState:
    reranked = cohere_rerank(state["question"], state["documents"])
    if not reranked:
        return {**state, "reranked_docs": [], "confidence": "not_found"}
    top_score  = reranked[0].metadata.get("cohere_score", 0.0)
    confidence = cohere_score_to_confidence(top_score)
    return {**state, "reranked_docs": reranked, "confidence": confidence}


def node_build_context(state: RAGState) -> RAGState:
    parts = []
    for doc in state["reranked_docs"]:
        m = doc.metadata
        header = (
            f"[{m.get('parva_name','?')} | "
            f"பக்கம் {m.get('page_start','?')}–{m.get('page_end','?')} | "
            f"cohere={m.get('cohere_score','?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return {**state, "context": "\n\n---\n\n".join(parts)}


def node_generate(state: RAGState) -> RAGState:
    llm = ChatOpenAI(
        model           = OR_CHAT_MODEL,
        openai_api_key  = OPENROUTER_KEY,
        openai_api_base = OPENROUTER_BASE,
        temperature     = 0,
    )
    structured_llm = llm.with_structured_output(BharatamAnswer)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Question: {state['question']}\n\nPassages:\n{state['context']}"},
    ]
    try:
        result: BharatamAnswer = structured_llm.invoke(messages)
        result.confidence = state["confidence"]
    except Exception as e:
        result = BharatamAnswer(answer=f"Generation failed: {e}", evidence=[], confidence="not_found")
    return {**state, "answer": result}


def node_validate(state: RAGState) -> RAGState:
    answer = state["answer"]
    if not answer or not answer.evidence:
        return {**state, "error": None}
    retrieved_texts = [d.page_content for d in state["reranked_docs"]]

    def is_grounded(quote: str) -> bool:
        if any(quote in ct for ct in retrieved_texts):
            return True
        q_words = set(quote.split())
        if not q_words:
            return True
        for ct in retrieved_texts:
            if len(q_words & set(ct.split())) / len(q_words) >= 0.70:
                return True
        return False

    bad = [ev.quote for ev in answer.evidence if not is_grounded(ev.quote)]
    if bad:
        return {**state, "error": f"Ungrounded quote: '{bad[0][:80]}'",
                "loop_count": state.get("loop_count", 0) + 1}
    return {**state, "error": None}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_rerank(state: RAGState) -> str:
    return "skip_llm" if state["confidence"] == "not_found" else "generate"

def route_after_validate(state: RAGState) -> str:
    if state["error"] and state.get("loop_count", 0) < MAX_GRAPH_LOOPS:
        return "re_retrieve"
    return "done"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph(ensemble_ret, chroma_ret):
    graph = StateGraph(RAGState)
    graph.add_node("detect_language", node_detect_language)
    graph.add_node("retrieve",        lambda s: node_retrieve(s, ensemble_ret, chroma_ret))
    graph.add_node("rerank",          node_rerank)
    graph.add_node("build_context",   node_build_context)
    graph.add_node("generate",        node_generate)
    graph.add_node("validate",        node_validate)
    graph.add_node("skip_llm", lambda s: {**s, "answer": BharatamAnswer(
        answer="இந்த தகவல் கொடுக்கப்பட்ட பக்கங்களில் இல்லை.",
        evidence=[], confidence="not_found",
    )})
    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "retrieve")
    graph.add_edge("retrieve",        "rerank")
    graph.add_conditional_edges("rerank", route_after_rerank,
                                 {"generate": "build_context", "skip_llm": "skip_llm"})
    graph.add_edge("build_context",   "generate")
    graph.add_edge("generate",        "validate")
    graph.add_conditional_edges("validate", route_after_validate,
                                 {"re_retrieve": "retrieve", "done": END})
    graph.add_edge("skip_llm", END)
    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────

def ask(question: str, collection, bm25_obj, chunks,
        parva_filter: int | None = None) -> tuple[BharatamAnswer, list[Document]]:
    ensemble_ret, chroma_ret = build_retrievers(collection, bm25_obj, chunks, parva_filter)
    compiled = build_graph(ensemble_ret, chroma_ret)
    final = compiled.invoke({
        "question": question, "parva_filter": parva_filter,
        "is_tamil": False, "documents": [], "reranked_docs": [],
        "context": "", "answer": None, "confidence": "low",
        "loop_count": 0, "error": None,
    })
    return final["answer"], final["reranked_docs"]
