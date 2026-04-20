---
title: Villi Bharatam RAG
emoji: 📖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: mit
short_description: Ask anything about the 15th-century Tamil Mahabharata
---

# வில்லி பாரதம் — Villi Bharatam RAG

Ask questions in **Tamil or English** about *Villi Bharatam* — a 15th-century Tamil retelling of the Mahabharata by Villiputhur Alwar.

## Corpus
| Parva | Pages |
|---|---|
| ஆதி பருவம் — Origins | 577 |
| சபா பருவம் — The Court | 288 |
| ஆரணிய பருவம் — Forest | 292 |
| விராட பருவம் — Virata | 161 |
| உத்தியோக பருவம் — Preparations | 243 |
| வீட்டும பருவம் — Bhishma | 194 |
| கன்ன பருவம் — Karna | 160 |
| சல்லிய-சௌப்திக பருவம் — Epilogue | 262 |
| **Total** | **2,177 pages / 10,370 chunks** |

## Stack
- **LangGraph** — stateful retrieval pipeline with retry loop
- **LangChain EnsembleRetriever** — Chroma (semantic) + BM25 (keyword) merged via RRF
- **Cohere rerank-multilingual-v3.0** — cross-encoder reranking, Tamil native
- **OpenRouter** — embeddings + LLM (gpt-4o-mini)

## Technical articles
- [Building RAG on a 2177-Page Tamil Epic](https://nishokvg.github.io/posts/building-rag-on-tamil-epic/)
- [From Manual RAG to LangGraph + Cohere + RAGAS](https://nishokvg.github.io/posts/langgraph-rag-cohere-ragas/)
