"""
app.py — Gradio UI for Villi Bharatam RAG (HF Spaces)
"""

import gradio as gr
from graph import load_index, ask, PARVA_NAMES

# ── Load index once at startup ────────────────────────────────────────────────

collection, bm25_obj, chunks = load_index()

# ── Parva choices ─────────────────────────────────────────────────────────────

PARVA_CHOICES = [("All Parvas", 0)] + [
    (f"{v} (Parva {k})", int(k)) for k, v in sorted(PARVA_NAMES.items(), key=lambda x: int(x[0]))
]


# ── Query handler ─────────────────────────────────────────────────────────────

def query(question: str, parva_choice: int) -> tuple[str, str, str, str]:
    if not question.strip():
        return "Please enter a question.", "", "", ""

    parva_filter = parva_choice if parva_choice else None

    try:
        answer, reranked_docs = ask(question, collection, bm25_obj, chunks, parva_filter)
    except Exception as e:
        return f"Error: {e}", "", "", ""

    # Answer text
    answer_text = answer.answer

    # Confidence badge
    conf_map = {
        "high":      "High confidence",
        "medium":    "Medium confidence",
        "low":       "Low confidence",
        "not_found": "Not found in corpus",
    }
    confidence_text = conf_map.get(answer.confidence, answer.confidence)

    # Citations
    if answer.evidence:
        lines = []
        for i, ev in enumerate(answer.evidence, 1):
            lines.append(
                f"**[{i}]** {ev.parva} · பக்கம் {ev.page}\n"
                f"> {ev.quote}"
            )
        citations_text = "\n\n".join(lines)
    else:
        citations_text = "_No citations._"

    # Top retrieved passages
    if reranked_docs:
        passages = []
        for doc in reranked_docs:
            m = doc.metadata
            score = m.get("cohere_score", "—")
            passages.append(
                f"**{m.get('parva_name', '?')}** · பக்கம் {m.get('page_start', '?')}–{m.get('page_end', '?')} "
                f"(score: {score})\n\n{doc.page_content[:400]}{'…' if len(doc.page_content) > 400 else ''}"
            )
        passages_text = "\n\n---\n\n".join(passages)
    else:
        passages_text = "_No passages retrieved._"

    return answer_text, confidence_text, citations_text, passages_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="வில்லி பாரதம் — Villi Bharatam RAG",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="""
    .header { text-align: center; padding: 1rem 0 0.5rem; }
    .header h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
    .header p  { color: #6b7280; font-size: 0.9rem; }
    .conf-high   { color: #16a34a; font-weight: 600; }
    .conf-medium { color: #ca8a04; font-weight: 600; }
    .conf-low    { color: #dc2626; font-weight: 600; }
    """,
) as demo:

    gr.HTML("""
    <div class="header">
      <h1>📖 வில்லி பாரதம்</h1>
      <p>Ask anything about <em>Villi Bharatam</em> — a 15th-century Tamil retelling of the Mahabharata.<br>
         Questions accepted in <strong>Tamil</strong> or <strong>English</strong>.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. Who is Karna's mother? / கர்ணனின் தாய் யார்?",
                lines=3,
            )
            parva_dd = gr.Dropdown(
                label="Filter by Parva (optional)",
                choices=PARVA_CHOICES,
                value=0,
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Answer", lines=8, interactive=False)
            confidence_box = gr.Textbox(label="Confidence", interactive=False)

    with gr.Accordion("Citations", open=True):
        citations_box = gr.Markdown()

    with gr.Accordion("Retrieved passages (debug)", open=False):
        passages_box = gr.Markdown()

    submit_btn.click(
        fn=query,
        inputs=[question_box, parva_dd],
        outputs=[answer_box, confidence_box, citations_box, passages_box],
    )
    question_box.submit(
        fn=query,
        inputs=[question_box, parva_dd],
        outputs=[answer_box, confidence_box, citations_box, passages_box],
    )

    gr.Examples(
        examples=[
            ["Who is Karna and what is his role in the epic?", 0],
            ["What happens during the dice game?", 2],
            ["கர்ணனின் தாய் யார்?", 8],
            ["பாண்டவர்கள் காட்டில் எவ்வளவு காலம் இருந்தார்கள்?", 3],
            ["What is the significance of Bhishma's vow?", 0],
        ],
        inputs=[question_box, parva_dd],
    )

    gr.HTML("""
    <div style="text-align:center; color:#9ca3af; font-size:0.75rem; margin-top:1rem;">
      Corpus: 2,177 pages · 10,370 chunks · 8 Parvas<br>
      Stack: LangGraph · LangChain · Chroma + BM25 · Cohere rerank-multilingual-v3.0 · GPT-4o-mini via OpenRouter
    </div>
    """)


if __name__ == "__main__":
    demo.launch()
