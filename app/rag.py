from app.vector_store import retrieve
from app.llm import ask_llm
from sentence_transformers import CrossEncoder


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def answer_question(question, chunks, k=3):

    # Retrieve a larger candidate pool using dense retrieval
    dense_results = retrieve(
        question,
        chunks,
        k=20
    )

    # Rerank the candidates using the Cross-Encoder
    pairs = [
        (question, result["chunk"]["text"])
        for result in dense_results
    ]

    scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(dense_results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Keep the top k results for the LLM
    results = [
        result
        for result, score in reranked_results[:k]
    ]

    # Build document context
    context_parts = []

    for result in results:

        chunk = result["chunk"]

        start_page = chunk["page"]
        end_page = chunk.get("end_page", start_page)

        if start_page == end_page:
            page_label = f"Page {start_page}"
        else:
            page_label = f"Pages {start_page}-{end_page}"

        context_parts.append(
            f"{page_label}:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Ask Gemini using only retrieved context
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the provided document context.

If the context does not contain enough information to answer the question,
say that you cannot find the answer in the provided document.

Do not make up information.

Document context:
{context}

User question:
{question}
"""

    answer = ask_llm(prompt)

    return {
        "answer": answer,
        "sources": results
    }