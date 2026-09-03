from app.vector_store import retrieve
from app.llm import ask_llm


def answer_question(question, chunks, k=3):
    results = retrieve(question, chunks, k=k)

    context_parts = []

    for result in results:
        chunk = result["chunk"]

        context_parts.append(
            f"Page {chunk['page']}:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

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