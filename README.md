# RAG Document QA System

A Retrieval-Augmented Generation (RAG) based document question-answering system that allows users to upload PDF documents, ask natural-language questions, and receive answers grounded in the retrieved document content.

The system combines semantic retrieval, Cross-Encoder reranking, and Gemini-based answer generation to improve the relevance and reliability of document-grounded answers.

## Overview

The system implements a complete RAG pipeline:

1. Upload PDF documents.
2. Extract text using PyMuPDF.
3. Split extracted content into structure-aware chunks.
4. Generate vector embeddings using Sentence Transformers.
5. Store embeddings in a FAISS vector index.
6. Retrieve the top 20 candidate chunks using dense semantic search.
7. Rerank the candidates using a Cross-Encoder.
8. Select the top 3 relevant chunks.
9. Provide the retrieved context to Gemini.
10. Generate a grounded answer.
11. Display the source document and page numbers.

## Architecture

```text
                 PDF Documents
                       |
                       v
                Text Extraction
                   (PyMuPDF)
                       |
                       v
          Structure-Aware Chunking
                       |
                       v
             Sentence Transformer
              (all-MiniLM-L6-v2)
                       |
                       v
                Vector Embeddings
                       |
                       v
                     FAISS
                       |
                       |
                 User Question
                       |
                       v
                Query Embedding
                       |
                       v
            Dense Retrieval - Top 20
                       |
                       v
             Cross-Encoder Reranking
                       |
                       v
                 Top 3 Chunks
                       |
                       v
                    Gemini
                       |
                       v
                Grounded Answer
                       |
                       v
           Source Document + Pages
```

## Key Features

- Multiple PDF document upload and processing in a single session
- PDF text extraction using PyMuPDF
- Structure-aware chunking
- Semantic vector embeddings
- FAISS similarity search
- Dense retrieval
- Cross-Encoder reranking
- Grounded Gemini answer generation
- Source document and page reporting
- Retrieval evaluation
- Streamlit web interface
- Local vector-store persistence
- Deployed using Streamlit Community Cloud

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| Cross-Encoder | Retrieval reranking |
| Gemini | Grounded answer generation |
| python-dotenv | Environment variable management |
| Streamlit | Web application interface |

## Project Structure

```text
RAG-Document-QA/
|
+-- app/
|   +-- __init__.py
|   +-- chunker.py
|   +-- embeddings.py
|   +-- ingestion.py
|   +-- llm.py
|   +-- main.py
|   +-- pdf_processor.py
|   +-- rag.py
|   +-- vector_store.py
|
+-- data/
|   +-- uploads/
|   |   +-- PDF documents
|   |
|   +-- vectorstore/
|       +-- chunks.json
|       +-- index.faiss
|
+-- experiments/
|   +-- compare_retrieval.py
|   +-- inspect_failures.py
|   +-- test_bm25.py
|   +-- test_hybrid.py
|   +-- test_ingestion.py
|   +-- test_llm.py
|   +-- test_query_expansion.py
|   +-- test_rag.py
|   +-- test_retrieval.py
|
+-- tests/
|   +-- __init__.py
|   +-- rebuild_vectorstore.py
|   +-- test_chunking.py
|   +-- test_embeddings.py
|   +-- test_multidoc_retrieval.py
|   +-- test_new_questions.py
|   +-- test_pdf.py
|   +-- test_reranker.py
|   +-- test_vector_store.py
|
+-- .gitignore
+-- README.md
+-- requirements.txt
+-- streamlit_app.py
```

## Installation

Clone the repository:

```powershell
git clone https://github.com/saihaneesh4-dotcom/RAG-Document-QA.git
cd RAG-Document-QA
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
```

The `.env` file contains the API key and is excluded from Git using `.gitignore`.

## Running the Application

The main user interface is the Streamlit application.

Run:

```powershell
streamlit run streamlit_app.py
```

The application allows users to:

- Upload one or more PDF documents.
- Process the uploaded documents.
- Ask questions about their contents.
- View generated answers.
- View the source document and page numbers used for the answer.

A command-line interface is also available for querying the system:

```powershell
python -m app.main
```

## Document Processing

The ingestion pipeline processes multiple PDF documents together.

```text
PDF Documents
      |
      v
Text Extraction
      |
      v
Structure-Aware Chunking
      |
      v
Embedding Generation
      |
      v
FAISS Index Creation
      |
      v
Chunk Metadata Storage
```

Each chunk stores metadata including:

- Chunk ID
- Source document
- Starting page
- Ending page
- Chunk text

The vector store consists of:

```text
data/vectorstore/
+-- index.faiss
+-- chunks.json
```

The FAISS index and chunk metadata are generated together so that vector positions remain aligned with their corresponding chunks.

## Retrieval Pipeline

The retrieval system uses two stages.

### 1. Dense Retrieval

The user's question is converted into an embedding using `all-MiniLM-L6-v2`.

FAISS performs similarity search using an `IndexFlatIP` index and retrieves the top 20 candidate chunks.

The embeddings are normalized before indexing, which allows inner-product similarity to correspond to cosine similarity.

### 2. Cross-Encoder Reranking

The 20 retrieved candidates are passed to `cross-encoder/ms-marco-MiniLM-L-6-v2`.

The Cross-Encoder evaluates the question and each candidate passage together and produces a relevance score for reranking. FAISS is not involved in this step, and the Cross-Encoder does not perform retrieval on its own; it only reorders the candidates that dense retrieval already returned.

The candidates are then reordered according to this relevance score, and the top 3 chunks are passed to Gemini as the document context.

The source score shown in the application UI reflects the dense FAISS retrieval result. Cross-Encoder scores are used internally for reranking and are not displayed as the source score.

## Answer Generation

Gemini (`gemini-3.5-flash-lite`) generates the final response using only the retrieved document context.

The prompt instructs the model to:

- Answer using the provided context.
- Avoid inventing information.
- State when the answer cannot be found in the retrieved context.

This acts as a grounding and guardrail mechanism that helps keep responses grounded in the retrieved document context. It does not guarantee that every answer will be fully accurate.

## Evaluation

The retrieval system was evaluated using a set of unseen questions covering different sections of the source documents.

### Dense Retrieval

| Metric | Result |
|---|---|
| Recall@3 | 64.29% |
| Recall@5 | 78.57% |
| Recall@10 | 100% |
| MRR | 0.5716 |

### Dense Retrieval + Cross-Encoder

| Metric | Result |
|---|---|
| Recall@3 | 85.71% |
| Recall@5 | 92.86% |
| Recall@10 | 92.86% |
| MRR | 0.7128 |

Cross-Encoder reranking substantially improved the ranking of relevant chunks, particularly in cases where dense retrieval initially placed relevant information lower in the candidate list.

## Design Decisions

### Structure-Aware Chunking

Some sections in the source documents continue onto the following page. If the text extracted from the next page starts with `Cont.`, that page is combined with the current page before chunking. This keeps content belonging to the same section together instead of splitting it across separate chunks. This is a simple, rule-based heuristic rather than a semantic segmentation method.

### Why FAISS with IndexFlatIP?

The document collection used for this project is relatively small, so an exact `IndexFlatIP` index is sufficient and avoids the added complexity of approximate nearest-neighbor indexing. Because the embeddings are normalized, inner-product similarity computed by FAISS is equivalent to cosine similarity for the retrieval step. This is not intended as a large-scale production vector database.

### Why Reranking?

Dense retrieval provides an efficient way to find a candidate set, but semantic similarity alone does not always produce the ideal ranking. The Cross-Encoder evaluates the question and each candidate passage jointly, allowing more detailed relevance scoring before the final context is sent to Gemini.

## Current Limitations

- The PDF processor extracts text but does not perform OCR.
- Scanned or image-only PDFs may produce little or no searchable text.
- Image and table understanding is not currently implemented.
- The system relies on Gemini for answer generation.
- Retrieval quality depends on the quality of the extracted text and chunking.

## Future Improvements

Possible future improvements include:

- OCR support for scanned PDFs
- Image and table understanding
- Document metadata filtering
- Improved chunking strategies
- Conversation history
- Additional retrieval strategies
- Larger-scale vector indexing
- More extensive evaluation and observability

## Example

**Question**

What are the five interrupt sources in the 8051?

**Retrieved Information**

The system retrieves the relevant section covering the 8051 interrupt sources.

**Answer**

The five interrupt sources are:

1. INT0
2. INT1
3. TF0
4. TF1
5. TI/RI

The generated answer is accompanied by the relevant source document and page numbers.

## Deployment

The Streamlit application is deployed using Streamlit Community Cloud.

The application can be accessed at:

<https://rag-document-app-n7qdu6w5dxhgx6pgvcq89g.streamlit.app>

The Gemini API key is configured through the deployment's secrets configuration rather than being stored in the GitHub repository.

## Security

API keys and local environment files are excluded from version control.

The following files and directories are ignored:

```text
.env
.venv/
data/uploads/
data/vectorstore/
__pycache__/
*.pyc
```

## License

This project was developed as part of an internship project.