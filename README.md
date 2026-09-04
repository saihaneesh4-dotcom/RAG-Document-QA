````markdown
# RAG Document QA System

A Retrieval-Augmented Generation (RAG) based document question-answering system that allows users to ask questions about PDF documents and receive answers grounded in the retrieved document content.

## Overview

This project implements a complete RAG pipeline for answering questions from PDF documents.

Instead of sending the entire document to a language model, the system:

1. Extracts text from the PDF.
2. Splits the document into meaningful chunks.
3. Converts the chunks into vector embeddings.
4. Stores the embeddings in a FAISS vector index.
5. Retrieves relevant candidate chunks for a user question.
6. Reranks the retrieved chunks using a Cross-Encoder.
7. Sends the best matching chunks to Gemini.
8. Generates an answer using the retrieved document context.
9. Displays the source pages used for the answer.

## Architecture

```text
                 PDF Document
                      │
                      ▼
              Text Extraction
                      │
                      ▼
             Structure-Aware Chunking
                      │
                      ▼
              Sentence Transformer
              (all-MiniLM-L6-v2)
                      │
                      ▼
                Vector Embeddings
                      │
                      ▼
                  FAISS Index
                      │
                      │
                User Question
                      │
                      ▼
                Query Embedding
                      │
                      ▼
              Dense Retrieval (Top 20)
                      │
                      ▼
             Cross-Encoder Reranking
                      │
                      ▼
                  Top 3 Chunks
                      │
                      ▼
                    Gemini
                      │
                      ▼
                Grounded Answer
                      │
                      ▼
                  Source Pages
````

## Key Features

* PDF text extraction using PyMuPDF
* Structure-aware document chunking
* Semantic vector embeddings
* FAISS-based similarity search
* Cross-Encoder reranking
* Grounded answer generation using Gemini
* Source page reporting
* Retrieval evaluation
* Local vector storage

## Technologies Used

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Core programming language       |
| PyMuPDF               | PDF text extraction             |
| Sentence Transformers | Text embeddings                 |
| all-MiniLM-L6-v2      | Embedding model                 |
| FAISS                 | Vector similarity search        |
| Cross-Encoder         | Retrieval reranking             |
| Gemini                | Answer generation               |
| python-dotenv         | Environment variable management |

## Project Structure

```text
RAG-Document-QA/
│
├── app/
│   ├── __init__.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── main.py
│   ├── pdf_processor.py
│   ├── rag.py
│   └── vector_store.py
│
├── data/
│   ├── uploads/
│   │   └── PDF documents
│   │
│   └── vectorstore/
│       ├── chunks.json
│       └── index.faiss
│
├── experiments/
│   ├── compare_retrieval.py
│   ├── inspect_failures.py
│   ├── test_bm25.py
│   ├── test_hybrid.py
│   ├── test_ingestion.py
│   ├── test_llm.py
│   ├── test_query_expansion.py
│   ├── test_rag.py
│   └── test_retrieval.py
│
├── tests/
│   ├── __init__.py
│   ├── rebuild_vectorstore.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_new_questions.py
│   ├── test_pdf.py
│   ├── test_reranker.py
│   └── test_vector_store.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```powershell
git clone <repository-url>
cd RAG-Document-QA
```

Create and activate the virtual environment:

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

The `.env` file contains the API key and should not be committed to Git.

## Running the Application

The current vector store can be queried using:

```powershell
python -m app.main
```

The system will prompt:

```text
Enter your question:
```

For example:

```text
What are the five interrupt sources in the 8051?
```

The system returns the generated answer along with the source pages used to answer the question.

## Processing Documents

The document ingestion pipeline follows these steps:

```text
PDF
 ↓
Text Extraction
 ↓
Structure-Aware Chunking
 ↓
Embedding Generation
 ↓
FAISS Index Creation
 ↓
Chunk Metadata Storage
```

The generated vector store is stored in:

```text
data/vectorstore/
```

The vector store is rebuilt for the currently processed document so that the FAISS index and chunk metadata remain aligned.

## Retrieval Pipeline

The retrieval process consists of two stages.

### 1. Dense Retrieval

The user's question is converted into an embedding using:

`all-MiniLM-L6-v2`

FAISS then retrieves the top 20 candidate chunks based on vector similarity.

### 2. Cross-Encoder Reranking

The 20 retrieved candidates are passed to a Cross-Encoder:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The Cross-Encoder evaluates the question and each candidate chunk together and reorders the candidates according to their relevance.

The top 3 chunks are then passed to Gemini.

## Evaluation

The retrieval system was evaluated using an unseen set of questions covering different sections of the document.

### Dense Retrieval

| Metric    | Result |
| --------- | -----: |
| Recall@3  | 64.29% |
| Recall@5  | 78.57% |
| Recall@10 |   100% |
| MRR       | 0.5716 |

### Dense Retrieval + Cross-Encoder

| Metric    | Result |
| --------- | -----: |
| Recall@3  | 85.71% |
| Recall@5  | 92.86% |
| Recall@10 | 92.86% |
| MRR       | 0.7128 |

The results show that Cross-Encoder reranking substantially improved the ordering of relevant chunks, particularly for questions where dense retrieval initially placed the relevant information lower in the candidate list.

## Design Decisions

### Structure-Aware Chunking

Some sections in the source document continue onto the following page. When a page begins with `Cont.`, it is combined with the previous page before chunking.

This helps keep information belonging to the same section together.

### Why FAISS?

The current document collection is relatively small, so an exact FAISS inner-product index is sufficient.

The embeddings are normalized, making inner product equivalent to cosine similarity for the retrieval step.

### Why Reranking?

Dense retrieval is useful for quickly finding a candidate set, but the initial ordering is not always optimal.

A Cross-Encoder evaluates the question and candidate passage together, allowing more detailed relevance scoring.

## Current Limitations

* The current PDF processor extracts text but does not perform OCR.
* Scanned or image-only PDFs may therefore produce little or no searchable text.
* The current vector store represents one processed document at a time.
* The system currently uses a command-line interface.
* Gemini is required for answer generation.

## Future Improvements

Possible future improvements include:

* OCR support for scanned PDFs
* Image and table understanding
* Multi-document vector stores
* Document metadata and filtering
* Improved chunking strategies
* Conversation history
* Web-based user interface
* More extensive retrieval evaluation

## Example

### Question

```text
What are the five interrupt sources in the 8051?
```

### Answer

The system retrieves the relevant section from the document and generates an answer containing:

```text
INT0'
INT1'
TF0
TF1
TI/RI
```

The answer is accompanied by the relevant source pages.

## License

This project was developed as part of an internship project.