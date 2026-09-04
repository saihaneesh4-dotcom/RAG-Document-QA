import os
import html
import streamlit as st

import app.vector_store as vector_store
from app.ingestion import process_documents
from app.rag import answer_question


UPLOAD_DIR = "data/uploads"
INDEX_PATH = "data/vectorstore/index.faiss"
CHUNKS_PATH = "data/vectorstore/chunks.json"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(99, 80, 180, 0.16),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(30, 120, 180, 0.10),
            transparent 30%
        ),
        #0b0d12;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================
   HERO
   ========================= */

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.35rem;
}

.hero-title span {
    background: linear-gradient(
        90deg,
        #ffffff,
        #aaa4ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}


/* =========================
   STATUS
   ========================= */

.status {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 7px 13px;

    border-radius: 999px;

    background: rgba(34, 197, 94, 0.10);

    border: 1px solid rgba(34, 197, 94, 0.25);

    color: #86efac;

    font-size: 0.85rem;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #4ade80;

    box-shadow:
        0 0 10px #4ade80;
}


/* =========================
   SECTION TITLES
   ========================= */

.section-title {
    font-size: 1.15rem;
    font-weight: 700;

    margin-top: 1rem;
    margin-bottom: 0.8rem;
}


/* =========================
   DOCUMENT CARDS
   ========================= */

.document-card {
    background: rgba(255,255,255,0.035);

    border: 1px solid rgba(255,255,255,0.07);

    border-radius: 12px;

    padding: 12px 15px;

    margin: 7px 0;
}

.document-name {
    font-weight: 600;
    color: #e5e7eb;
}

.document-type {
    color: #8b93a1;

    font-size: 0.78rem;

    margin-top: 3px;
}


/* =========================
   INFORMATION CARD
   ========================= */

.info-card {
    background: rgba(255,255,255,0.035);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 16px;

    padding: 1.2rem;

    margin-top: 1.2rem;
}


/* =========================
   ANSWER
   ========================= */

.answer-card {
    background:
        linear-gradient(
            135deg,
            rgba(99, 102, 241, 0.11),
            rgba(59, 130, 246, 0.04)
        );

    border: 1px solid rgba(129, 140, 248, 0.20);

    border-radius: 18px;

    padding: 1.3rem 1.5rem;

    margin-top: 1rem;
}

.answer-label {
    color: #a5b4fc;

    font-size: 0.8rem;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 1px;

    margin-bottom: 0.8rem;
}


/* =========================
   SOURCE
   ========================= */

.source-card {
    background: rgba(255,255,255,0.025);

    border: 1px solid rgba(255,255,255,0.07);

    border-radius: 12px;

    padding: 12px 15px;

    margin: 8px 0;
}

.source-document {
    font-weight: 600;
}

.source-page {
    color: #9ca3af;

    font-size: 0.82rem;

    margin-top: 4px;
}


/* =========================
   METRICS
   ========================= */

.metric-card {
    text-align: center;

    background: rgba(255,255,255,0.035);

    border: 1px solid rgba(255,255,255,0.07);

    border-radius: 14px;

    padding: 15px 8px;
}

.metric-number {
    font-size: 1.5rem;
    font-weight: 750;
}

.metric-label {
    color: #8b93a1;

    font-size: 0.75rem;
}


/* =========================
   BUTTONS
   ========================= */

.stButton > button {
    border-radius: 10px;

    font-weight: 650;

    min-height: 42px;
}


/* =========================
   INPUT
   ========================= */

[data-testid="stTextInput"] input {
    border-radius: 12px;

    min-height: 48px;
}


/* =========================
   FILE UPLOADER
   ========================= */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.025);

    border-radius: 15px;
}


/* =========================
   SIDEBAR
   ========================= */

[data-testid="stSidebar"] {
    background: #090b10;

    border-right:
        1px solid rgba(255,255,255,0.06);
}


/* =========================
   HIDE STREAMLIT BRANDING
   ========================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([5, 1])


with header_left:

    st.markdown(
        '<div class="hero-title">📚 <span>RAG Document AI</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Ask questions. Find answers. Stay grounded in your documents.'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    st.markdown(
        """
        <div style="text-align:right; padding-top:20px;">
            <div class="status">
                <span class="status-dot"></span>
                System Ready
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### 📚 RAG Document AI"
    )

    st.markdown(
        "### 🧠 Pipeline"
    )

    st.markdown(
        """
        📄 PDF Extraction  
        ✂️ Smart Chunking  
        🔢 Embeddings  
        🔎 FAISS Retrieval  
        🎯 Cross-Encoder Reranking  
        ✨ Gemini Generation
        """
    )

    st.divider()

    st.markdown(
        "### ℹ️ About"
    )

    st.caption(
        "This application uses Retrieval-Augmented "
        "Generation to answer questions using "
        "information retrieved from uploaded PDF "
        "documents."
    )


# ============================================================
# MAIN COLUMNS
# ============================================================

left, right = st.columns(
    [1, 1.55],
    gap="large"
)


# ============================================================
# LEFT COLUMN — DOCUMENTS
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">📚 Your Documents</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )


    if uploaded_files:

        st.caption(
            f"{len(uploaded_files)} document(s) selected"
        )

        for file in uploaded_files:

            safe_name = html.escape(
                file.name
            )

            size_mb = file.size / (
                1024 * 1024
            )

            st.markdown(
                f'<div class="document-card">'
                f'<div class="document-name">📄 {safe_name}</div>'
                f'<div class="document-type">PDF • {size_mb:.1f} MB</div>'
                f'</div>',
                unsafe_allow_html=True
            )


    process_button = st.button(
        "⚡ Process Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    )


    if process_button:

        os.makedirs(
            UPLOAD_DIR,
            exist_ok=True
        )

        pdf_paths = []

        with st.spinner(
            "Extracting, chunking and indexing documents..."
        ):

            for file in uploaded_files:

                file_path = os.path.join(
                    UPLOAD_DIR,
                    file.name
                )

                with open(
                    file_path,
                    "wb"
                ) as output:

                    output.write(
                        file.getbuffer()
                    )

                pdf_paths.append(
                    file_path
                )


            chunks = process_documents(
                pdf_paths
            )


        st.session_state["processed"] = True

        st.session_state["chunk_count"] = len(
            chunks
        )

        st.session_state["document_count"] = len(
            uploaded_files
        )

        st.success(
            "Documents processed successfully."
        )


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    if st.session_state.get(
        "processed",
        False
    ):

        st.markdown(
            '<div class="section-title">📊 Knowledge Base</div>',
            unsafe_allow_html=True
        )

        metric1, metric2 = st.columns(2)

        with metric1:

            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-number">{st.session_state["document_count"]}</div>'
                f'<div class="metric-label">Documents</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        with metric2:

            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-number">{st.session_state["chunk_count"]}</div>'
                f'<div class="metric-label">Chunks</div>'
                f'</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown("### 🧠 How it works")

    st.caption(
        "Documents are converted into embeddings and stored "
        "in a FAISS vector database. Relevant passages are "
        "retrieved and reranked before generating a grounded answer."
    )


# ============================================================
# RIGHT COLUMN — QUESTIONS
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">✨ Ask Your Documents</div>',
        unsafe_allow_html=True
    )


    question = st.text_input(
        "Question",
        placeholder="Ask anything about your uploaded documents...",
        label_visibility="collapsed"
    )


    ask_button = st.button(
        "✨ Ask Question",
        type="primary",
        use_container_width=True,
        disabled=not question.strip()
    )


    if ask_button:

        if not os.path.exists(
            INDEX_PATH
        ):

            st.error(
                "Please upload and process your documents first."
            )

        else:

            with st.spinner(
                "Searching your documents..."
            ):

                vector_store.load_index(
                    INDEX_PATH
                )

                chunks = vector_store.load_chunks(
                    CHUNKS_PATH
                )

                result = answer_question(
                    question,
                    chunks,
                    k=3
                )


            # =================================================
            # ANSWER CARD
            # =================================================

            st.markdown(
                '<div class="answer-card">'
                '<div class="answer-label">✨ AI Answer</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                result["answer"]
            )


            # =================================================
            # SOURCES
            # =================================================

            st.markdown(
                '<div class="section-title">🔎 Sources</div>',
                unsafe_allow_html=True
            )


            for source in result["sources"]:

                chunk = source["chunk"]

                document = html.escape(
                    chunk["document"]
                )

                start_page = chunk["page"]

                end_page = chunk.get(
                    "end_page",
                    start_page
                )


                if start_page == end_page:

                    page_label = (
                        f"Page {start_page}"
                    )

                else:

                    page_label = (
                        f"Pages {start_page}–{end_page}"
                    )


                with st.expander(
                    f"📄 {document}  •  {page_label}"
                ):

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-document">
                                📄 {document}
                            </div>

                            <div class="source-page">
                                {page_label}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(
                        chunk["text"]
                    )


    # ========================================================
    # EXAMPLE QUESTIONS
    # ========================================================

    if not question:

        st.markdown(
            """
            <div style="
                margin-top:35px;
                color:#7f8795;
                font-size:0.85rem;
            ">
                <b>Try asking:</b>
            </div>
            """,
            unsafe_allow_html=True
        )


        examples = [
            "What are the five interrupt sources in the 8051?",
            "What is CRC?",
            "What are the characteristics of embedded systems?"
        ]


        for example in examples:

            st.markdown(
                f"""
                <div style="
                    background:rgba(255,255,255,0.025);
                    border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;
                    padding:10px 13px;
                    margin:7px 0;
                    color:#aeb4c0;
                    font-size:0.85rem;
                ">
                    💬 {html.escape(example)}
                </div>
                """,
                unsafe_allow_html=True
            )