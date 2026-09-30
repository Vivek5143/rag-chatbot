# Agentic AI RAG Chatbot

A production-oriented Retrieval-Augmented Generation (RAG) chatbot built to answer questions **strictly from the provided Agentic AI eBook**.

The application utilizes document ingestion, OpenAI embeddings, Pinecone vector search, LangGraph orchestration, evidence sufficiency evaluation, constrained answer generation, and independent grounding validation.

The system is designed to **fail closed**: if the retrieved evidence is insufficient, or if the generated answer cannot be strictly grounded in the supporting evidence, the application refuses to answer rather than relying on general model knowledge.

---

## Current Verification & Submission Status

| Component / Stage | Status | Verification Detail |
|---|---|---|
| Repository Structure & Data | ✅ **VERIFIED** | Clean architecture, `data/Ebook-Agentic-AI.pdf` present |
| PDF Ingestion & Text Extraction | ✅ **VERIFIED** | 60 pages extracted, 59 text-bearing, page 1 empty cover |
| Deterministic Chunking | ✅ **VERIFIED** | 119 chunks produced, stable chunk IDs & SHA-256 provenance |
| OpenAI Embedding Adapter | ✅ **VERIFIED (Offline)** | Tested with `text-embedding-3-small` (1536 dim), batching & error handling |
| Pinecone Vector Adapter | ✅ **VERIFIED (Offline)** | Dimension, cosine metric, host, and namespace isolation verified |
| Pinecone Remote Index | ✅ **PROVISIONED** | Index `rag-chatbot` created (Dense, cosine, dim=1536); currently contains 0 vectors |
| Evidence Evaluation Gate | ✅ **VERIFIED (Offline)** | Fail-closed evaluation (relevance, completeness, support, entailment) |
| LangGraph Workflow | ✅ **VERIFIED (Offline)** | Graph routing, state propagation, and refusal branch execution |
| Constrained Generation | ✅ **VERIFIED (Offline)** | Generator receives only grader-approved supporting evidence |
| Grounding Validation Gate | ✅ **VERIFIED (Offline)** | Independent claim-by-claim verification against supporting evidence |
| FastAPI Application (`/health`, `/chat`) | ✅ **VERIFIED** | Process liveness, 422 validation, typed contracts, 503 operational mapping |
| Automated Test Suite | ✅ **VERIFIED** | **105 passed** in `python -m pytest -q` |
| Bytecode Compilation | ✅ **VERIFIED** | `python -m compileall -q app.py src tests` passed with 0 errors |
| API Smoke Test Suite | ✅ **VERIFIED** | `python smoke_test.py` passed all assertion checks |
| **Live Remote Indexing** | ⛔ **BLOCKED** | Blocked by external OpenAI API quota (`429 insufficient_quota`) |
| **Live RAG End-to-End Benchmark** | ⛔ **NOT VERIFIED** | Awaiting OpenAI account credit/quota replenishment |

> **Current Live Blocker Note:**  
> The offline implementation, orchestration architecture, and safety guardrails are 100% complete and fully verified. However, live document embedding and live query execution cannot complete because the available OpenAI API key currently returns HTTP `429 insufficient_quota`.  
> Consequently, the remote Pinecone index (`rag-chatbot`, namespace `agentic-ai`) is provisioned and healthy but currently holds **0 vectors**. Live RAG functionality is **not claimed as verified** until OpenAI quota is restored.

---

## 1. Problem Statement & Objectives

Standard LLM chat applications frequently hallucinate or rely on pre-trained parametric knowledge when asked domain-specific questions, or when asked questions outside their supplied reference materials.

For this system, the core objectives are:
1. **The eBook is the Single Source of Truth**: The system must answer exclusively from `data/Ebook-Agentic-AI.pdf`.
2. **Retrieval ≠ Answerability**: The mere fact that vector similarity returns candidate chunks does not mean the question is answerable. Retrieved chunks must be critically graded before generation.
3. **Fail-Closed Stance**: If retrieved evidence is incomplete, tangential, or missing, the system must immediately refuse.
4. **Constrained Generation**: The generator only sees evidence explicitly vetted by the evidence grader.
5. **Independent Grounding Validation**: An independent evaluator verifies every claim in the generated answer against the supporting chunks. If any claim lacks entailment, the answer is replaced with a canonical refusal.
6. **Separation of Refusals vs. Operational Errors**: Document-level refusals return HTTP 200 with a standard refusal payload. Infrastructure or provider failures (e.g., timeouts, rate limits, malformed model outputs) return HTTP 503.

---

## 2. Architecture & Complete RAG Flow

### End-to-End Architecture

```text
                                INGESTION PIPELINE (Offline / Setup)
                                
       ┌────────────────────────┐
       │ data/Ebook-Agentic-AI  │
       │        .pdf            │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  pypdf Text Extractor  │ ──► 60 Pages (59 text-bearing, Page 1 empty cover)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ RecursiveCharacter-    │ ──► 119 Chunks (Size: 1000, Overlap: 200)
       │      TextSplitter      │ ──► Deterministic Chunk IDs & Provenance
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ OpenAI Embeddings      │ ──► Model: text-embedding-3-small (1536 dim)
       │ (text-embedding-3-small)│
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Pinecone Vector DB     │ ──► Index: rag-chatbot | Namespace: agentic-ai
       │   (119 Dense Vectors)  │
       └────────────────────────┘

                                QUERY & INFERENCE PIPELINE (Runtime)

                                   ┌──────────────────────┐
                                   │      User Query      │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │   FastAPI POST /chat │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │  LangGraph Workflow  │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ 1. Retrieve Node     │
                                   │  - Embed query       │
                                   │  - Pinecone Top-K    │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ 2. Evidence Grader   │
                                   │  - Relevance         │
                                   │  - Completeness      │
                                   │  - Support score     │
                                   │  - Entailment score  │
                                   └──────────┬───────────┘
                                              │
                                ┌─────────────┴─────────────┐
                         [Insufficient]                [Sufficient]
                                │                           │
                                ▼                           ▼
                      ┌──────────────────┐        ┌──────────────────┐
                      │  Refusal Branch  │        │ 3. Generate Node │
                      └────────┬─────────┘        │  - Supporting    │
                               │                  │    context only  │
                               │                  └─────────┬────────┘
                               │                            │
                               │                            ▼
                               │                  ┌──────────────────┐
                               │                  │ 4. Grounding     │
                               │                  │    Validator     │
                               │                  └─────────┬────────┘
                               │                            │
                               │              ┌─────────────┴─────────────┐
                               │         [Ungrounded /              [Fully
                               │          Uncertain]               Grounded]
                               │              │                         │
                               ▼              ▼                         ▼
                      ┌──────────────────────────┐            ┌──────────────────┐
                      │     Canonical Refusal    │            │    Grounded      │
                      │  confidence_score: 0.0   │            │     Answer       │
                      │  grounded: false         │            │  provenance-rich │
                      │  refused: true           │            └──────────────────┘
                      └──────────────────────────┘
```

---

## 3. LangGraph Workflow Details

The system orchestrates the query lifecycle through a state machine built with LangGraph:

1. **`retrieve`**:
   - Takes the cleaned user query.
   - Generates a 1536-dimensional embedding using OpenAI `text-embedding-3-small`.
   - Queries Pinecone index `rag-chatbot` in namespace `agentic-ai` for top-$K$ candidates.
   - Preserves chunk text, page provenance, chunk IDs, and similarity scores.

2. **`evaluate_evidence`**:
   - Critically assesses the retrieved candidates against the question.
   - Evaluates relevance, completeness, missing aspects, support score, and entailment score.
   - Filters out non-supporting chunks and identifies exact `supporting_chunk_ids`.
   - **Fail-Closed Gate**: If evidence is missing, tangential, or incomplete, the graph branches directly to refusal.

3. **`generate`**:
   - Executes only when the evidence grader marks the context as sufficient.
   - Isolates generation to **only** the vetted supporting chunks.
   - Strict system prompt prohibits outside knowledge, extrapolation, and fabricated citations.

4. **`validate`**:
   - An independent grounding validation node receives the generated answer alongside the supporting evidence.
   - Evaluates every factual claim for direct support, ambiguity, and contradiction.
   - Checks entailment thresholds.

5. **`answer / refusal`**:
   - If grounded, constructs the final response with cited page provenance and confidence score.
   - If ungrounded or ambiguous, overwrites the answer with the canonical refusal string.

---

## 4. Strict Grounding & Fail-Closed Guardrails

### 1. Retrieval Does Not Equal Answerability
A vector database will always return nearest neighbors based on cosine similarity, even for completely unrelated queries or adversarial questions. The presence of retrieved chunks is treated strictly as **candidate evidence**, never as proof that an answer can be synthesized.

### 2. Evidence Evaluation & Fail-Closed Behavior
Before any answer generation can take place, the `EvidenceGrader` must confirm:
- `is_relevant = True`
- `is_sufficient = True`
- `is_complete = True`
- `support_score >= threshold` (default: 0.70)
- `entailment_score >= threshold` (default: 0.70)
- Supporting candidate list is non-empty.

If any criterion fails, the grader fails closed. The application can downgrade an overly optimistic score, but it never upgrades insufficient evidence.

### 3. Constrained Generation
Unselected retrieved chunks are discarded before generation. The generation prompt enforces:
- Answering solely from provided chunk blocks.
- No general pre-training knowledge.
- No speculation or logical leaps beyond what the text states.

### 4. Independent Grounding Validation
The generation model is never allowed to grade its own homework. An independent validation step evaluates the generated answer against the supporting chunks. If unsupported claims or ambiguous statements are detected, the graph diverts to refusal.

### 5. Canonical Refusal Contract
Whenever a question cannot be answered from the eBook, the response is deterministic:

```text
I cannot answer this based on the information available in the provided document.
```

With payload fields:
```json
{
  "confidence_score": 0.0,
  "grounded": false,
  "refused": true
}
```

### 6. Separation of Document Refusals vs. Operational Errors
- **Document Refusal (HTTP 200)**: The RAG pipeline executed successfully and determined that the eBook does not contain sufficient information to answer the question.
- **Operational Error (HTTP 503)**: An upstream dependency failed (OpenAI rate limit / 429, Pinecone network timeout, invalid schema parsing). Operational errors are never masked as document refusals.

---

## 5. Project Structure

```text
rag-chatbot/
│
├── data/
│   └── Ebook-Agentic-AI.pdf          # Canonical eBook source document
│
├── src/
│   ├── __init__.py
│   ├── api.py                        # FastAPI router & endpoint definitions
│   ├── config.py                     # Typed settings (Pydantic Settings)
│   ├── embeddings.py                 # OpenAI embedding provider & adapter
│   ├── generation.py                 # Constrained generation service & prompt
│   ├── grading.py                    # Evidence evaluation & fail-closed logic
│   ├── graph.py                      # LangGraph state machine & workflow
│   ├── indexing.py                   # Ingestion & Pinecone vector indexing script
│   ├── ingestion.py                  # PDF extraction & deterministic chunking
│   ├── models.py                     # Core Pydantic domain & API models
│   ├── retrieval.py                  # Pinecone vector retrieval service
│   └── validation.py                 # Independent grounding validation service
│
├── tests/
│   ├── test_api.py                   # API contract, validation & 503 error tests
│   ├── test_app.py                   # FastAPI /health & basic endpoint tests
│   ├── test_config.py                # Environment configuration tests
│   ├── test_embeddings.py            # Embedding adapter & batching tests
│   ├── test_generation.py            # Evidence-constrained generation tests
│   ├── test_grading.py               # Grader sufficiency & fail-closed tests
│   ├── test_graph.py                 # LangGraph routing & state transition tests
│   ├── test_indexing.py              # Ingestion & Pinecone indexing contract tests
│   ├── test_ingestion.py             # PDF text extraction & chunk stability tests
│   ├── test_models.py                # Request/response model validation tests
│   ├── test_retrieval.py             # Retrieval query & metadata handling tests
│   └── test_validation.py            # Grounding claim validation tests
│
├── docs/                             # Design documents & specifications
├── app.py                            # FastAPI application entrypoint
├── smoke_test.py                     # Fast standalone API integration smoke tests
├── requirements.txt                  # Python dependencies
├── .env.example                      # Documented environment variable template
├── .gitignore                        # Git exclusion rules (protects secrets/.env)
├── README.md                         # Project documentation
├── prd.md                            # Product Requirements Document
├── backlog.md                        # Task tracking & milestone backlog
├── implementation_plan.md            # Technical architecture & implementation plan
├── implementation_plan_phase4.md     # Phase 4 verification & guardrails plan
└── system.md                         # System architecture & interface reference
```

---

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core runtime environment |
| **FastAPI** | High-performance ASGI web framework for `/health` and `/chat` |
| **LangGraph** | Stateful multi-step RAG workflow orchestration and conditional routing |
| **LangChain Text Splitters** | Document chunking with recursive boundary splitting |
| **pypdf** | Robust PDF page and text extraction |
| **OpenAI API** | Vector embeddings (`text-embedding-3-small`) and reasoning (`gpt-4o-mini`) |
| **Pinecone** | Managed vector database for dense cosine similarity search |
| **Pydantic v2** | Strict data modeling, runtime type enforcement, and settings management |
| **Pytest** | Automated unit and integration testing suite |
| **Uvicorn** | Production-ready ASGI web server |

---

## 7. Setup & Installation

### Prerequisites
- Python 3.10 or newer
- An active Pinecone account and API key
- An active OpenAI account and API key

### 1. Clone & Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to a local `.env` file:
```bash
cp .env.example .env
```

Populate the required configuration options in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=

PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-chatbot
PINECONE_NAMESPACE=agentic-ai
```

> **Security Note:**  
> Never commit `.env` or real API keys to source control. The repository `.gitignore` explicitly ignores `.env`, virtual environments, and caches. Verify with `git status` before pushing changes.

---

## 8. Canonical Source Document & Ingestion

### Source Document
The sole knowledge source for this application is:
```text
data/Ebook-Agentic-AI.pdf
```
- **Total Pages**: 60
- **Page 1**: Empty cover page
- **Pages 2–60**: Text-bearing document content (59 pages)
- **SHA-256 Digest**: `532452cd94aff725328eb02a9722fdcbb370803abfc552b2d2e1c93620b7d302`

### Ingestion & Chunking
The ingestion pipeline (`src/ingestion.py`) processes the PDF as follows:
1. Loads pages using `pypdf`, preserving 1-based page numbers.
2. Filters unextractable pages (such as blank cover pages).
3. Splits text using `RecursiveCharacterTextSplitter` configured for `chunk_size = 1000` and `chunk_overlap = 200`.
4. Produces exactly **119 chunks** for the canonical document.
5. Generates deterministic vector IDs formatted as:
   ```text
   {document_sha256[:16]}-p{page_number}-c{chunk_index}
   ```
6. Stores rich provenance metadata per chunk:
   - `chunk_id`
   - `source` (`Ebook-Agentic-AI.pdf`)
   - `page` (integer page number)
   - `source_hash`
   - `chunk_index`
   - `content` (chunk text)

---

## 9. Pinecone Configuration & Indexing

### Index Specifications
The application targets a pre-provisioned Pinecone index:
- **Index Name**: `rag-chatbot`
- **Vector Type**: `Dense`
- **Metric**: `cosine`
- **Dimension**: `1536`
- **Namespace**: `agentic-ai`

> **Index Lifecycle Notice:**  
> The repository **does not** automatically create, delete, or recreate the Pinecone index during application startup or runtime. This prevents unintended index deletion, data loss, or cold-start race conditions. The index must be provisioned in your Pinecone console before running indexing.

### Indexing Command
To ingest the PDF, generate embeddings, and upsert vectors into Pinecone, run:
```bash
python -m src.indexing
```

**Indexing Workflow:**
1. Loads and chunks `data/Ebook-Agentic-AI.pdf`.
2. Verifies the Pinecone index exists and validates dimension (`1536`) and metric (`cosine`).
3. Passes the 119 chunks to the OpenAI embedding provider in batches.
4. Upserts 119 vectors with full provenance metadata into namespace `agentic-ai`.

---

## 10. Running the API

Start the FastAPI application with Uvicorn:
```bash
python -m uvicorn app:app
```
By default, the server listens at `http://127.0.0.1:8000`.

### Health Check Endpoint
**`GET /health`**

Confirms application process liveness.
```bash
curl http://127.0.0.1:8000/health
```

**Example Response:**
```json
{
  "status": "ok",
  "phase": "api"
}
```
*(Note: `/health` confirms API process liveness only; it does not invoke external providers).*

---

## 11. Chat API Specification

**`POST /chat`**

Processes questions through the LangGraph RAG workflow.

### Request Body
```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

### Successful Grounded Response (Example)
```json
{
  "final_answer": "Agentic AI refers to autonomous systems capable of perceiving their environment, reasoning, making decisions, and taking actions to accomplish specific goals with minimal human intervention.",
  "retrieved_context": [
    {
      "chunk_id": "532452cd94aff725-p3-c0",
      "source": "Ebook-Agentic-AI.pdf",
      "page": 3,
      "similarity_score": 0.884
    },
    {
      "chunk_id": "532452cd94aff725-p4-c1",
      "source": "Ebook-Agentic-AI.pdf",
      "page": 4,
      "similarity_score": 0.821
    }
  ],
  "confidence_score": 0.884,
  "grounded": true,
  "refused": false
}
```

### Refusal Response (Example)
Returned with HTTP 200 when evidence is insufficient or question is outside the eBook:
```json
{
  "final_answer": "I cannot answer this based on the information available in the provided document.",
  "retrieved_context": [],
  "confidence_score": 0.0,
  "grounded": false,
  "refused": true
}
```

### Operational Error Response (Example)
Returned with HTTP 503 if an external provider or network call fails:
```json
{
  "detail": "Upstream model provider is currently unavailable or returned an error."
}
```

---

## 12. Benchmark Questions

The assignment defines five specific benchmark questions to evaluate both grounded retrieval and refusal handling:

| # | Benchmark Question | Expected System Behavior |
|---|---|---|
| 1 | **What is Agentic AI according to the eBook?** | Grounded answer derived strictly from eBook introductory pages with chunk citations. |
| 2 | **How do AI agents differ from traditional automation systems?** | Grounded comparative answer detailing autonomy, reasoning, and adaptive behavior vs. rule-based automation. |
| 3 | **What are the core components of an Agentic Architecture?** | Grounded breakdown of architectural components (perception, planning, reasoning, memory, execution). |
| 4 | **What role does memory play in Agentic AI workflows?** | Grounded explanation of short-term vs. long-term memory, state retention, and context preservation. |
| 5 | **Who won the 2022 FIFA World Cup?** | **Canonical Refusal.** The document contains no sports or FIFA World Cup data. Must not use general LLM knowledge. |

### Expected FIFA Refusal Payload
```json
{
  "final_answer": "I cannot answer this based on the information available in the provided document.",
  "confidence_score": 0.0,
  "grounded": false,
  "refused": true
}
```

---

## 13. Testing & Verification

The repository contains an exhaustive offline test suite that validates every component, edge case, and safety gate without consuming external API credits.

### 1. Run Unit & Integration Tests
```bash
python -m pytest -q
```
**Verified Result:**
```text
105 passed in 54.52s
```

### 2. Bytecode Compilation Verification
```bash
python -m compileall -q app.py src tests
```
**Verified Result:** Passed with 0 errors (clean compilation).

### 3. Standalone Smoke Test Suite
```bash
python smoke_test.py
```
**Verified Result:**
```text
SMOKE TEST PASSED: /health, 422, grounded 200, refusal 200, 503
```

---

## 14. Steps to Complete Live Validation

Once an OpenAI API key with active credits/quota is configured, the live end-to-end pipeline can be validated using the following steps:

1. **Populate Credentials**: Ensure `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env` have available balance.
2. **Execute Indexing**:
   ```bash
   python -m src.indexing
   ```
3. **Verify Pinecone Vectors**: Confirm that index `rag-chatbot`, namespace `agentic-ai`, contains **119 vectors** with complete metadata.
4. **Launch Application**:
   ```bash
   python -m uvicorn app:app
   ```
5. **Run Benchmark Queries**: Execute questions 1–4 via `POST /chat` and verify page provenance and positive confidence scores.
6. **Run Refusal Verification**: Execute question 5 ("Who won the 2022 FIFA World Cup?") and adversarial prompts; confirm exact canonical refusal response.
7. **Verify Grounding Audit**: Verify that all successful responses only contain statements directly supported by the retrieved chunks.

---

## 15. Limitations

The system design deliberately excludes out-of-scope complexity to keep the RAG implementation clean, robust, and verifiable:
- **Single-Document Scope**: Built specifically for `data/Ebook-Agentic-AI.pdf`.
- **No OCR Support**: Only processes digitally embedded text; scanned or pure image PDFs require an OCR pre-processor.
- **Stateless Chat**: Does not maintain multi-turn conversational history; each query is treated as an independent question.
- **No Automatic Cloud Provisioning**: Does not automatically create or destroy cloud infrastructure.
- **No Custom UI**: Exposes an HTTP API for integration rather than a graphical frontend.