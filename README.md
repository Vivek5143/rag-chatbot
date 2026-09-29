# Agentic AI eBook RAG Chatbot

This repository is a strictly document-grounded RAG chatbot over the provided
Agentic AI eBook. The current implementation includes typed configuration, PDF
ingestion/chunking, OpenAI-to-Pinecone indexing, retrieval, evidence evaluation,
LangGraph orchestration, constrained evidence-only answer generation,
independent grounding validation, and a **Phase 8 FastAPI integration** that
exposes the validated workflow through `POST /chat`.

## Phase 8 API boundary

`POST /chat` invokes the single compiled LangGraph as its only orchestration
path. The API performs no embedding, Pinecone retrieval, evidence grading,
answer generation, or grounding validation of its own — the graph does all of
that. The API is a transport and error-mapping boundary only; it never
generates fallback answers, uses general model knowledge, overrides evidence
grading, overrides grounding validation, or marks an answer grounded.

## HTTP status mapping

| Situation | HTTP | Response |
| --- | --- | --- |
| Valid grounded answer | `200` | `final_answer`, provenance-rich `retrieved_context`, evidence `confidence_score`, `grounded=true` |
| Evidence insufficient | `200` | Canonical refusal, `confidence_score=0.0`, `grounded=false`, diagnostic context preserved |
| Grounding validation refusal | `200` | Canonical refusal, `confidence_score=0.0`, `grounded=false`, unsupported generated answer never exposed |
| Invalid request (blank/whitespace/missing/wrong type/oversized) | `422` | Standard FastAPI validation error |
| Provider/Pinecone/embedding/grader/generator/validator/config operational failure | `503` | `{"detail": "RAG service is currently unavailable."}` — no stack traces, no secrets |

**200 refusal ≠ system failure.** A document refusal is a legitimate outcome
when the eBook does not support the answer. **503 ≠ document refusal.** An
operational failure is a service problem, never a claim that the document lacks
the answer. An empty or unprovisioned Pinecone index is a `503`, not a refusal.

## Prerequisites

- Python 3.10 or newer
- OpenAI and Pinecone credentials are not required to run the offline test suite.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Populate `.env` only when a later phase needs provider access. Do not commit
that file.

For explicit indexing, populate these provider settings:

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=rag-chatbot
PINECONE_NAMESPACE=agentic-ai
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=64
PINECONE_UPSERT_BATCH_SIZE=100
```

`PINECONE_NAMESPACE` is intentionally required for indexing. Use a
content-versioned namespace (for example, a document-hash-derived value) so a
changed eBook is not silently mixed with a previous version.

## PDF ingestion

The canonical source is expected at `data/Ebook-Agentic-AI.pdf`. It is not
included in this repository. The ingestion API raises a clear missing-file error
until the approved PDF is supplied; it never downloads or substitutes a source.

`src.ingestion` exposes:

- `load_pdf_pages(path)` for page-aware `Document` objects;
- `chunk_documents(documents, chunk_size, chunk_overlap)` for embedding-ready
  chunks with source, page, document hash, chunk index, and deterministic ID;
- `ingest_pdf(path, ...)` to run both steps explicitly.

Ingestion remains separate from FastAPI startup.

## Explicit indexing

The index must be provisioned separately before running ingestion. This project
never creates, deletes, or recreates a Pinecone index. For the default
`text-embedding-3-small` model, configure an existing dense Pinecone index with
**cosine** similarity and dimension **1536**. The command validates readiness,
metric, and dimension before it requests embeddings or writes vectors.

With the approved PDF present at `data/Ebook-Agentic-AI.pdf` and provider
configuration supplied, run:

```powershell
.venv\Scripts\python.exe -m src.indexing
```

The command loads and chunks the PDF, embeds non-empty chunks in batches, and
upserts bounded batches to the configured namespace. It reports counts without
printing credentials. If the canonical PDF is absent, it fails clearly and does
not download or substitute a document.

Each Pinecone vector ID is Phase 2's deterministic `chunk_id` (derived from
the document hash, page, chunk position, and content). Re-running the same
document upserts those same IDs, so it updates existing logical vectors instead
of creating duplicates. Vector metadata preserves `chunk_id`, `source`, `page`,
and `source_hash`, plus the chunk text once as `content` for future
evidence-preserving retrieval. The default 1,000-character chunks fit the
pipeline's 8 KB metadata-content guard; an oversized chunk fails instead of
being silently truncated.

FastAPI does not call this command, OpenAI, or Pinecone during import or
startup. `GET /health` and `POST /chat` remain Phase 1 behavior until a later
phase deliberately wires the validated workflow to the API.

## Run the API

```powershell
python -m uvicorn app:app --reload
```

- `GET /health` confirms that the API process is running. It is lightweight and
  performs no ingestion, embedding, retrieval, or provider calls.
- `POST /chat` validates its request, invokes the compiled LangGraph once, and
  returns a typed response. It never fabricates a RAG answer.

### `GET /health`

Returns `{"status": "ok", "phase": "api"}`. Process liveness only.

### `POST /chat`

Request body — `ChatRequest`:

```json
{ "query": "What is Agentic AI according to the eBook?" }
```

`query` is trimmed, must contain at least one non-whitespace character, and is
capped at 2,000 characters. Blank, whitespace-only, missing, wrong-type, and
oversized queries return standard FastAPI `422` validation errors. The API never
silently rewrites the user's question.

Response body — `ChatResponse`:

```json
{
  "final_answer": "Agentic AI is ...",
  "retrieved_context": [
    {
      "chunk_id": "sha256:abc123",
      "content": "The eBook states ...",
      "source": "Ebook-Agentic-AI.pdf",
      "page": 12,
      "source_hash": "d6e2c5...",
      "similarity_score": 0.91
    }
  ],
  "confidence_score": 0.88,
  "grounded": true,
  "refusal_reason": null
}
```

`retrieved_context` is capped at the configured retrieval `top_k` (default 4)
and preserves `chunk_id`, `content`, `source`, `page`, `source_hash`, and raw
`similarity_score` for every candidate. `confidence_score` is the existing
evidence-confidence value (`min(support_score, entailment_score)`); it is
`0.0` for every refusal and non-zero only for grounded answers.

### Refusal example

```json
{
  "final_answer": "I cannot answer this based on the information available in the provided document.",
  "retrieved_context": [],
  "confidence_score": 0.0,
  "grounded": false,
  "refusal_reason": "insufficient_evidence"
}
```

### `503` operational-error behavior

```json
{ "detail": "RAG service is currently unavailable." }
```

This is returned for OpenAI provider failure/timeout, Pinecone failure, embedding
failure, malformed structured grader/generation/validation output, invalid score,
invalid supporting chunk ID, configuration/index-readiness failure, and other
typed operational RAG errors. It is never converted into a document refusal and
never exposes API keys, credentials, stack traces, or raw provider responses.

### Index provisioning requirement

The Pinecone index must be provisioned and the eBook ingested **before** real
queries. The API does not create, delete, or recreate the index and does not
ingest the PDF at startup. See *Explicit indexing* above.

## Test

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q app.py src tests
```

Tests use offline fakes for OpenAI embeddings and Pinecone. No live indexing is
claimed unless the canonical PDF, real credentials, and the explicit command
have all been used.

## Live evaluation status

The canonical PDF is present and validated (`data/Ebook-Agentic-AI.pdf`,
20,267,549 bytes, SHA-256
`532452cd94aff725328eb02a9722fdcbb370803abfc552b2d2e1c93620b7d302`).
Offline ingestion produces **119 chunks** across 59 pages (2–60) with unique
deterministic chunk IDs and full provenance. All 105 tests pass, compilation is
clean, and the smoke suite (`/health`, `422`, grounded `200`, refusal `200`,
`503`) passes.

**Live ingestion and benchmark evaluation could not be completed:**

| Check | Result |
| --- | --- |
| Canonical PDF present | ✅ |
| Offline ingestion (chunks, IDs, provenance) | ✅ 119 chunks / 59 pages |
| Offline test suite | ✅ 105 passed |
| Compilation | ✅ `COMPILE_OK` |
| Smoke suite (API contract) | ✅ passed |
| Pinecone index `rag-chatbot` exists | ✅ dense, cosine, 1536 dim, ready (host `rag-chatbot-cj953jk.svc.aped-4627-b74a.pinecone.io`) |
| OpenAI key usable | ❌ 429 `insufficient_quota` — no credits remaining |
| OpenRouter fallback | ❌ **Architecturally incompatible** — no embedding models offered |

Per the project's explicit provisioning policy, this repository never creates,
deletes, or recreates a Pinecone index, so the indexing command fails at the
embedding call with `Indexing failed: OpenAI embedding request failed`. The API
itself is unaffected — it starts and serves `GET /health`; `POST /chat` is ready
to run against a provisioned index and funded key.

## Optional OpenAI-compatible base URL

`OPENAI_BASE_URL` is an optional, empty-by-default setting. When set, every
OpenAI SDK client (embeddings, evidence grading, generation, grounding
validation) is constructed with that `base_url`, redirecting them at an
alternate OpenAI-compatible provider. It is an explicit operator choice, never
a silent default, and OpenAI remains the documented assignment default:

```env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

No provider logic was hardcoded: `build_openai_client(settings)` in
`src/config.py` applies the base URL only when non-blank, and the four adapters
(`OpenAIEmbeddingProvider`, `OpenAIEvidenceGraderAdapter`,
`OpenAIGenerationAdapter`, `OpenAIGroundingValidatorAdapter`) each accept an
optional `base_url` parameter. Offline tests are unaffected (105 pass).

### OpenRouter compatibility findings

A live probe of the configured OpenRouter key (`sk-or-v1-...`) found three
incompatible capabilities; no architectural change was made:

| Operation | OpenRouter result |
| --- | --- |
| Chat completion (`openai/gpt-4o-mini`) | ❌ 402 — account has never purchased credits (`total_credits: 0`) |
| Structured output (`responses.parse`) on free models | ❌ 429 — daily free-model limit reached ("Add 10 credits to unlock 1000 free model requests per day") |
| **Embeddings (`openai/text-embedding-3-small`)** | ❌ **Not offered at all — OpenRouter lists zero embedding models** |

The embedding gap is the hard blocker: chunk embedding and question embedding
both require a text embedding model, and the configured dimension (1536) must
match the Pinecone index. Changing the base URL alone cannot supply one, so
retrieval — and therefore the entire RAG pipeline — cannot run through
OpenRouter without a different embedding provider. This was reported before
any architectural change, per the compatibility-check requirement.

The project plan and architecture are documented in `prd.md`, `backlog.md`,
`implementation_plan.md`, and `system.md`.
