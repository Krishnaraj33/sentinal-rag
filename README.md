# SentinelRAG: RAG Search with Access Control and Document Updates

## 1. Overview
I built SentinelRAG to work through two common problems in Retrieval-Augmented Generation (RAG) systems.
The first problem is access control. A vector search can return a document that a user should not be allowed to read. Filtering those results afterward can also leave the model with too little useful context.
The second problem is keeping vectors up to date. When a source document changes, old chunks can stay in the vector database unless the system updates them. That can lead to answers based on old information.
### What the project does
Pre-Retrieval Document-Level Security (DLS): User roles and clearance are turned into Qdrant payload filters before the vector search runs. This keeps unauthorized documents out of the retrieval step. In the current test setup, the DLS test reported 0.00% context leakage.
Chunk-Level Updates with SHA-256: Each chunk gets a SHA-256 hash and a deterministic Point ID. When a document changes, the system compares the new chunk hashes with the stored ones and only re-embeds the chunks that changed. The benchmark reports 70% to 95% savings in embedding token use.
Access Checks Before the LLM: Requests from users who do not have the required clearance return HTTP 403 before LLM generation, so those requests use 0 LLM generation tokens.
Streamlit Demo: The project includes a UI where I can switch between test users, try queries, change documents, and see how permissions and vector updates behave.

## 2. System Architecture
The system is split into a few services so that the query path and document update path can be tested separately.
```
┌────────────────────────────────────────────────────────────────────────┐
│                        Interactive Demo UI (Streamlit)                 │
│         - Persona Switcher (Contractor, Engineer, Finance, Executive)  │
│         - Real-Time Query Console & Payload Filter Visualizer          │
│         - Document Mutation & Invalidation Sandbox                     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ HTTP (Port 8501)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway / Query Engine                     │
│  - Bearer JWT Validation & Claims Parsing                              │
│  - Pre-filter Translation & Hybrid Search Coordination                 │
│  - Cross-Encoder Re-Ranking & Grounded Response Synthesis              │
└──────────────┬──────────────────────────────────────────┬──────────────┘
               │                                          │
        [Query Path]                              [CDC Event Path]
               │                                          │
               ▼                                          ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│     Hybrid Search Engine     │          │     Redis Message Broker     │
│  - Qdrant Pre-Filter Search  │          │  - Ingestion Tasks           │
│  - BM25 Payload Search       │          │  - Invalidation Queues       │
│  - Cross-Encoder Re-ranker   │          └──────────────┬───────────────┘
└──────────────┬───────────────┘                         │
               │                                         ▼
               │                          ┌──────────────────────────────┐
               │                          │   Async Ingestion Worker     │
               │                          │  - Structural Chunking       │
               │                          │  - SHA-256 Diff Engine       │
               │                          │  - Embedding Generator       │
               │                          └──────────────┬───────────────┘
               ▼                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           Storage Subsystems                           │
│  ┌──────────────────────────────┐     ┌─────────────────────────────┐  │
│  │     Qdrant Vector Engine     │     │     PostgreSQL Metadata     │  │
│  │  - HNSW with Payload Filters │     │  - Documents & Chunk Hashes │  │
│  │  - Sub-second Tombstoning    │     │  - Role Mapping & Audits    │  │
│  └──────────────────────────────┘     └─────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```
## 3. Access Control Model
SentinelRAG checks document classification and user roles before a document can be included in retrieval.
### Test Personas
The project uses a few preset users to make the permission checks easy to test.
| Persona | User ID (`sub`) | Clearance | Assigned Roles | Accessible Documents |
| :--- | :--- | :--- | :--- | :--- |
| **Guest / Contractor** | `USR-CONTRACTOR-01` | `public` | `["contractor"]` | Public documents with wildcard `*` or `contractor` role. |
| **Software Engineer** | `USR-ENG-42` | `internal` | `["engineering"]` | Internal technical specs, runbooks, and public docs. |
| **Finance Manager** | `USR-FIN-09` | `confidential` | `["finance"]` | Confidential budgets, payroll, plus internal & public docs. |
| **Chief Executive Officer** | `USR-CEO-01` | `executive` | `["c-suite", "engineering", "finance", "hr"]` | All corporate records across all classifications and departments. |

---
## 4. Chunk Hashing and Point IDs
To avoid re-processing every chunk when a document changes, each chunk gets a stable ID and a SHA-256 hash of its content.

The update flow is simple: split the changed document into chunks, calculate the hashes, compare them with the stored values, and only process the chunks whose content changed.
```
                   ┌───────────────────────────────────────────────┐
                   │          Incoming Modified Document           │
                   └───────────────────────┬───────────────────────┘
                                           │ Split into Chunks
                                           ▼
                                ┌─────────────────────┐
                                │ Compute Chunk Hash  │
                                │   SHA-256(Chunk)    │
                                └──────────┬──────────┘
                                           │
                    Match against Database Record: (doc_id, chunk_index)
                                           │
                      ┌────────────────────┴────────────────────┐
                      │                                         │
                Hash Matches                              Hash Differs
                      │                                         │
                      ▼                                         ▼
            [Bypass Processing]                        [Targeted Action]
         • Zero Embedding API Cost                  • Call Embedding API
         • Zero Vector DB Writes                    • Overwrite Vector at Point ID
         • Zero Re-indexing Latency                 • Update Metadata & Timestamp
```
---
## 5. Repository Structure
The repository is organized around the API, search, database, vector handling, LLM code, background workers, UI, and tests.
```
SentinalRAG/
├── .env.example              # Environment configuration template
├── .gitignore                # Git exclusions (preserves secrets)
├── docker-compose.yml        # Orchestration for PostgreSQL, Redis, Qdrant, API, Worker, UI
├── Dockerfile.backend        # Container image for FastAPI Gateway and Celery Worker
├── Dockerfile.ui             # Container image for Streamlit Showcase UI
├── init.sql                  # PostgreSQL schema, indices, and extensions
├── requirements.txt          # Python dependencies
├── srs.md                    # IEEE Std 830-1998 Software Requirements Specification
├── README.md                 # System overview and operational guide
├── app/
│   ├── __init__.py
│   ├── config.py             # Pydantic BaseSettings with strict validation
│   ├── constants.py          # Ordinal clearance rankings and preset personas
│   ├── cli.py                # Administrative CLI (reindex, inspect, seed)
│   ├── main.py               # FastAPI application factory and lifespan hooks
│   ├── api/
│   │   ├── deps.py           # JWT claims extractor and webhook secret verification
│   │   └── v1/
│   │       ├── api.py        # Aggregate router
│   │       ├── auth.py       # JWT tokens and persona definitions
│   │       ├── query.py      # POST /api/v1/query (hybrid search + grounded synthesis)
│   │       ├── events.py     # POST /api/v1/events/document-change (CDC webhook)
│   │       ├── documents.py  # Document catalog and Security Boundary Inspector stats
│   │       └── health.py     # System readiness health check
│   ├── core/
│   │   ├── security.py       # JWT encoding/decoding and UserClaims schema
│   │   └── dls.py            # Dual-axis DLS evaluation logic
│   ├── db/
│   │   ├── base.py           # SQLAlchemy declarative base
│   │   ├── models.py         # Document, DocumentChunk, RetrievalAuditLog
│   │   └── session.py        # Async and sync database session factories
│   ├── vector/
│   │   ├── hashing.py        # SHA-256 content hashing and UUIDv5 deterministic point IDs
│   │   ├── filter_builder.py # Translates UserClaims to native Qdrant Filter
│   │   └── client.py         # Qdrant client manager, indexing, and point operations
│   ├── search/
│   │   ├── dense.py          # Pre-filtered dense vector retrieval
│   │   ├── bm25.py           # BM25Okapi lexical sparse scorer
│   │   ├── rrf.py            # Reciprocal Rank Fusion (k=60)
│   │   ├── reranker.py       # Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
│   │   └── hybrid_engine.py  # Search pipeline coordinator with metrics timing
│   ├── llm/
│   │   ├── client.py         # OpenAI-compatible / OpenRouter / Groq / Mock LLM
│   │   ├── embeddings.py     # SentenceTransformers / OpenAI / Mock vector embeddings
│   │   └── synthesizer.py    # Grounded synthesis with citation integrity verification
│   └── worker/
│       ├── celery_app.py     # Celery instance and Redis broker settings
│       ├── chunker.py        # Structural markdown and table-preserving chunker
│       ├── diff_engine.py    # Deterministic SHA-256 diff engine and targeted invalidator
│       └── tasks.py          # Background CDC tasks and inline execution fallback
├── ui/
│   ├── app.py                # Streamlit interactive showcase console
│   └── api_client.py         # HTTP client communicating with backend gateway
├── scripts/
│   ├── seed_data.py          # Populates canonical enterprise benchmark documents
│   └── run_benchmarks.py     # Automated Verification CLI (Section 11 scorecard)
└── tests/
    ├── conftest.py           # In-memory SQLite and in-memory Qdrant fixtures
    ├── test_dls_isolation.py # TEST-DLS-01: Zero context leakage across 1,000 synthetic chunks
    ├── test_chunk_diffing.py # TEST-DIFF-02: SHA-256 invalidation & 80% cost suppression
    ├── test_tombstoning.py   # TEST-PURGE-03: Deterministic point tombstoning in <= 100ms
    ├── test_permissions.py   # TEST-PERM-04: In-place reclassification with 0 embedding calls
    └── test_e2e_benchmark.py # Stage 2: 15-case end-to-end integration benchmark
```
---
## 6. Running the Project
There are two ways to run the project. I normally use the local Conda environment while working on the code. Docker Compose is also included for starting the full 6-service stack together.
### Option A: Local Python Environment
1. Install dependencies
Review `requirements.txt` and install the packages:
   ```pip install -r requirements.txt```
2. Configure the environment
Copy `.env.example` to `.env` and add the required settings:
   ```cp .env.example .env```
3. Run the benchmark
   ```python scripts/run_benchmarks.py ```
4. Run the test suite
   ```conda run -n omni pytest -v tests/```
5. Start the API
   ```uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload```
6. Start the Streamlit UI
   ```streamlit run ui/app.py --server.port 8501```
---
### Option B: Docker Compose
The project can also start the full service stack with Docker Compose:
```
docker compose up --build -d
```
To check the running containers and API logs:
```bash
docker compose ps
docker compose logs -f api
```
To run the benchmark inside the API container:
```bash
docker compose run --rm api python scripts/run_benchmarks.py
```