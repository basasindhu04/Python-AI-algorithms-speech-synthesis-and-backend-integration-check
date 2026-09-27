# AI Engineering Algorithms & Backend Integration Platform

A production-grade Python platform exposing core AI algorithms via high-performance FastAPI endpoints with strict Pydantic validation.

---

## 🚀 Overview & Key Modules

This repository implements the complete suite of AI engineering concepts evaluated in the assessment:

1. **Heuristic Search & Graph Reasoning (`app/services/search_service.py`)**
   - **A* Algorithm**: Min-heap (`heapq`) based A* algorithm for grid pathfinding and weighted graphs.
   - **Heuristics**: Manhattan distance, Euclidean distance, and Chebyshev distance.
   - **Graph Reasoning**: NetworkX integration for graph edge weight calculation and node traversal.

2. **Recommendation Systems (`app/services/recommendation_service.py`)**
   - **Collaborative Filtering**: User-User cosine similarity matrix calculation using NumPy and Pandas DataFrame pivot tables.
   - **Content-Based Filtering**: Item feature vectors matched against weighted user profile vectors.
   - **Hybrid Engine**: Weighted combination (60% Collaborative + 40% Content) with cold-start user fallback handling.

3. **NLP & Vector Embeddings (`app/services/nlp_service.py`)**
   - **Gensim TF-IDF**: Term Frequency-Inverse Document Frequency matrix similarity indexing.
   - **Vector Average Embedding**: Token feature hashing and cosine similarity ranking.

4. **Speech Synthesis (`app/services/speech_service.py`)**
   - **Local TTS (`pyttsx3`)**: Offline speech synthesis using native OS voice engines (SAPI5/eSpeak/NSSpeechSynthesizer).
   - **Async Thread Execution**: Non-blocking thread-pool execution model to prevent event loop blocking.
   - **Trade-off Analysis**: Comprehensive breakdown comparing local offline synthesis vs. cloud-based neural TTS services.

5. **API & Data Validation Layer (`app/api/routers/` & `app/schemas.py`)**
   - **FastAPI**: REST endpoints with HTTP error handling and OpenAPI interactive documentation (`/docs`).
   - **Pydantic v2**: Strict request/response payload validation schemas with forbidden extra fields.

---

## 🛠 Project Architecture

```
gpp/
├── app/
│   ├── main.py                  # FastAPI application entrypoint
│   ├── schemas.py               # Pydantic data schemas
│   ├── services/
│   │   ├── search_service.py    # A* Heuristic search & Graph reasoning
│   │   ├── recommendation_service.py # RecSys algorithms (NumPy / Pandas)
│   │   ├── nlp_service.py       # Gensim TF-IDF & Vector similarity
│   │   └── speech_service.py    # Local pyttsx3 speech synthesis
│   └── api/
│       └── routers/
│           ├── search.py        # /api/v1/search endpoints
│           ├── recommendation.py# /api/v1/recommendations endpoint
│           ├── nlp.py           # /api/v1/nlp endpoints
│           └── speech.py        # /api/v1/speech endpoint
├── tests/
│   ├── test_search.py
│   ├── test_recommendation.py
│   ├── test_nlp.py
│   ├── test_speech.py
│   └── test_api.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🧠 Assessment Written Reasoning & Trade-offs

### 1. Heuristic Search (A*)
> **Reasoning:** A* search guarantees optimality when supplied with an admissible heuristic $h(n)$ that never overestimates the true cost to the goal (such as Manhattan distance for grid movement without diagonals). Using a min-heap priority queue (`heapq`) yields $O(E \log V)$ time complexity by efficiently expanding the node with the minimum estimated total cost $f(n) = g(n) + h(n)$.

### 2. Vector Embeddings & Gensim NLP
> **Reasoning:** Gensim's TF-IDF model balances word frequency with corpus rarity to create sparse vector representations that highlight distinctive terms. For semantic search without full neural network overhead, combining term frequency weighting with NumPy batch cosine similarity provides sub-millisecond query matching across document indices.

### 3. Recommendation Engines (Collaborative vs. Content-Based)
> **Reasoning:** Collaborative filtering captures latent user preferences by computing similarity vectors across historical ratings, but suffers from cold-start failures when new items or users enter the system. Content-based filtering overcomes cold-start by comparing item feature vectors directly against a target user's preference profile, making a hybrid weighted model the standard production approach.

### 4. Local vs. Cloud Speech Synthesis (pyttsx3 vs. Cloud APIs)
> **Reasoning:** Local synthesis using `pyttsx3` provides zero operational cost, sub-10ms response latency, complete privacy, and offline resilience, though voice naturalness depends on host OS SAPI5/eSpeak fonts. Cloud APIs (such as ElevenLabs or AWS Polly) deliver superior neural expressiveness and emotion control but introduce per-character API costs, 100ms+ network latency, and external data privacy considerations.

### 5. Backend Exposing & FastAPI Integration
> **Reasoning:** Exposing heavy Python AI routines via FastAPI requires wrapping synchronous CPU-bound or I/O-bound code in non-blocking threads (`asyncio.to_thread`) to ensure the ASGI event loop remains responsive. Pydantic schemas enforce type safety and input sanitization at the API boundary, returning HTTP 400 status codes on malformed inputs before invoking model pipelines.

---

## 🧪 Running Tests & Local Server

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pytest Suite
```bash
python -m pytest
```

### 3. Start Development Server
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
