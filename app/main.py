from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import search, recommendation, nlp, speech

app = FastAPI(
    title="AI Algorithms & Backend Integration Platform",
    description=(
        "Production-grade Python backend evaluating practical AI engineering concepts: "
        "A* Heuristic Search, Recommendation Systems (Collaborative/Content/Hybrid), "
        "Gensim NLP Vector Embeddings, and pyttsx3 Local Speech Synthesis."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(search.router, prefix="/api/v1")
app.include_router(recommendation.router, prefix="/api/v1")
app.include_router(nlp.router, prefix="/api/v1")
app.include_router(speech.router, prefix="/api/v1")

@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "service": "AI Algorithms & Backend Integration Engine",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "healthy"}
