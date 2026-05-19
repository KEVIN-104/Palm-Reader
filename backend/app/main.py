import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optimize OpenCV for low-memory container environments
import cv2
try:
    cv2.setNumThreads(1)
except Exception:
    pass

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from backend.app.core.config import settings
from backend.app.core.database import Base, engine
from backend.app.api.endpoints import router as api_router
from backend.app.services.mediapipe_service import MediaPipeService
from backend.app.services.opencv_service import OpenCVService
from backend.app.services.llm_service import LLMService

# Initialize SQLite database tables on startup
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize ML and helper services inside the spawned worker process to prevent
    # C++ thread context corruption across fork/spawn boundaries.
    app.state.mp_service = MediaPipeService()
    app.state.cv_service = OpenCVService()
    app.state.llm_service = LLMService()
    yield
    # Properly dispose detector threads/resources on shutdown
    if hasattr(app.state.mp_service, 'close'):
        app.state.mp_service.close()

app = FastAPI(
    title=settings.APP_NAME,
    description="Premium, completely free and local AI Palm Reading Platform using MediaPipe, OpenCV, and Ollama.",
    version="1.0.0",
    lifespan=lifespan
)


# CORS setup for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for scanned images
static_dir = os.path.join(str(settings.BASE_DIR), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API endpoints router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the premium single-page web app frontend."""
    template_path = os.path.join(str(settings.BASE_DIR), "backend", "app", "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html>
            <body style="font-family: sans-serif; background-color: #0d0e12; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh;">
                <h1>AuraPalm AI Platform</h1>
                <p>Welcome! Backend is running successfully. Frontend template is missing or compiling.</p>
                <p>Visit API documentation at <a href="/docs" style="color: #6366f1;">/docs</a></p>
            </body>
        </html>
        """
