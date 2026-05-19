import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "AuraPalm AI"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    UPLOAD_DIR: str = os.path.join(str(BASE_DIR), "static", "uploads")
    ANALYZED_DIR: str = os.path.join(str(BASE_DIR), "static", "analyzed")
    
    # Database
    DATABASE_URL: str = f"sqlite:///{os.path.join(str(BASE_DIR), 'static', 'palm_readings.db')}"
    
    # Local AI (Ollama / Local Vision Server)
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # Fallback text model
    VISION_MODEL: str = os.getenv("VISION_MODEL", "qwen2-vl:latest")  # Local vision model (e.g. qwen2-vl, llava, minicpm-v)
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.ANALYZED_DIR, exist_ok=True)
