from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from datetime import datetime
from backend.app.core.database import Base

class PalmReading(Base):
    __tablename__ = "palm_readings"

    id = Column(Integer, primary_key=True, index=True)
    client_ip = Column(String, nullable=True)
    original_image_path = Column(String, nullable=False)
    analyzed_image_path = Column(String, nullable=True)
    confidence_score = Column(Float, default=0.0)
    
    # Physical Hand traits (extracted via MediaPipe + OpenCV)
    hand_type = Column(String, nullable=True)  # Left or Right
    palm_shape = Column(String, nullable=True)  # Square, Spatulate, Conic, Philosophical, Psychic, Elementary
    
    # Core lines details (JSON object: {"length": float, "curvature": float, "depth": float, "description": str})
    life_line_data = Column(JSON, nullable=True)
    heart_line_data = Column(JSON, nullable=True)
    head_line_data = Column(JSON, nullable=True)
    fate_line_data = Column(JSON, nullable=True)
    
    # Other features
    mounts_data = Column(JSON, nullable=True)     # Active mounts and their score
    finger_lengths = Column(JSON, nullable=True)   # Relative finger lengths and ratios
    
    # AI Interpretation response (JSON object: {"personality": str, "career": str, "relationships": str, "strengths": list, "energy_profile": dict})
    ai_analysis = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
