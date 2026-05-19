from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

class LineDetail(BaseModel):
    length: float       # 0 to 100
    curvature: float    # 0 to 100
    depth: float        # 0 to 100
    clarity: str        # e.g., "Deep & Clear", "Faint", "Chained", "Broken"
    meaning: str        # Palmistry interpretation

class SectionDetail(BaseModel):
    analysis: str
    key_takeaways: List[str]
    score: Optional[int] = None

class PalmReadingResponse(BaseModel):
    id: int
    hand_type: str      # Left or Right
    palm_shape: str     # Square, Spatulate, Conic, etc.
    confidence_score: float
    
    # Standard lines (matching user's request format)
    life_line: LineDetail
    heart_line: LineDetail
    head_line: LineDetail
    fate_line: Optional[LineDetail] = None
    
    # Extra segments requested/suggested
    mounts: Dict[str, str]        # Name: Intensity description
    fingers: Dict[str, str]       # Finger name: description
    
    # Interpretations (career, relationships, personality)
    personality: SectionDetail
    career: SectionDetail
    relationships: SectionDetail
    energy_profile: Dict[str, int]  # Vitality, Emotion, Intellect, Ambition (0-100)
    
    original_image: str
    analyzed_image: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class HistoryResponse(BaseModel):
    id: int
    hand_type: str
    palm_shape: str
    confidence_score: float
    created_at: datetime
    original_image: str
    analyzed_image: Optional[str] = None
    
    class Config:
        from_attributes = True
