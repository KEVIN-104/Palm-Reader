from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request, status
from sqlalchemy.orm import Session
import cv2
import numpy as np
import os
import uuid
from typing import List

from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.models.reading import PalmReading
from backend.app.schemas.reading import PalmReadingResponse, HistoryResponse, LineDetail, SectionDetail
from backend.app.services.mediapipe_service import MediaPipeService
from backend.app.services.opencv_service import OpenCVService
from backend.app.services.llm_service import LLMService

router = APIRouter()

# Services are lazy-loaded via request.app.state during requests


def map_db_to_response(db_reading: PalmReading) -> PalmReadingResponse:
    """Helper to transform database model to Pydantic Response schema."""
    ai = db_reading.ai_analysis or {}
    
    # Format finger descriptions
    fl = db_reading.finger_lengths or {}
    fingers_formatted = {
        "thumb": f"Relative length: {fl.get('thumb', 0.0)} px",
        "index": f"Relative length: {fl.get('index', 0.0)} px",
        "middle": f"Relative length: {fl.get('middle', 0.0)} px",
        "ring": f"Relative length: {fl.get('ring', 0.0)} px",
        "pinky": f"Relative length: {fl.get('pinky', 0.0)} px"
    }
    
    # Handle image paths to relative web URLs
    orig_url = f"/static/uploads/{os.path.basename(db_reading.original_image_path)}"
    anal_url = f"/static/analyzed/{os.path.basename(db_reading.analyzed_image_path)}" if db_reading.analyzed_image_path else None

    return PalmReadingResponse(
        id=db_reading.id,
        hand_type=db_reading.hand_type or "Unknown",
        palm_shape=db_reading.palm_shape or "Unknown",
        confidence_score=db_reading.confidence_score,
        life_line=LineDetail(
            length=db_reading.life_line_data.get("length", 0.0),
            curvature=db_reading.life_line_data.get("curvature", 0.0),
            depth=db_reading.life_line_data.get("depth", 0.0),
            clarity=db_reading.life_line_data.get("clarity", "Unknown"),
            meaning=ai.get("life_line_meaning", "Not analyzed")
        ),
        heart_line=LineDetail(
            length=db_reading.heart_line_data.get("length", 0.0),
            curvature=db_reading.heart_line_data.get("curvature", 0.0),
            depth=db_reading.heart_line_data.get("depth", 0.0),
            clarity=db_reading.heart_line_data.get("clarity", "Unknown"),
            meaning=ai.get("heart_line_meaning", "Not analyzed")
        ),
        head_line=LineDetail(
            length=db_reading.head_line_data.get("length", 0.0),
            curvature=db_reading.head_line_data.get("curvature", 0.0),
            depth=db_reading.head_line_data.get("depth", 0.0),
            clarity=db_reading.head_line_data.get("clarity", "Unknown"),
            meaning=ai.get("head_line_meaning", "Not analyzed")
        ),
        fate_line=LineDetail(
            length=db_reading.fate_line_data.get("length", 0.0),
            curvature=db_reading.fate_line_data.get("curvature", 0.0),
            depth=db_reading.fate_line_data.get("depth", 0.0),
            clarity=db_reading.fate_line_data.get("clarity", "Unknown"),
            meaning=ai.get("fate_line_meaning", "Not analyzed")
        ) if db_reading.fate_line_data else None,
        mounts=db_reading.mounts_data or {},
        fingers=fingers_formatted,
        personality=SectionDetail(
            analysis=ai.get("personality", {}).get("analysis", "No analysis"),
            key_takeaways=ai.get("personality", {}).get("key_takeaways", [])
        ),
        career=SectionDetail(
            analysis=ai.get("career", {}).get("analysis", "No analysis"),
            key_takeaways=ai.get("career", {}).get("key_takeaways", []),
            score=ai.get("career", {}).get("score", 50)
        ),
        relationships=SectionDetail(
            analysis=ai.get("relationships", {}).get("analysis", "No analysis"),
            key_takeaways=ai.get("relationships", {}).get("key_takeaways", []),
            score=ai.get("relationships", {}).get("score", 50)
        ),
        energy_profile=ai.get("energy_profile", {"vitality": 50, "emotion": 50, "intellect": 50, "ambition": 50}),
        original_image=orig_url,
        analyzed_image=anal_url,
        created_at=db_reading.created_at
    )


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Simple API status checks."""
    return {"status": "healthy", "service": settings.APP_NAME}


@router.post("/analyze", response_model=PalmReadingResponse, status_code=status.HTTP_201_CREATED)
async def analyze_palm_image(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Main endpoint for hand palm analysis.
    Validates quality, runs MediaPipe & OpenCV, gets AI interpretations, and stores locally.
    """
    # 1. Retrieve lazy-loaded services from app state
    mp_service = request.app.state.mp_service
    cv_service = request.app.state.cv_service
    llm_service = request.app.state.llm_service

    file_bytes = None
    img_bgr = None
    cropped_palm = None
    cv_data = None
    crop_bytes = None

    try:
        # 1. Read file bytes
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file upload.")

        # Decode image with OpenCV
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        # 2. Quality checking (blur, brightness)
        is_valid, msg = mp_service.validate_image_quality(img_bgr)
        if not is_valid:
            raise HTTPException(status_code=400, detail=msg)

        # 3. MediaPipe Landmark detection
        hand_data = mp_service.detect_hand(img_bgr)
        if not hand_data:
            raise HTTPException(
                status_code=400, 
                detail="No hand detected. Please make sure your hand is fully visible, well lit, and centered in the frame."
            )

        # 4. Crop and rotate palm
        cropped_palm, crop_meta = mp_service.align_and_crop_palm(img_bgr, hand_data)
        if cropped_palm is None or cropped_palm.size == 0:
            raise HTTPException(status_code=500, detail="Palm alignment failed.")

        # 5. Extract features using OpenCV
        cv_data = cv_service.process_palm(cropped_palm, crop_meta["cropped_landmarks"], crop_meta["hand_label"])
        
        # 6. Save image files
        file_id = str(uuid.uuid4())
        orig_filename = f"{file_id}_original.jpg"
        anal_filename = f"{file_id}_analyzed.jpg"
        
        orig_path = os.path.join(settings.UPLOAD_DIR, orig_filename)
        anal_path = os.path.join(settings.ANALYZED_DIR, anal_filename)
        
        cv2.imwrite(orig_path, img_bgr)
        cv2.imwrite(anal_path, cv_data["overlay_img"])

        # Encode cropped palm for vision model
        _, crop_buf = cv2.imencode('.jpg', cropped_palm)
        crop_bytes = crop_buf.tobytes()

        # 7. Get local AI interpretation (using LLM service)
        ai_analysis = await llm_service.get_reading(cv_data, crop_bytes)

        # 8. Save to Database
        db_reading = PalmReading(
            client_ip=request.client.host if request.client else "127.0.0.1",
            original_image_path=orig_path,
            analyzed_image_path=anal_path,
            confidence_score=cv_data["confidence_score"],
            hand_type=crop_meta["hand_label"],
            palm_shape=cv_data["palm_shape"],
            life_line_data={
                "length": cv_data["life_line"]["length"],
                "depth": cv_data["life_line"]["depth"],
                "curvature": cv_data["life_line"]["curvature"],
                "clarity": cv_data["life_line"]["clarity"]
            },
            heart_line_data={
                "length": cv_data["heart_line"]["length"],
                "depth": cv_data["heart_line"]["depth"],
                "curvature": cv_data["heart_line"]["curvature"],
                "clarity": cv_data["heart_line"]["clarity"]
            },
            head_line_data={
                "length": cv_data["head_line"]["length"],
                "depth": cv_data["head_line"]["depth"],
                "curvature": cv_data["head_line"]["curvature"],
                "clarity": cv_data["head_line"]["clarity"]
            },
            fate_line_data={
                "length": cv_data["fate_line"]["length"],
                "depth": cv_data["fate_line"]["depth"],
                "curvature": cv_data["fate_line"]["curvature"],
                "clarity": cv_data["fate_line"]["clarity"]
            },
            mounts_data=cv_data["mounts"],
            finger_lengths=cv_data["finger_lengths"],
            ai_analysis=ai_analysis
        )

        db.add(db_reading)
        db.commit()
        db.refresh(db_reading)

        return map_db_to_response(db_reading)

    finally:
        # Close upload spool file to release descriptor and buffer memory
        try:
            await file.close()
        except Exception:
            pass
        # Clean up temporary large variables to assist garbage collection
        del file_bytes
        del img_bgr
        del cropped_palm
        del cv_data
        del crop_bytes
        
        import gc
        gc.collect()



@router.get("/history", response_model=List[HistoryResponse])
def get_reading_history(db: Session = Depends(get_db)):
    """Fetches a list of past scans sorted by date (newest first)."""
    readings = db.query(PalmReading).order_by(PalmReading.created_at.desc()).all()
    
    # Map paths to relative URLs
    history = []
    for r in readings:
        orig_url = f"/static/uploads/{os.path.basename(r.original_image_path)}"
        anal_url = f"/static/analyzed/{os.path.basename(r.analyzed_image_path)}" if r.analyzed_image_path else None
        history.append(HistoryResponse(
            id=r.id,
            hand_type=r.hand_type or "Unknown",
            palm_shape=r.palm_shape or "Unknown",
            confidence_score=r.confidence_score,
            created_at=r.created_at,
            original_image=orig_url,
            analyzed_image=anal_url
        ))
    return history


@router.get("/history/{reading_id}", response_model=PalmReadingResponse)
def get_single_reading(reading_id: int, db: Session = Depends(get_db)):
    """Fetches full details of a specific scan session."""
    reading = db.query(PalmReading).filter(PalmReading.id == reading_id).first()
    if not reading:
        raise HTTPException(status_code=404, detail="Palm reading record not found.")
    return map_db_to_response(reading)


@router.delete("/history/{reading_id}", status_code=status.HTTP_200_OK)
def delete_single_reading(reading_id: int, db: Session = Depends(get_db)):
    """Removes a scan session from history database and deletes its local file assets."""
    reading = db.query(PalmReading).filter(PalmReading.id == reading_id).first()
    if not reading:
        raise HTTPException(status_code=404, detail="Palm reading record not found.")
    
    # Delete file assets
    try:
        if os.path.exists(reading.original_image_path):
            os.remove(reading.original_image_path)
        if reading.analyzed_image_path and os.path.exists(reading.analyzed_image_path):
            os.remove(reading.analyzed_image_path)
    except Exception as e:
        logger.error(f"Error removing files during database delete: {str(e)}")

    db.delete(reading)
    db.commit()
    return {"detail": "Palm reading record deleted successfully."}
