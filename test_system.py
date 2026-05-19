import cv2
import sys
import asyncio
import json
from backend.app.services.mediapipe_service import MediaPipeService
from backend.app.services.opencv_service import OpenCVService
from backend.app.services.llm_service import LLMService

async def main():
    if len(sys.argv) < 2:
        print("==================================================")
        print(" AuraPalm AI - Offline Pipeline Testing Script ")
        print("==================================================")
        print("Error: Please provide a path to a palm image file.")
        print("Usage: .venv\\Scripts\\python test_system.py <path_to_image>")
        print("Example: .venv\\Scripts\\python test_system.py my_palm.jpg")
        print("==================================================")
        return

    img_path = sys.argv[1]
    print(f"\n[+] Loading image from: {img_path}")
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"[-] Error: Could not read image file at {img_path}. Verify path.")
        return

    # Initialize services
    mp_service = MediaPipeService()
    cv_service = OpenCVService()
    llm_service = LLMService()

    print("\n[1] Validating image quality parameters...")
    is_valid, msg = mp_service.validate_image_quality(img)
    print(f"    - Quality status: {is_valid}")
    print(f"    - Message details: {msg}")
    
    if not is_valid:
        print("[-] Exiting due to insufficient image quality.")
        return

    print("\n[2] Performing MediaPipe landmark detection...")
    hand_data = mp_service.detect_hand(img)
    if not hand_data:
        print("[-] Error: No hand detected in the image. Ensure the palm is fully visible and facing the camera.")
        return
    print(f"    - Handedness label: {hand_data['hand_label']}")
    print(f"    - MediaPipe tracking confidence: {hand_data['confidence']:.2f}")

    print("\n[3] Executing rotation alignment & palm cropping...")
    cropped_palm, crop_meta = mp_service.align_and_crop_palm(img, hand_data)
    print(f"    - Cropped palm resolution: {cropped_palm.shape[1]}x{cropped_palm.shape[0]} px")

    print("\n[4] Running OpenCV geometric crease & mount calculations...")
    cv_data = cv_service.process_palm(cropped_palm, crop_meta["cropped_landmarks"], crop_meta["hand_label"])
    print(f"    - Palm Elemental shape: {cv_data['palm_shape']}")
    print(f"    - Pipeline confidence score: {cv_data['confidence_score']}%")
    print(f"    - Life line completeness: {cv_data['life_line']['length']}% (depth: {cv_data['life_line']['depth']}%)")
    print(f"    - Heart line completeness: {cv_data['heart_line']['length']}% (depth: {cv_data['heart_line']['depth']}%)")
    print(f"    - Head line completeness: {cv_data['head_line']['length']}% (depth: {cv_data['head_line']['depth']}%)")
    
    prominent = [m for m, desc in cv_data['mounts'].items() if 'Prominent' in desc]
    print(f"    - Prominent fleshy mounts: {', '.join(prominent) if prominent else 'None'}")

    print("\n[5] Simulating local AI interpretation (Ollama with fallback)...")
    _, crop_buf = cv2.imencode('.jpg', cropped_palm)
    crop_bytes = crop_buf.tobytes()
    
    ai_reading = await llm_service.get_reading(cv_data, crop_bytes)
    print(f"    - AI reasoning source: {ai_reading['ai_source']}")
    print("    - Personality overview extract:")
    print(f"      \"{ai_reading['personality']['analysis'][:160]}...\"")
    print(f"    - Career Vocation Score: {ai_reading['career']['score']}/100")
    print(f"    - Relationship Harmony Score: {ai_reading['relationships']['score']}/100")

    # Save output visualization
    out_path = "test_result.jpg"
    cv2.imwrite(out_path, cv_data["overlay_img"])
    print(f"\n[+] Success! Diagnostic overlay generated and saved as: {out_path}")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
