import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Dict, Any, List, Optional
import math
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

def download_model_if_needed():
    """Downloads the MediaPipe hand landmarker model file if not present locally."""
    if not os.path.exists(MODEL_PATH):
        print(f"[+] Downloading MediaPipe hand landmarker model from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[+] Model downloaded successfully and cached at:", MODEL_PATH)
        except Exception as e:
            print(f"[-] Error downloading model file: {str(e)}")
            raise RuntimeError(f"Failed to fetch HandLandmarker model weights: {str(e)}")

# Standard hand skeleton landmark connections mapping
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (9, 10), (10, 11), (11, 12),     # Middle
    (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)        # Knuckles
]

class MediaPipeService:
    def __init__(self):
        # Auto-download model weights on startup
        download_model_if_needed()
        
        # Configure and initialize MediaPipe Tasks HandLandmarker
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def validate_image_quality(self, img_bgr: np.ndarray) -> Tuple[bool, str]:
        """
        Validates image quality parameters:
        - Resolution (minimum 300x300 pixels)
        - Blurriness using Laplacian Variance (threshold ~60.0)
        - Exposure (mean grayscale brightness between 40 and 240)
        """
        if img_bgr is None or img_bgr.size == 0:
            return False, "Empty or invalid image data."

        h, w = img_bgr.shape[:2]
        if h < 300 or w < 300:
            return False, f"Image resolution too low ({w}x{h}). Please upload an image of at least 400x400 pixels."

        # Convert to grayscale for calculation
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Blurriness Check (Laplacian Variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 60.0:
            return False, f"Image is too blurry (blur score: {lap_var:.1f}). Please capture a sharp, stable image."

        # 2. Brightness Check
        mean_brightness = np.mean(gray)
        if mean_brightness < 40:
            return False, f"Image is too dark (brightness: {mean_brightness:.1f}). Please turn on lights or use flash."
        if mean_brightness > 240:
            return False, f"Image is overexposed (brightness: {mean_brightness:.1f}). Please avoid direct harsh light."

        return True, "Image quality validated."

    def detect_hand(self, img_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Runs MediaPipe HandLandmarker on the input image.
        Returns coordinate arrays and handedness properties.
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Wrap numpy array into MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Execute HandLandmarker inference
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.hand_landmarks:
            return None

        # Take first detected hand
        landmarks = detection_result.hand_landmarks[0]
        handedness = detection_result.handedness[0][0]
        
        hand_label = handedness.category_name  # "Left" or "Right"
        confidence = handedness.score

        # Extract landmarks as pixel coordinates
        h, w = img_bgr.shape[:2]
        landmarks_px = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_px.append((cx, cy, lm.z))

        return {
            "landmarks_px": landmarks_px,
            "hand_label": hand_label,
            "confidence": confidence,
            "raw_landmarks": landmarks
        }

    def align_and_crop_palm(self, img_bgr: np.ndarray, hand_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aligns the hand vertically (wrist at bottom, middle finger up) and crops the palm area.
        Returns:
            - Cropped palm image
            - Crop metadata (rotation angle, bounds, aligned landmarks)
        """
        h, w = img_bgr.shape[:2]
        landmarks = hand_data["landmarks_px"]

        # Landmark 0: Wrist
        # Landmark 9: MCP joint of middle finger (base of middle finger)
        wrist = np.array([landmarks[0][0], landmarks[0][1]])
        mcp_middle = np.array([landmarks[9][0], landmarks[9][1]])

        # Calculate rotation angle to align wrist (bottom) and MCP middle (top) vertically
        dy = mcp_middle[1] - wrist[1]
        dx = mcp_middle[0] - wrist[0]
        angle_rad = math.atan2(dy, dx)
        # We want the vector to point straight up (which is -90 degrees or -pi/2 in image coordinates)
        angle_deg = math.degrees(angle_rad) + 90.0

        # Center of rotation is the palm center (rough approximation using average of wrist, index base, pinky base)
        # Landmark 0: Wrist, Landmark 5: Index MCP, Landmark 17: Pinky MCP
        p0 = np.array(landmarks[0][:2])
        p5 = np.array(landmarks[5][:2])
        p17 = np.array(landmarks[17][:2])
        center = np.mean([p0, p5, p17], axis=0).astype(int)
        cx, cy = int(center[0]), int(center[1])

        # Get rotation matrix
        rot_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        
        # Determine size of rotated image to prevent cropping corners
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix to account for new center
        rot_matrix[0, 2] += (new_w / 2) - cx
        rot_matrix[1, 2] += (new_h / 2) - cy
        
        # Rotate image
        rotated_img = cv2.warpAffine(img_bgr, rot_matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Transform landmarks to new rotated coordinates
        rotated_landmarks = []
        for lm in landmarks:
            v = np.array([lm[0], lm[1], 1.0])
            rv = np.dot(rot_matrix, v)
            rotated_landmarks.append((int(rv[0]), int(rv[1]), lm[2]))

        # Define palm bounding box in the rotated coordinates
        rlm = rotated_landmarks
        
        x_pts = [rlm[0][0], rlm[1][0], rlm[2][0], rlm[5][0], rlm[9][0], rlm[13][0], rlm[17][0]]
        y_pts = [rlm[0][1], rlm[1][1], rlm[2][1], rlm[5][1], rlm[9][1], rlm[13][1], rlm[17][1]]
        
        min_x, max_x = min(x_pts), max(x_pts)
        min_y, max_y = min(y_pts), max(y_pts)
        
        # Add padding
        pad_x = int((max_x - min_x) * 0.15)
        pad_y = int((max_y - min_y) * 0.1)
        
        crop_x1 = max(0, min_x - pad_x)
        crop_x2 = min(new_w, max_x + pad_x)
        crop_y1 = max(0, min_y - pad_y)
        crop_y2 = min(new_h, max_y + pad_y)

        # Ensure valid crop box
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            crop_x1, crop_x2 = 0, new_w
            crop_y1, crop_y2 = 0, new_h

        cropped_palm = rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Shift rotated landmarks to be relative to the cropped image
        cropped_landmarks = []
        for rlm_pt in rlm:
            cx_pt = rlm_pt[0] - crop_x1
            cy_pt = rlm_pt[1] - crop_y1
            cropped_landmarks.append((cx_pt, cy_pt, rlm_pt[2]))

        crop_meta = {
            "angle_deg": angle_deg,
            "center": (cx, cy),
            "crop_box": (crop_x1, crop_y1, crop_x2, crop_y2),
            "rotated_landmarks": rotated_landmarks,
            "cropped_landmarks": cropped_landmarks,
            "hand_label": hand_data["hand_label"]
        }

        return cropped_palm, crop_meta

    def draw_landmarks(self, img_bgr: np.ndarray, hand_data: Dict[str, Any]) -> np.ndarray:
        """
        Draws skeleton and landmark circles on the original image for visual debugging.
        Uses manual OpenCV drawing to remain independent of legacy solutions packages.
        """
        img_copy = img_bgr.copy()
        landmarks = hand_data.get("landmarks_px", [])
        
        # Draw skeleton connector paths
        for start, end in HAND_CONNECTIONS:
            if start < len(landmarks) and end < len(landmarks):
                p1 = (landmarks[start][0], landmarks[start][1])
                p2 = (landmarks[end][0], landmarks[end][1])
                cv2.line(img_copy, p1, p2, (0, 0, 255), 2)  # Red skeleton line
                
        # Draw joint coordinates
        for lm in landmarks:
            cv2.circle(img_copy, (lm[0], lm[1]), 5, (0, 255, 0), -1)  # Green joint circle
            
        return img_copy
