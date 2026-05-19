import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import math
import os

class OpenCVService:
    def __init__(self):
        pass

    def process_palm(self, cropped_palm: np.ndarray, landmarks: List[Tuple[int, int, float]], hand_label: str) -> Dict[str, Any]:
        """
        Main OpenCV processing pipeline:
        1. Background removal.
        2. Lighting normalization (CLAHE).
        3. Line enhancement (Blackhat filter).
        4. Line tracking and parameter extraction.
        5. Palm shape classification.
        6. Mount analysis.
        7. Generate visualized output image.
        """
        h, w = cropped_palm.shape[:2]
        
        # 1. Background Masking (using landmark hull)
        mask = np.zeros((h, w), dtype=np.uint8)
        hull_pts = np.array([[lm[0], lm[1]] for lm in landmarks], dtype=np.int32)
        cv2.drawContours(mask, [cv2.convexHull(hull_pts)], -1, 255, -1)
        
        # Apply mask to palm image
        palm_masked = cv2.bitwise_and(cropped_palm, cropped_palm, mask=mask)
        
        # 2. Convert to Grayscale & Normalize Lighting
        gray = cv2.cvtColor(palm_masked, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # 3. Enhance Palm Creases using Blackhat filter
        # Blackhat highlights elements darker than their surroundings (the wrinkles/creases)
        kernel_size = max(9, int(min(h, w) * 0.035))  # Dynamic kernel size based on palm size
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold to get binary line map
        _, thresh = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
        # Apply bilateral filtering to reduce noise
        thresh_blur = cv2.bilateralFilter(thresh, 5, 50, 50)
        
        # 4. Finger Lengths and Ratios
        finger_metrics = self._analyze_fingers(landmarks)
        
        # 5. Palm Shape Classification
        palm_shape, palm_metrics = self._classify_palm_shape(landmarks, finger_metrics)
        
        # 6. Trace Core Palm Lines
        life_line = self._trace_life_line(gray_enhanced, thresh_blur, landmarks, hand_label)
        heart_line = self._trace_heart_line(gray_enhanced, thresh_blur, landmarks, hand_label)
        head_line = self._trace_head_line(gray_enhanced, thresh_blur, landmarks, hand_label)
        fate_line = self._trace_fate_line(gray_enhanced, thresh_blur, landmarks, hand_label)
        
        # 7. Analyze Mount Prominence
        mounts = self._analyze_mounts(gray_enhanced, landmarks, hand_label)
        
        # 8. Generate Visual Overlay
        overlay_img = self._generate_visual_overlay(cropped_palm, landmarks, life_line, heart_line, head_line, fate_line, mounts, palm_shape)
        
        # Calculate overall confidence score based on landmark resolution and line detection quality
        confidence = self._calculate_confidence(landmarks, life_line, heart_line, head_line)

        return {
            "palm_shape": palm_shape,
            "palm_metrics": palm_metrics,
            "finger_lengths": finger_metrics,
            "life_line": life_line,
            "heart_line": heart_line,
            "head_line": head_line,
            "fate_line": fate_line,
            "mounts": mounts,
            "confidence_score": confidence,
            "processed_gray": gray_enhanced,
            "line_mask": thresh_blur,
            "overlay_img": overlay_img
        }

    def _analyze_fingers(self, landmarks: List[Tuple[int, int, float]]) -> Dict[str, float]:
        """
        Calculates physical length of each finger based on landmarks:
        - Thumb: Base (1) to Tip (4)
        - Index: Base (5) to Tip (8)
        - Middle: Base (9) to Tip (12)
        - Ring: Base (13) to Tip (16)
        - Pinky: Base (17) to Tip (20)
        """
        def dist(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        thumb = dist(landmarks[1], landmarks[4])
        index = dist(landmarks[5], landmarks[8])
        middle = dist(landmarks[9], landmarks[12])
        ring = dist(landmarks[13], landmarks[16])
        pinky = dist(landmarks[17], landmarks[20])
        
        return {
            "thumb": thumb,
            "index": index,
            "middle": middle,
            "ring": ring,
            "pinky": pinky
        }

    def _classify_palm_shape(self, landmarks: List[Tuple[int, int, float]], fingers: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """
        Classifies hand shape according to elements (Earth, Air, Fire, Water) based on palm dimension and middle finger length:
        - Earth: Square palm, short fingers
        - Air: Square palm, long fingers
        - Water: Rectangular palm, long fingers
        - Fire: Rectangular palm, short fingers
        """
        def dist(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        # Palm width: Index base (5) to Pinky base (17)
        palm_width = dist(landmarks[5], landmarks[17])
        # Palm height: Wrist (0) to Middle base (9)
        palm_height = dist(landmarks[0], landmarks[9])
        
        ratio_w_h = palm_width / (palm_height if palm_height > 0 else 1)
        
        # Middle finger compared to palm height
        middle_finger = fingers["middle"]
        finger_palm_ratio = middle_finger / (palm_height if palm_height > 0 else 1)
        
        # Classification criteria
        is_square_palm = ratio_w_h >= 0.85
        is_long_fingers = finger_palm_ratio >= 0.8
        
        if is_square_palm and not is_long_fingers:
            shape = "Earth Hand (Practical, Grounded, Reliable)"
        elif is_square_palm and is_long_fingers:
            shape = "Air Hand (Intellectual, Communicative, Analytical)"
        elif not is_square_palm and is_long_fingers:
            shape = "Water Hand (Intuitive, Sensitive, Creative)"
        else:
            shape = "Fire Hand (Passionate, Dynamic, Charismatic)"
            
        metrics = {
            "palm_width": round(palm_width, 1),
            "palm_height": round(palm_height, 1),
            "width_to_height_ratio": round(ratio_w_h, 2),
            "finger_to_palm_ratio": round(finger_palm_ratio, 2)
        }
        
        return shape, metrics

    def _get_path_metrics(self, gray: np.ndarray, mask: np.ndarray, path_points: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Analyzes pixel values along a proposed coordinates path to compute:
        - Length (ratio of active line pixels on the mask to total path points)
        - Curvature (variance of direction vectors along the path)
        - Depth (contrast of gray values in the original vs neighbors)
        """
        h, w = gray.shape
        active_points = 0
        total_points = len(path_points)
        gray_values = []
        contrast_values = []

        if total_points == 0:
            return {"length": 0.0, "curvature": 0.0, "depth": 0.0, "points": []}

        valid_points = []
        for (px, py) in path_points:
            if 0 <= px < w and 0 <= py < h:
                valid_points.append((px, py))
                # Check line mask
                if mask[py, px] > 0:
                    active_points += 1
                
                # Check original contrast: compare point to immediate neighborhood
                val = int(gray[py, px])
                gray_values.append(val)
                
                # Neighbors
                neighbors = []
                for dx, dy in [(-2,0), (2,0), (0,-2), (0,2)]:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        neighbors.append(int(gray[ny, nx]))
                if neighbors:
                    # Line is darker than surrounding skin
                    contrast = max(0, np.mean(neighbors) - val)
                    contrast_values.append(contrast)

        # Length score: percentage of path containing lines
        length_score = (active_points / total_points) * 100.0 if total_points > 0 else 0
        # Depth score: normalized contrast
        depth_score = np.mean(contrast_values) * 5.0 if contrast_values else 0.0
        depth_score = min(100.0, max(0.0, depth_score))
        
        # Curvature score: deviation from a straight line
        curvature = 0.0
        if len(valid_points) > 4:
            # Fit polynomial or calculate standard deviation of angles between segments
            angles = []
            for i in range(1, len(valid_points) - 1):
                p1 = valid_points[i-1]
                p2 = valid_points[i]
                p3 = valid_points[i+1]
                
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                d1 = math.hypot(*v1)
                d2 = math.hypot(*v2)
                if d1 > 0 and d2 > 0:
                    dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (d1 * d2)
                    dot = max(-1.0, min(1.0, dot))
                    angle = math.acos(dot)
                    angles.append(angle)
            if angles:
                # Sum of deviations
                curvature = sum(angles) * 15.0
                curvature = min(100.0, curvature)

        # Normalize metrics to keep them clean
        length_score = round(min(100.0, max(15.0, length_score * 1.5)), 1)
        depth_score = round(min(100.0, max(20.0, depth_score)), 1)
        curvature = round(min(100.0, max(5.0, curvature)), 1)

        # Assign clarity description based on metrics
        if depth_score > 60:
            clarity = "Deep & Clear"
        elif depth_score > 40:
            clarity = "Moderate & Well-Defined"
        elif length_score < 30:
            clarity = "Broken / Segmented"
        else:
            clarity = "Faint / Superficial"

        return {
            "length": length_score,
            "depth": depth_score,
            "curvature": curvature,
            "clarity": clarity,
            "points": valid_points
        }

    def _trace_life_line(self, gray: np.ndarray, mask: np.ndarray, landmarks: List[Tuple[int, int, float]], hand_label: str) -> Dict[str, Any]:
        """
        Life line: Curves around the base of the thumb (Mount of Venus).
        Starts between index MCP (5) and thumb base (2), swoops down and around thumb base towards wrist (0).
        """
        p5 = np.array(landmarks[5][:2])
        p2 = np.array(landmarks[2][:2])
        p0 = np.array(landmarks[0][:2])
        
        # Origin of life line (between index base and thumb base)
        start_pt = (p5 + p2) / 2
        
        # Bending anchor: outer curve of the thumb base (around landmark 3 or 4)
        anchor_pt = np.array(landmarks[3][:2])
        if hand_label == "Right":
            # Adjust anchor outwards to capture Venus mount curvature
            anchor_pt[0] = anchor_pt[0] - abs(p5[0] - p2[0]) * 0.3
        else:
            anchor_pt[0] = anchor_pt[0] + abs(p5[0] - p2[0]) * 0.3
            
        end_pt = p0 + (anchor_pt - p0) * 0.2 # Ends near wrist base

        # Generate quadratic Bezier curve points
        path_pts = []
        for t in np.linspace(0, 1, 50):
            # Bezier formula
            pt = (1-t)**2 * start_pt + 2*(1-t)*t * anchor_pt + t**2 * end_pt
            path_pts.append((int(pt[0]), int(pt[1])))
            
        return self._get_path_metrics(gray, mask, path_pts)

    def _trace_heart_line(self, gray: np.ndarray, mask: np.ndarray, landmarks: List[Tuple[int, int, float]], hand_label: str) -> Dict[str, Any]:
        """
        Heart line: Upper horizontal line.
        Starts below Pinky MCP (17) on the outer edge, runs across towards Index MCP (5) or Middle MCP (9).
        """
        p17 = np.array(landmarks[17][:2])
        p5 = np.array(landmarks[5][:2])
        p9 = np.array(landmarks[9][:2])
        
        # Start: Outer edge below pinky
        start_x_offset = int(abs(p17[0] - p5[0]) * 0.1)
        if hand_label == "Right":
            start_pt = p17 + np.array([start_x_offset, int(p17[1]*0.15)])
            end_pt = p9 + np.array([-int(abs(p17[0]-p5[0])*0.1), int(p9[1]*0.1)])
            anchor_pt = (start_pt + end_pt) / 2 + np.array([0, int(p17[1]*0.05)])
        else:
            start_pt = p17 + np.array([-start_x_offset, int(p17[1]*0.15)])
            end_pt = p9 + np.array([int(abs(p17[0]-p5[0])*0.1), int(p9[1]*0.1)])
            anchor_pt = (start_pt + end_pt) / 2 + np.array([0, int(p17[1]*0.05)])

        path_pts = []
        for t in np.linspace(0, 1, 50):
            pt = (1-t)**2 * start_pt + 2*(1-t)*t * anchor_pt + t**2 * end_pt
            path_pts.append((int(pt[0]), int(pt[1])))
            
        return self._get_path_metrics(gray, mask, path_pts)

    def _trace_head_line(self, gray: np.ndarray, mask: np.ndarray, landmarks: List[Tuple[int, int, float]], hand_label: str) -> Dict[str, Any]:
        """
        Head line: Middle horizontal/diagonal line.
        Starts at/near the life line origin (between 2 and 5) and runs across the middle of the palm.
        """
        p5 = np.array(landmarks[5][:2])
        p2 = np.array(landmarks[2][:2])
        p17 = np.array(landmarks[17][:2])
        p0 = np.array(landmarks[0][:2])
        
        # Starts near life line origin
        start_pt = (p5 + p2) / 2
        
        # End: Runs towards outer palm (Luna mount area, above wrist, below pinky)
        # Midpoint of wrist and pinky base gives outer palm level
        luna_level = (p0 + p17) / 2
        
        if hand_label == "Right":
            end_pt = luna_level - np.array([int(abs(p17[0]-p5[0])*0.1), 0])
        else:
            end_pt = luna_level + np.array([int(abs(p17[0]-p5[0])*0.1), 0])
            
        anchor_pt = (start_pt + end_pt) / 2 + np.array([0, int(abs(p17[1] - p5[1])*0.1)])

        path_pts = []
        for t in np.linspace(0, 1, 50):
            pt = (1-t)**2 * start_pt + 2*(1-t)*t * anchor_pt + t**2 * end_pt
            path_pts.append((int(pt[0]), int(pt[1])))
            
        return self._get_path_metrics(gray, mask, path_pts)

    def _trace_fate_line(self, gray: np.ndarray, mask: np.ndarray, landmarks: List[Tuple[int, int, float]], hand_label: str) -> Dict[str, Any]:
        """
        Fate line: Vertical line running from wrist (0) up towards Middle MCP (9).
        Often faint or missing entirely (which is also common in palmistry).
        """
        p0 = np.array(landmarks[0][:2])
        p9 = np.array(landmarks[9][:2])
        
        start_pt = p0 - np.array([0, int(p0[1]*0.05)])
        end_pt = p9 + np.array([0, int(p9[1]*0.15)])
        
        # Path is a soft curve or line
        anchor_pt = (start_pt + end_pt) / 2
        if hand_label == "Right":
            anchor_pt[0] += int(abs(p0[0] - p9[0]) * 0.2)
        else:
            anchor_pt[0] -= int(abs(p0[0] - p9[0]) * 0.2)

        path_pts = []
        for t in np.linspace(0, 1, 40):
            pt = (1-t)**2 * start_pt + 2*(1-t)*t * anchor_pt + t**2 * end_pt
            path_pts.append((int(pt[0]), int(pt[1])))
            
        metrics = self._get_path_metrics(gray, mask, path_pts)
        
        # The fate line is often very faint, so if depth is extremely low, we flag it as "Absent or Very Faint"
        if metrics["depth"] < 25 or metrics["length"] < 25:
            metrics["clarity"] = "Absent / Faint"
            metrics["length"] = round(metrics["length"] * 0.5, 1)
            metrics["depth"] = round(metrics["depth"] * 0.5, 1)
            
        return metrics

    def _analyze_mounts(self, gray_enhanced: np.ndarray, landmarks: List[Tuple[int, int, float]], hand_label: str) -> Dict[str, str]:
        """
        Analyzes fleshy regions (Mounts) by inspecting local texture brightness and gradient variance.
        Fleshy, healthy mounts are usually bright and have smooth gradients.
        - Venus: Base of thumb (around landmarks 2, 3, 4)
        - Jupiter: Under Index (landmark 5)
        - Saturn: Under Middle (landmark 9)
        - Sun/Apollo: Under Ring (landmark 13)
        - Mercury: Under Pinky (landmark 17)
        - Luna/Moon: Lower outer palm (opposite to Venus)
        """
        h, w = gray_enhanced.shape
        
        def get_region_variance(cx, cy, radius=20) -> float:
            if cx - radius < 0 or cx + radius >= w or cy - radius < 0 or cy + radius >= h:
                return 0.0
            roi = gray_enhanced[cy-radius:cy+radius, cx-radius:cx+radius]
            # Calculate gradient variance (textures)
            sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            # Higher variance = more textured, lower = smooth/fleshy
            mean_intensity = np.mean(roi)
            # Combine mean intensity and smoothness
            score = mean_intensity * 0.6 + (100.0 - np.std(grad_mag)) * 0.4
            return float(score)

        # Coordinates for mounts
        # Venus: Center of thumb base area (Landmark 2/3 midpoint)
        v_cx = int((landmarks[2][0] + landmarks[3][0]) / 2)
        v_cy = int((landmarks[2][1] + landmarks[3][1]) / 2)
        
        # Jupiter: Just below Index base (5)
        j_cx = landmarks[5][0]
        j_cy = landmarks[5][1] + int(abs(landmarks[5][1] - landmarks[0][1]) * 0.1)
        
        # Saturn: Below Middle base (9)
        s_cx = landmarks[9][0]
        s_cy = landmarks[9][1] + int(abs(landmarks[9][1] - landmarks[0][1]) * 0.1)
        
        # Sun: Below Ring base (13)
        su_cx = landmarks[13][0]
        su_cy = landmarks[13][1] + int(abs(landmarks[13][1] - landmarks[0][1]) * 0.1)
        
        # Mercury: Below Pinky base (17)
        m_cx = landmarks[17][0]
        m_cy = landmarks[17][1] + int(abs(landmarks[17][1] - landmarks[0][1]) * 0.1)
        
        # Luna: Opposite to Venus. Outer bottom palm.
        # Mirror of Venus relative to the center vertical axis of the palm
        mcp_mid = landmarks[9] # Center axis top
        wrist = landmarks[0]   # Center axis bottom
        
        # Midpoint of wrist and pinky base gives Luna zone
        lu_cx = int((landmarks[0][0] + landmarks[17][0]) / 2)
        lu_cy = int((landmarks[0][1] + landmarks[17][1]) / 2)
        if hand_label == "Right":
            lu_cx += int(w * 0.05)
        else:
            lu_cx -= int(w * 0.05)

        # Get scores
        scores = {
            "Jupiter": get_region_variance(j_cx, j_cy, int(w*0.06)),
            "Saturn": get_region_variance(s_cx, s_cy, int(w*0.06)),
            "Apollo (Sun)": get_region_variance(su_cx, su_cy, int(w*0.06)),
            "Mercury": get_region_variance(m_cx, m_cy, int(w*0.06)),
            "Venus": get_region_variance(v_cx, v_cy, int(w*0.08)),
            "Luna (Moon)": get_region_variance(lu_cx, lu_cy, int(w*0.08))
        }

        # Convert numerical scores to qualitative descriptors
        # Higher score = prominent/well-developed, lower = flat/underdeveloped
        sorted_scores = sorted(scores.values())
        low_thresh = sorted_scores[1]  # Bottom 33%
        high_thresh = sorted_scores[4] # Top 33%

        results = {}
        for mount, val in scores.items():
            if val >= high_thresh:
                results[mount] = "Prominent & Fleshy (High energy, active traits)"
            elif val <= low_thresh:
                results[mount] = "Flat / Quiet (Subtle presence, receptive traits)"
            else:
                results[mount] = "Balanced (Healthy, stable integration)"

        return results

    def _generate_visual_overlay(self, cropped_palm: np.ndarray, landmarks: List[Tuple[int, int, float]], 
                                 life: Dict[str, Any], heart: Dict[str, Any], head: Dict[str, Any], fate: Dict[str, Any],
                                 mounts: Dict[str, str], palm_shape: str) -> np.ndarray:
        """
        Draws colored overlay showing:
        - Hand skeleton in semi-transparent white/green.
        - Traced Life Line (Green), Heart Line (Red), Head Line (Blue), Fate Line (Purple).
        - Mount circles for visual verification.
        """
        overlay = cropped_palm.copy()
        
        # 1. Draw Skeleton Lines
        mp_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (9, 10), (10, 11), (11, 12),     # Middle
            (13, 14), (14, 15), (15, 16),    # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)        # Knuckles
        ]
        
        for start, end in mp_connections:
            if start < len(landmarks) and end < len(landmarks):
                p1 = (landmarks[start][0], landmarks[start][1])
                p2 = (landmarks[end][0], landmarks[end][1])
                cv2.line(overlay, p1, p2, (220, 220, 220), 2)
                
        # 2. Draw Landmark joints
        for lm in landmarks:
            cv2.circle(overlay, (lm[0], lm[1]), 4, (0, 255, 0), -1)

        # 3. Draw Palmistry Lines
        def draw_traced_line(pts, color, thickness=3):
            if not pts:
                return
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i+1], color, thickness, lineType=cv2.LINE_AA)

        # Draw main lines
        draw_traced_line(life.get("points", []), (0, 220, 0), 4)      # Green for Life
        draw_traced_line(heart.get("points", []), (50, 50, 255), 4)   # Red for Heart
        draw_traced_line(head.get("points", []), (255, 150, 0), 4)    # Blue for Head
        
        if fate.get("clarity") != "Absent / Faint":
            draw_traced_line(fate.get("points", []), (200, 50, 200), 3) # Purple for Fate

        # Add text metadata overlay in corner
        # Draw background panel
        h, w = cropped_palm.shape[:2]
        cv2.rectangle(overlay, (5, h - 35), (w - 5, h - 5), (0, 0, 0), -1)
        # Put text
        cv2.putText(overlay, f"AuraPalm AI | {palm_shape.split(' ')[0]} Hand", (12, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Blend overlay with original image slightly (transparency)
        blended = cv2.addWeighted(cropped_palm, 0.4, overlay, 0.6, 0)
        return blended

    def _calculate_confidence(self, landmarks: List[Tuple[int, int, float]], 
                              life: Dict[str, Any], heart: Dict[str, Any], head: Dict[str, Any]) -> float:
        """
        Calculates confidence score based on physical landmarks presence and line detection clarity.
        """
        # Landmaks checking
        if len(landmarks) < 21:
            return 30.0
            
        # Line scores
        life_q = life.get("depth", 0) * 0.4 + life.get("length", 0) * 0.6
        heart_q = heart.get("depth", 0) * 0.4 + heart.get("length", 0) * 0.6
        head_q = head.get("depth", 0) * 0.4 + head.get("length", 0) * 0.6
        
        base_score = (life_q + heart_q + head_q) / 3.0
        
        # MediaPipe landmark z-depth stabilization
        z_avg = np.mean([abs(lm[2]) for lm in landmarks])
        stability_modifier = 20.0 * (1.0 - min(1.0, z_avg * 5.0))
        
        final_score = base_score + stability_modifier
        # Keep score in realistic range (55 to 98)
        return round(min(98.0, max(55.0, final_score)), 1)
