import httpx
import base64
import json
import logging
from typing import Dict, Any, Optional
from backend.app.core.config import settings

logger = logging.getLogger("uvicorn.error")

class LLMService:
    def __init__(self):
        self.ollama_url = settings.OLLAMA_URL
        self.text_model = settings.OLLAMA_MODEL
        self.vision_model = settings.VISION_MODEL

    async def get_reading(self, cv_data: Dict[str, Any], image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Attempts to query the local Ollama vision model (if image is provided) or text model
        with the palm data. Falls back to a deterministic, high-quality rule-based engine 
        if Ollama is unavailable or fails.
        """
        
        # Prepare standard payload describing CV measurements
        hand_type = cv_data.get("palm_shape", "Unknown").split(" ")[0]
        
        # Extract features for easy prompt integration
        features = {
            "hand_type": cv_data.get("hand_label", "Unknown"),
            "palm_shape": cv_data.get("palm_shape", "Unknown"),
            "life_line": {
                "length": cv_data["life_line"]["length"],
                "depth": cv_data["life_line"]["depth"],
                "curvature": cv_data["life_line"]["curvature"],
                "clarity": cv_data["life_line"]["clarity"]
            },
            "heart_line": {
                "length": cv_data["heart_line"]["length"],
                "depth": cv_data["heart_line"]["depth"],
                "curvature": cv_data["heart_line"]["curvature"],
                "clarity": cv_data["heart_line"]["clarity"]
            },
            "head_line": {
                "length": cv_data["head_line"]["length"],
                "depth": cv_data["head_line"]["depth"],
                "curvature": cv_data["head_line"]["curvature"],
                "clarity": cv_data["head_line"]["clarity"]
            },
            "fate_line": {
                "length": cv_data["fate_line"]["length"],
                "depth": cv_data["fate_line"]["depth"],
                "curvature": cv_data["fate_line"]["curvature"],
                "clarity": cv_data["fate_line"]["clarity"]
            },
            "mounts": cv_data["mounts"],
            "confidence_score": cv_data["confidence_score"]
        }

        # Attempt local LLM reasoning
        ai_response = None
        try:
            # Check if Ollama is running
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code == 200:
                    # Ollama is available!
                    # If we have an image and vision model is configured, try vision inference
                    if image_bytes and self.vision_model:
                        ai_response = await self._query_ollama_vision(features, image_bytes)
                    
                    # If vision failed or wasn't run, fallback to local text LLM
                    if not ai_response:
                        ai_response = await self._query_ollama_text(features)
        except Exception as e:
            logger.warning(f"Local Ollama model unavailable or failed: {str(e)}. Using local rule-based interpretation engine.")
        
        # If AI model didn't return a valid response, use rule-based engine
        if not ai_response:
            ai_response = self._generate_rule_based_reading(features)
            ai_response["ai_source"] = "Local CV Rule Engine"
        else:
            ai_response["ai_source"] = "Local AI Model (Ollama)"

        return ai_response

    async def _query_ollama_vision(self, features: Dict[str, Any], image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Sends the cropped palm image and CV stats to local vision model.
        """
        try:
            base64_img = base64.b64encode(image_bytes).decode('utf-8')
            
            prompt = f"""
            You are a professional Vedic palmistry (chiromancy) and astrology expert. Analyze this palm image along with these computer vision measurements:
            {json.dumps(features, indent=2)}
            
            Provide a deep, realistic, and commercially premium interpretation utilizing rich astrological terminology (e.g. planetary houses, zodiac energies, celestial mounts).
            You MUST return a JSON object EXACTLY conforming to this schema:
            {{
              "personality": {{
                "analysis": "detailed paragraph explaining inner nature, behavioral patterns based on palm shape, head line, and planetary mounts",
                "key_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3"]
              }},
              "career": {{
                "analysis": "detailed paragraph analyzing ambitions, professional talents, fate line (Saturn's path), and mercury/jupiter mounts",
                "key_takeaways": ["takeaway 1", "takeaway 2"],
                "score": 85
              }},
              "relationships": {{
                "analysis": "detailed paragraph explaining emotional profile, attachment style, based on heart line and venus/luna mounts",
                "key_takeaways": ["takeaway 1", "takeaway 2"],
                "score": 90
              }},
              "life_line_meaning": "specific explanation of physical energy, Venusian vitality, and life path flow",
              "heart_line_meaning": "specific explanation of emotional depth, romantic temperament, and interpersonal style",
              "head_line_meaning": "specific explanation of mental clarity, Mercurial intelligence, and cognitive focus",
              "fate_line_meaning": "specific explanation of career path, Saturnian structure, and professional adaptability",
              "energy_profile": {{
                "vitality": 80,
                "emotion": 75,
                "intellect": 85,
                "ambition": 70
              }}
            }}
            Return ONLY raw JSON, without markdown formatting. Do not wrap in ```json ```.
            """

            payload = {
                "model": self.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64_img]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2
                },
                "format": "json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.ollama_url}/api/chat", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    content = data["message"]["content"]
                    return json.loads(content)
        except Exception as e:
            logger.error(f"Ollama Vision error: {str(e)}")
        return None

    async def _query_ollama_text(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sends the CV stats to local text model.
        """
        try:
            prompt = f"""
            You are a professional chiromancy and astrology expert. Write a premium palmistry reading using these computer vision measurements from a hand:
            {json.dumps(features, indent=2)}

            Provide a deep, realistic, and commercially premium interpretation utilizing rich astrological terminology (planetary houses, zodiac aspects, mounts, and celestial influences).
            You MUST return a JSON object EXACTLY conforming to this schema:
            {{
              "personality": {{
                "analysis": "detailed paragraph explaining inner nature, behavioral patterns based on palm shape, head line, and planetary mounts",
                "key_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3"]
              }},
              "career": {{
                "analysis": "detailed paragraph analyzing ambitions, professional talents, fate line (Saturn's path), and mercury/jupiter mounts",
                "key_takeaways": ["takeaway 1", "takeaway 2"],
                "score": 85
              }},
              "relationships": {{
                "analysis": "detailed paragraph explaining emotional profile, attachment style, based on heart line and venus/luna mounts",
                "key_takeaways": ["takeaway 1", "takeaway 2"],
                "score": 90
              }},
              "life_line_meaning": "specific explanation of physical energy, Venusian vitality, and life path flow",
              "heart_line_meaning": "specific explanation of emotional depth, romantic temperament, and interpersonal style",
              "head_line_meaning": "specific explanation of mental clarity, Mercurial intelligence, and cognitive focus",
              "fate_line_meaning": "specific explanation of career path, Saturnian structure, and professional adaptability",
              "energy_profile": {{
                "vitality": 80,
                "emotion": 75,
                "intellect": 85,
                "ambition": 70
              }}
            }}
            Return ONLY raw JSON, without markdown formatting. Do not wrap in ```json ```.
            """

            payload = {
                "model": self.text_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                },
                "format": "json"
            }

            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    content = data["response"]
                    return json.loads(content)
        except Exception as e:
            logger.error(f"Ollama Text error: {str(e)}")
        return None

    def _generate_rule_based_reading(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        A highly detailed rule-based palmistry interpretation engine tailored for astrological purposes.
        Ensures the client receives a premium reading even when the local AI models are offline.
        """
        # Get line details
        life = features["life_line"]
        heart = features["heart_line"]
        head = features["head_line"]
        fate = features["fate_line"]
        mounts = features["mounts"]
        
        hand_type = features["hand_type"]
        palm_shape = features["palm_shape"]

        # Parse palm shape type
        shape_type = "Earth"
        if "Air" in palm_shape:
            shape_type = "Air"
        elif "Water" in palm_shape:
            shape_type = "Water"
        elif "Fire" in palm_shape:
            shape_type = "Fire"

        # 1. Evaluate Life Line (Venus / Vitality)
        life_meanings = []
        vitality_val = int(life["depth"] * 0.5 + life["length"] * 0.5)
        if life["length"] > 60:
            life_meanings.append("Your Life Line is exceptionally long and sweeping, indicating high physical endurance, Venusian vitality, and a robust constitution that enables you to bounce back quickly from setbacks.")
        else:
            life_meanings.append("Your Life Line is moderate in length, indicating that you manage your energy carefully. Rather than burning through physical reserves, you prefer balanced pacing to maintain stamina.")

        if "Deep" in life["clarity"]:
            life_meanings.append("The depth of the line suggests a highly grounded individual with strong natural resistance. You possess a stable core energy profile.")
        else:
            life_meanings.append("The lighter nature of your Life Line indicates a sensitive nervous system. You are highly responsive to environmental stress and must prioritize restful environments.")

        if life["curvature"] > 50:
            life_meanings.append("Its strong curve around the Mount of Venus reflects an outgoing, warm-hearted personality with a strong love for family life and adventure.")
        else:
            life_meanings.append("Its straighter path shows a reserved, self-reliant nature, choosing independence and privacy over crowd-seeking behaviors.")

        life_meaning_str = " ".join(life_meanings)

        # 2. Evaluate Heart Line (Venus / Moon / Emotions)
        heart_meanings = []
        emotion_val = int(heart["depth"] * 0.5 + heart["length"] * 0.5)
        if heart["length"] > 60:
            heart_meanings.append("Your Heart Line runs extensively across the upper palm, suggesting a deeply empathetic, expressive, and romantic nature.")
        else:
            heart_meanings.append("Your Heart Line is concise, indicating a highly logical, private approach to emotions, prioritizing stability and actions over verbal declarations of love.")

        if heart["curvature"] > 45:
            heart_meanings.append("Its upward curve towards the fingers shows that you wear your heart on your sleeve, expressing emotions freely and passionately.")
        else:
            heart_meanings.append("Its straight orientation reveals a rational, composed emotional style, preferring to analyze feelings before acting on them.")

        if "Deep" in heart["clarity"]:
            heart_meanings.append("The deep indentation reflects intense loyalty and long-term commitment capacity once trust is established.")
        else:
            heart_meanings.append("A softer heart line indicates a cautious, guarded heart that requires time and security to fully open up to others.")

        heart_meaning_str = " ".join(heart_meanings)

        # 3. Evaluate Head Line (Mercury / Intellect)
        head_meanings = []
        intellect_val = int(head["depth"] * 0.5 + head["length"] * 0.5)
        if head["length"] > 65:
            head_meanings.append("Your Head Line is long and prominent, denoting an analytical mind, excellent concentration, and a thirst for diverse knowledge.")
        else:
            head_meanings.append("Your Head Line is short and direct, showing a highly practical, execution-oriented, and decisive cognitive style. You value common sense over abstract theories.")

        if head["curvature"] > 40:
            head_meanings.append("Its downward slope towards the Mount of Luna reflects a creative, highly imaginative mind, drawing inspiration from art, writing, and intuition.")
        else:
            head_meanings.append("Its straight trajectory across the palm indicates a scientific, logical, and realistic mindset, focusing on facts and structured reasoning.")

        head_meaning_str = " ".join(head_meanings)

        # 4. Evaluate Fate Line (Saturn / Destiny)
        fate_meanings = []
        ambition_val = 50
        if fate["clarity"] == "Absent / Faint":
            fate_meanings.append("Your Fate Line is faint or absent, which in classical palmistry denotes a self-made path. You are not bound by rigid expectations and enjoy high professional flexibility.")
            ambition_val = int(35 + (mounts.get("Jupiter", "").find("Prominent") != -1) * 20)
        else:
            fate_meanings.append("A visible Fate Line indicates a structured life path, with strong career alignment and a sense of duty, purpose, and direction starting early on.")
            ambition_val = int(60 + fate["depth"] * 0.3)

        fate_meaning_str = " ".join(fate_meanings)

        # 5. Build Mount highlights
        prominent_mounts = [m for m, desc in mounts.items() if "Prominent" in desc]
        prominent_mounts_str = ", ".join(prominent_mounts) if prominent_mounts else "Balanced Mounts"

        # 6. Compose Personality Segment
        p_analysis = f"As an individual with a {palm_shape.split(' ')[0]} hand, your base nature is highly shaped by the {shape_type} element. "
        if shape_type == "Earth":
            p_analysis += "Under the lens of Vedic astrology, your Earth nature anchors you in Saturnian discipline and reliability, making you grounded, patient, and highly dependable. "
        elif shape_type == "Air":
            p_analysis += "Under the lens of Vedic astrology, your Air nature is driven by Mercurial curiosity and intellectual dialogue, highlighting an active mental field. "
        elif shape_type == "Water":
            p_analysis += "Under the lens of Vedic astrology, your Water nature represents Lunar sensitivity and high intuitive connectivity, absorbing environmental energies like a sponge. "
        else: # Fire
            p_analysis += "Under the lens of Vedic astrology, your Fire nature reflects Solar warmth and Martian courage, driving you to lead, inspire, and seek constant activity. "

        p_analysis += f"With a {head['clarity'].lower()} Head Line and prominent activity in the Mount of {prominent_mounts[0] if prominent_mounts else 'Jupiter'}, your personality represents a unique blend of logic and ambition. "
        p_analysis += "You balance internal contemplation with external execution, finding success when you align your physical surroundings with your mental goals."

        p_takeaways = [
            f"Zodiac element: {shape_type} - governing your core temperament.",
            f"Active planetary mounts: {prominent_mounts_str}.",
            "Celestial alignment: Highly responsive cognitive style with natural focus."
        ]

        # 7. Compose Career Segment
        c_score = int(65 + (intellect_val * 0.2) + (ambition_val * 0.15))
        c_analysis = f"Your professional destiny is guided by a {head['clarity'].lower()} line of intellect (ruled by Mercury) and a {fate['clarity'].lower()} path of destiny (ruled by Saturn). "
        if "Absent" in fate["clarity"]:
            c_analysis += "In astrological terms, your faint Saturn line grants you high karmic freedom. You thrive in entrepreneurial, creative, or flexible roles where you dictate your own direction. "
        else:
            c_analysis += "Your clear Saturn line shows a structured karmic path. You excel in corporate leadership, organizational administration, or projects requiring long-term discipline. "
        
        if "Jupiter" in prominent_mounts:
            c_analysis += "The dominance of your Jupiter mount highlights strong leadership instincts, public-speaking capability, and executive potential."
        elif "Mercury" in prominent_mounts:
            c_analysis += "Your Mercury mount indicates natural commercial sense, persuasion capability, and technological/business intelligence."
        else:
            c_analysis += "You work best when utilizing analytical precision and steady, reliable skill accumulation."

        c_takeaways = [
            "Best suited environment: Autonomy with clear output metrics." if "Absent" in fate["clarity"] else "Best suited environment: Structured authority with growth path.",
            "Strongest vocational skill: Practical problem solving and strategic planning."
        ]

        # 8. Compose Relationships Segment
        r_score = int(60 + (emotion_val * 0.2) + (vitality_val * 0.1))
        r_analysis = f"In relationships, your {heart['clarity'].lower()} Heart Line indicates a { 'deeply expressive' if heart['length'] > 60 else 'thoughtful and reserved'} approach to connection. "
        if heart["curvature"] > 45:
            r_analysis += "Your Venusian energy craves high emotional reciprocity, passionate dialogue, and warm validation. "
        else:
            r_analysis += "You value steady reliability and shared intellectual interests, showing love through dedicated service. "
            
        if "Venus" in prominent_mounts:
            r_analysis += "Your prominent Venus mount infuses your relationships with warmth, sensuality, and a strong appreciation for harmony and aesthetics."
        else:
            r_analysis += "You seek partners who respect your personal space and offer calm, stable companionship."

        r_takeaways = [
            "Attachment style: High loyalty with measured emotional boundaries.",
            "Key relationship need: Mutual respect, intellectual compatibility, and clear boundaries."
        ]

        # Ensure all scores fit in bounds
        c_score = min(98, max(50, c_score))
        r_score = min(98, max(50, r_score))

        return {
            "personality": {
                "analysis": p_analysis,
                "key_takeaways": p_takeaways
            },
            "career": {
                "analysis": c_analysis,
                "key_takeaways": c_takeaways,
                "score": c_score
            },
            "relationships": {
                "analysis": r_analysis,
                "key_takeaways": r_takeaways,
                "score": r_score
            },
            "life_line_meaning": life_meaning_str,
            "heart_line_meaning": heart_meaning_str,
            "head_line_meaning": head_meaning_str,
            "fate_line_meaning": fate_meaning_str,
            "energy_profile": {
                "vitality": vitality_val,
                "emotion": emotion_val,
                "intellect": intellect_val,
                "ambition": ambition_val
            }
        }
