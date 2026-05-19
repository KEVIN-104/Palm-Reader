# AuraPalm AI - Free Local AI Palm Reading Platform

A premium, production-quality, completely offline AI Palmistry and Hand Analysis platform. AuraPalm AI runs hand tracking, image quality checks, contour and crease feature extraction, and deep neural interpretations **100% locally on your computer** without any external cloud APIs, billing, or third-party data collection.

---

## 🌟 Key Features

1. **Local Image Processing Pipeline**:
   - **Quality checks**: Automatically rejects blurred, under-exposed, or over-exposed palm photographs using Laplacian Variance and luminance histogram analysis.
   - **Landmark Detection**: Uses local MediaPipe Hands models to identify 21 structural landmarks.
   - **Hand Normalization**: Automatically calculates wrist-to-finger angles, rotates the hand vertically, and crops the palm bounding box.
   - **Background Masking**: Isolates the palm from the surrounding background using convex hulls.

2. **OpenCV Feature Extraction**:
   - **Line Enhancement**: Applies a morphological Blackhat filter combined with CLAHE (Contrast Limited Adaptive Histogram Equalization) to isolate and highlight palm creases.
   - **Metrics Extraction**: Quantifies line length, depth, and curvature for the **Life Line**, **Head Line**, **Heart Line**, and **Fate Line** by tracing customized Bezier search zones.
   - **Palm Shape Classification**: Classifies hand types into elements (Earth, Air, Fire, Water) based on width-to-height and finger-to-palm ratios.
   - **Mount Prominence**: Analyzes skin-gradient variances to measure the prominence of the Jupiter, Saturn, Apollo, Mercury, Venus, and Luna mounts.

3. **Hybrid AI Engine**:
   - **Multimodal Vision Integration**: Encourages native Ollama integration (Qwen2-VL or Llava models) to analyze physical images along with OpenCV indicators.
   - **Fallback CV Rule Engine**: Incorporates a detailed, template-driven local palmistry rules engine. If Ollama is offline or unavailable, the system continues to output full readings immediately based on physical measurements, avoiding system crashes.

4. **Premium Dark Dashboard UI**:
   - Responsive dark-mode glassmorphic interface.
   - Live browser camera capture (supports mobile cameras and webcams).
   - Real-time animated processing step trackers.
   - SVG-rendered confidence dials, energy progress meters, expandable accordion line sheets, and mount cards.
   - Client-side printable layout designed for downloading styled PDF reports.

---

## 🛠️ Technology Stack

- **Backend Framework**: Python FastAPI
- **Hand Tracking**: MediaPipe Hands
- **Image Processing**: OpenCV, NumPy, Pillow
- **Local AI Client**: Ollama (Qwen2-VL / Qwen2.5 / Llama 3)
- **Database**: SQLite (SQLAlchemy ORM)
- **Containerization**: Docker & Docker Compose
- **Frontend**: Single-Page HTML5, CSS3 (Vanilla Glassmorphism), JavaScript (ES6, Camera Streams, SVG)

---

## 📂 Project Structure

```text
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py                # FastAPI main entrypoint and static file configurations
│       ├── api/
│       │   ├── __init__.py
│       │   └── endpoints.py       # REST endpoints (health, analyze, history CRUD)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py          # App settings (Pydantic Settings)
│       │   └── database.py        # SQLite connections and session setups
│       ├── models/
│       │   ├── __init__.py
│       │   └── reading.py         # SQLAlchemy SQLite data models
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── reading.py         # Pydantic validation schemas
│       ├── services/
│       │   ├── __init__.py
│       │   ├── mediapipe_service.py # Quality validation, tracking, alignment & cropping
│       │   ├── opencv_service.py    # Grayscale, Blackhat filters, line tracing, mount analysis
│       │   └── llm_service.py       # Ollama API client & local rule fallback generator
│       └── templates/
│           └── index.html         # Premium dark mode glassmorphic SPA UI
├── static/                        # Local file asset uploads (git-ignored)
│   ├── uploads/                   # Original uploads
│   └── analyzed/                  # Overlay visualizations
├── test_system.py                 # Asynchronous pipeline testing CLI script
├── requirements.txt               # Python package dependencies
├── Dockerfile                     # Headless package configuration for container execution
├── docker-compose.yml             # Local networking routing compose
├── run_app.bat                    # One-click Windows startup script
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started (Native Windows)

### Option A: One-Click Startup (Recommended)
Simply double-click the **`run_app.bat`** script in the project root. The script will:
1. Detect local Python installations.
2. Initialize the `.venv` folder automatically.
3. Upgrade `pip` and install all dependencies listed in `requirements.txt`.
4. Open your web browser to `http://127.0.0.1:8000`.
5. Launch the local FastAPI Uvicorn reload server.

### Option B: Manual Setup via Terminal
1. Open PowerShell or Command Prompt in the project folder.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Start the server:
   ```powershell
   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
5. Navigate to `http://127.0.0.1:8000` in your web browser.

---

## 🐳 Docker Deployment

The application includes full containerization. To run the complete system (including mapping network gateways to communicate with your host's Ollama installation):

1. Start Ollama natively on your host machine.
2. Run Docker Compose in the project folder:
   ```bash
   docker-compose up --build
   ```
3. Open your browser to `http://localhost:8000`.
4. *Note: Database changes and uploaded images are fully mapped and persisted in your local directory.*

---

## 🧠 Setting Up Local AI (Ollama)

AuraPalm AI works immediately out-of-the-box using the built-in mathematical CV rules engine. To activate advanced AI model reasoning:

1. Download Ollama from [ollama.com](https://ollama.com).
2. Install a vision model (e.g., `qwen2-vl:latest` or `llava`) or a text model (e.g., `qwen2.5:7b` or `llama3`) by executing:
   ```bash
   ollama run qwen2-vl
   ```
3. Launch AuraPalm AI. The sidebar status indicator for Ollama will switch from **Offline** to **Online (X models)**.
4. When you upload a hand image, the backend will feed the cropped palm and OpenCV measurements directly to the vision model to construct a fully personalized reading.

---

## 🧪 Pipeline Test Script

You can test the entire pipeline on a local image file directly from the terminal without launching the web server. Run:

```powershell
.venv\Scripts\python test_system.py path/to/your/palm_photo.jpg
```

This will run quality checks, detect hands, crop the palm, trace lines, simulate Ollama/fallback interpretations, print the full JSON result to the console, and output a diagnostic visualization image named `test_result.jpg`.
