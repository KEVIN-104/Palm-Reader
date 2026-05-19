# Deploying AuraPalm AI to Netlify and Render (100% Free Tier)

Because AuraPalm AI uses high-performance computer vision (OpenCV, MediaPipe) and local database storage (SQLite), the application is divided into a **Static Frontend (SPA)** and a **Dockerized Backend**.

Here is the step-by-step guide to hosting both components online.

---

## Part 1: Deploying the Backend to Render

Render provides a free tier that supports Docker containers. It reads the `Dockerfile` in this repository, compiles the dependencies (OpenCV, MediaPipe, FastAPI), and serves the API.

### Method A: Using Render Blueprint (Recommended - 1 Click Setup)
We have included a `render.yaml` blueprint file in the repository. This automatically configures your service name, docker environment, free tier plan, and configures the **1 GB persistent disk** at `/app/static` so you don't lose your database or scanned uploads when the container restarts.

1. **Push your code to GitHub / GitLab / Bitbucket**:
   Ensure all files (including `Dockerfile`, `render.yaml`, `backend/`, and `static/`) are pushed to a repository on your Git hosting provider.
2. **Go to Render Blueprints**:
   - Go to [dashboard.render.com](https://dashboard.render.com) and sign in.
   - Click **New** -> **Blueprint**.
   - Connect your Git repository.
3. **Approve and Deploy**:
   - Render will parse `render.yaml` and show the resources it will create (`aurapalm-api` web service with a persistent disk volume).
   - Click **Apply** or **Deploy**.
   - Render will build and deploy the application. Your backend URL will look like: `https://aurapalm-api.onrender.com`.

---

### Method B: Manual Deployment
If you prefer not to use the Blueprint:

1. **Push your code to GitHub / GitLab / Bitbucket**.
2. **Create a new Web Service on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com), click **New** -> **Web Service**.
   - Connect your Git repository.
3. **Configure Build Settings**:
   - **Name**: `aurapalm-api`
   - **Runtime**: Select **Docker** (Render will automatically detect the root `Dockerfile`).
   - **Instance Type**: Select **Free**.
4. **Set Up Persistent Storage**:
   - Scroll down to the **Advanced** section.
   - Click **Add Disk**.
   - **Name**: `aurapalm-storage`
   - **Mount Path**: `/app/static`
   - **Size**: `1 GB`
5. **Deploy**:
   - Click **Create Web Service**.

> [!NOTE]
> The SQLite database file is configured to be saved automatically at `/app/static/palm_readings.db`. Because the persistent disk is mounted at `/app/static`, your database history and images will be fully preserved across updates and restarts.

---

## Part 2: Deploying the Frontend to Netlify

Netlify is a static site hosting platform. Our single-page application (`index.html`) is completely self-contained, making Netlify hosting straightforward.

### Method A: Git-based Continuous Deployment (Recommended)
1. Log in to [Netlify](https://www.netlify.com/).
2. Click **Add new site** -> **Import from an existing project**.
3. Choose your Git provider and select the repository.
4. **Build settings**:
   - **Base directory**: `backend/app/templates`
   - **Build command**: (Leave empty)
   - **Publish directory**: `.` (This serves our static `index.html`).
5. Click **Deploy Site**.

### Method B: Drag and Drop (Instant Deploy)
1. Create a new folder on your computer named `public`.
2. Copy `backend/app/templates/index.html` and paste it inside the `public` folder.
3. Open the Netlify app dashboard, go to the **Sites** page, and scroll to the bottom.
4. Drag and drop the `public` folder directly into the upload area on Netlify.
5. Your frontend is instantly live (e.g., `https://aurapalm.netlify.app`).

---

## Part 3: Connecting Frontend to Backend

Since we enabled CORS wildcards in `backend/app/main.py`, the frontend on Netlify can safely send requests to the backend on Render.

1. Open your live Netlify website on your phone or PC.
2. In the sidebar, click on **Local AI Setup** (or the Settings icon).
3. Find the new **FastAPI Backend Server URL** configuration field.
4. Paste your Render backend URL (e.g., `https://aurapalm-api.onrender.com`) and click **Save URL**.
5. Go back to the **Scan Palm** page and perform a scan. The static frontend will now securely dispatch scans, display the MediaPipe lines overlay, and fetch database history from your deployed server!
