# ClassPulse AI — Multimodal Attendance & Engagement Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.15-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Charts-Plotly%20Dark-38BDF8.svg)](https://plotly.com/)
[![Testing: Pytest](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](https://pytest.org/)

An academic, multimodal artificial intelligence agent designed to automate presence-based classroom attendance, analyze acoustic classroom activity, estimate student engagement heuristics, and provide non-prescriptive, explainable pedagogical suggestions to instructors without requiring cloud APIs or invasive biometric facial recognition.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Functional Requirements](#4-functional-requirements)
5. [Architecture & System Flow](#5-architecture--system-flow)
6. [Technology Stack](#6-technology-stack)
7. [Windows Installation Instructions](#7-windows-installation-instructions)
8. [Virtual-Environment Setup](#8-virtual-environment-setup)
9. [Dependency Installation](#9-dependency-installation)
10. [Application Startup Command](#10-application-startup-command)
11. [Webcam Instructions](#11-webcam-instructions)
12. [Video-File Processing Instructions](#12-video-file-processing-instructions)
13. [Audio-Input Instructions](#13-audio-input-instructions)
14. [Attendance Calculation Logic](#14-attendance-calculation-logic)
15. [Audio-Analysis Method & VAD](#15-audio-analysis-method--vad)
16. [Multimodal Fusion Approach](#16-multimodal-fusion-approach)
17. [Engagement-Scoring Formula](#17-engagement-scoring-formula)
18. [Recommendation Logic](#18-recommendation-logic)
19. [Analytics, Archive & Export Instructions](#19-analytics-archive--export-instructions)
20. [Testing & Self-Test Instructions](#20-testing--self-test-instructions)
21. [Privacy & Ethical Safeguards](#21-privacy--ethical-safeguards)
22. [Known Limitations](#22-known-limitations)
23. [Troubleshooting Instructions](#23-troubleshooting-instructions)

---

## 1. Project Overview

**ClassPulse AI** is a fully functional, offline multimodal AI application engineered to support modern educational environments. Traditional roll-call attendance is disruptive, prone to proxy attendance, and unreflective of actual classroom presence. Furthermore, instructors in large lecture halls or remote seminars often struggle to gauge student participation and room dynamics in real time.

ClassPulse AI resolves these challenges by fusing **computer vision face tracking** and **acoustic signal processing** into an explainable heuristic reasoning agent that operates 100% locally on Windows without third-party cloud dependencies or external inference costs.

---

## 2. Problem Statement

1. **Inefficient Attendance Tracking:** Manual roll-calls consume 5–10% of lecture time. Single-frame photographic attendance methods falsely mark individuals as present if they merely walk through the camera frame for a fraction of a second.
2. **Single-Modality Blindspots:** Vision-only systems cannot detect vocal interaction, verbal questions, or distracting ambient background noise; audio-only systems cannot determine physical head presence or posture movement.
3. **Biometric Privacy Hazards:** Cloud-based facial identity recognition introduces severe ethical risks, regulatory hurdles (FERPA/GDPR), and identity theft vulnerabilities.
4. **Opaque "Black-Box" Metrics:** Proprietary AI tools frequently produce unexplained engagement numbers that misinterpret quiet note-taking as inattention or loud classroom noise as student enthusiasm.

---

## 3. Objectives

- **Autonomous Presence-Based Attendance:** Calculate attendance based on accumulated observed presence over time relative to session duration (default 60%) rather than instantaneous detections.
- **Privacy-Preserving Vision Tracking:** Maintain anonymous participant identifiers (`Participant_01`, `Participant_02`) across frames using MediaPipe BlazeFace detection and hybrid IoU/centroid association without storing biometric facial embeddings.
- **Acoustic Signal Characterization:** Compute Root Mean Square (RMS) energy, rolling 10th-percentile ambient noise floor (unaffected by speech), and flicker-free Voice Activity Detection (VAD) with hangover logic.
- **Multimodal Signal Fusion:** Synchronize asynchronous vision and audio streams into discrete, timestamped 1.0-second intervals while handling single-modality sessions gracefully.
- **Explainable Engagement Index:** Provide a transparent, rule-based 0–100 engagement score with component contribution breakdowns and mathematical formula inspection.
- **Data-Driven Pedagogical Guidance:** Generate context-aware suggestions for instructors based on computed trends with explicit supporting metrics and caveats.

---

## 4. Functional Requirements

Mapped directly to the **SkillWallet Project Specification (Epics 1–6)**:
- **Milestone 1:** Offline dependency configuration, Windows virtual environment isolation, reproducible `requirements.txt`.
- **Milestone 2:** Live webcam and pre-recorded video support, MediaPipe BlazeFace face detection with Haar cascade fallback, hybrid IoU/centroid tracking, relative percentage attendance duration, gap tolerance bridging.
- **Milestone 3:** Audio track extraction from video files, standalone WAV/MP3 ingestion, signal normalization, RMS energy calculation, rolling 10th-percentile noise floor estimation, flicker-free VAD with hangover.
- **Milestone 4:** Timestamp-based signal synchronization, defensive missing modality handling, explainable multimodal engagement scoring.
- **Milestone 5:** Rule-based teaching suggestions (sustained low engagement, excessive background noise, sudden drops, participation surges), session analytics generation, CSV, JSON (with schema version), and printable HTML/PDF reports.
- **Milestone 6:** Modern Streamlit dashboard featuring Plotly dark visualizations, top KPI cards, interactive score explainer, past session comparer, and automated unit test suite.

---

## 5. Architecture & System Flow

```
                      ┌─────────────────────────────────────────┐
                      │          ClassPulse AI System           │
                      └─────────────────────────────────────────┘
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
 ┌──────────────────────┐                                    ┌──────────────────────┐
 │    Vision Module     │                                    │     Audio Module     │
 │ (Video File / Webcam)│                                    │ (WAV / Extracted MP4)│
 └──────────┬───────────┘                                    └──────────┬───────────┘
            │                                                           │
            ▼                                                           ▼
 ┌──────────────────────┐                                    ┌──────────────────────┐
 │ FaceDetector         │                                    │ AudioProcessor       │
 │  - MediaPipe Primary │                                    │  - Resample 16 kHz   │
 │  - Haar Fallback     │                                    │  - Mono conversion   │
 │  - CLAHE Low-light   │                                    │  - Time Segmentation │
 │ Hybrid IoU Tracker   │                                    └──────────┬───────────┘
 │ AttendanceManager    │                                               │
 │  - Relative % (60%)  │                                               ▼
 │  - Gap Tolerance     │                                    ┌──────────────────────┐
 └──────────┬───────────┘                                    │ AudioActivityAnalyzer│
            │                                                │  - RMS Signal Energy │
            │                                                │  - 10th %ile Noise   │
            │                                                │  - VAD with Hangover │
            │                                                └──────────┬───────────┘
            │                                                           │
            └─────────────────────────────┬─────────────────────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │    MultimodalFusion     │
                             │ (Timestamp-aligned bins)│
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │   EngagementEstimator   │
                             │ (Explainable 0-100 Score│
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │  RecommendationEngine   │
                             │ (Pedagogical Guidance)  │
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │ SessionReportGenerator  │
                             │  - Streamlit Dashboard  │
                             │  - HTML / PDF Reports   │
                             │  - JSON / CSV Exports   │
                             └─────────────────────────┘
```

---

## 6. Technology Stack

| Layer | Component | Version / Library | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.9+ (tested on Python 3.15.1) | Core application runtime |
| **Computer Vision** | MediaPipe & OpenCV | `mediapipe>=0.10.0`, `opencv-python>=4.10.0` | Primary BlazeFace detection + Haar fallback |
| **Face Tracking** | Custom Hybrid | IoU + Euclidean Centroid | Anonymous tracking with re-entry persistence |
| **Audio Processing** | SoundFile, Librosa, ImageIO-FFMPEG | `soundfile`, `librosa`, `imageio-ffmpeg` | Audio track extraction, resampling, RMS, VAD |
| **Numerical Processing** | NumPy, SciPy | `numpy>=2.0.0`, `scipy>=1.15.0` | Signal vectors, distance matrices, percentiles |
| **Data Structuring** | Pandas | `pandas==3.0.6` | Tabular attendance and metric records |
| **Web Dashboard** | Streamlit | `streamlit>=1.35.0` | Interactive educator interface |
| **Plotting & Charts** | Plotly | `plotly>=5.20.0` | Interactive dark-theme charts with hover tooltips |
| **Test Automation** | Pytest | `pytest>=8.0.0` | Automated unit and integration testing |

---

## 7. Windows Installation Instructions

Open PowerShell on your Windows computer and navigate to the project directory:

```powershell
cd C:\Projects\ClassPulseAI
```

Ensure Python is installed on your system:
```powershell
python --version
```

---

## 8. Virtual-Environment Setup

Create an isolated virtual environment named `venv` inside the project root:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment in PowerShell
.\venv\Scripts\Activate.ps1
```

*(Note: If PowerShell displays an `Execution_Policies` restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activation script).*

---

## 9. Dependency Installation

With the virtual environment activated, install the locked, compatible dependencies from `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Application Startup Command

### A. Launch Streamlit Web Dashboard

```powershell
.\venv\Scripts\streamlit run app.py
```
This automatically starts the local web server and opens the dashboard in your default web browser at `http://localhost:8501`.

### B. Headless CLI Execution

To run the pipeline directly from the command line:

```powershell
.\venv\Scripts\python cli.py --session-id "lecture_01" --video "data/sample_media/sample_classroom_video.mp4" --audio "data/sample_media/sample_classroom_audio.wav" --min-presence-ratio 0.60
```

### C. System Self-Test Command

To run the automated subsystem diagnostic test suite:

```powershell
.\venv\Scripts\python cli.py --selftest
```

---

## 11. Webcam Instructions

1. Open the dashboard and navigate to **🚀 Live & Video Processing**.
2. Select **📹 Live Webcam Feed** under Primary Input Source.
3. Click **▶️ Start Camera Analysis**.
4. The application captures video frames in real time, applies MediaPipe face detection, tracks movement, and accumulates presence.
5. Click **⏹️ Stop & Finalize** to finish processing and view reports.

---

## 12. Video-File Processing Instructions

1. Place pre-recorded video files (`.mp4`, `.avi`, `.mkv`, `.mov`) in any accessible folder.
2. Navigate to **🚀 Live & Video Processing** and select **📁 Upload Classroom Video**.
3. Drag and drop the video into the Streamlit uploader.
4. Set frame skipping (e.g. 2 for 2x real-time speed).
5. Click **🚀 Process Video in Streaming Fashion**.
6. The pipeline automatically decodes visual frames, extracts the embedded audio soundtrack using local FFMPEG, performs multimodal fusion, and generates analytics.

---

## 13. Audio-Input Instructions

1. Standalone audio files (`.wav`, `.mp3`, `.ogg`) can be processed independently of video.
2. Navigate to **🚀 Live & Video Processing** and select **🎙️ Standalone Audio File**.
3. Click **🚀 Process Audio Only**.
4. The system operates in **AUDIO_ONLY** modality mode, producing acoustic energy profiles, noise floor estimations, and speech activity indicators while logging `has_vision = False`.

---

## 14. Attendance Calculation Logic

The presence-based attendance algorithm operates in [`vision/attendance.py`](file:///C:/Projects/ClassPulseAI/vision/attendance.py):

1. **Relative Percentage Threshold:** The required presence duration is calculated as a configurable percentage of the total session length:
   $$\text{Required Presence Seconds} = \text{Session Duration} \times \text{min\_presence\_ratio} \quad (\text{default: } 60\%)$$
   *(An absolute-seconds override can also be toggled in the sidebar).*
2. **Threshold Feasibility Validation:** If a user specifies an impossible absolute threshold ($\text{required} > \text{session\_duration}$), the UI warns the instructor immediately.
3. **Gap Tolerance Merging:** If a subsequent detection occurs within $\Delta t \le \text{gap\_tolerance}$ (default 2.5s), the previous interval is extended rather than split.
4. **Overlapping Interval Union:** All recorded intervals are mathematically merged to strictly prevent double-counting:
   $$\text{Accumulated Presence} = \sum_{i} (\text{merged\_interval\_end}_i - \text{merged\_interval\_start}_i)$$
5. **Detailed Absence Rationale:** If a participant is marked absent, the exact shortfall is explained (e.g., `Observed 4.2s / 7.2s required (58.3%)`).

---

## 15. Audio-Analysis Method & VAD

Located in [`audio/activity.py`](file:///C:/Projects/ClassPulseAI/audio/activity.py):

1. **RMS Energy:**
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x[n]^2}$$
2. **Robust Background Noise Floor:** Uses a rolling 10th percentile of quiet, non-speech segments. High-energy vocal speech does **not** inflate the background noise floor.
3. **Signal-to-Noise Ratio (SNR):**
   $$\text{SNR}_{\text{est}} = 20 \log_{10} \left( \max\left(1.0, \frac{\text{RMS}}{\text{NoiseFloor}}\right) \right)$$
4. **Flicker-Free Voice Activity Detection (VAD) with Hangover:**
   - Raw detection: $\text{RMS} \ge \text{NoiseFloor} \times 1.60$ and $0.015 \le \text{ZCR} \le 0.40$.
   - **Hangover Logic:** When speech ceases, a 2-window (1.0s) hangover buffer sustains the active speech classification across inter-syllable pauses to prevent timeline flickering.

---

## 16. Multimodal Fusion Approach

Located in [`agent/fusion.py`](file:///C:/Projects/ClassPulseAI/agent/fusion.py):

- Synchronizes vision data (sampled at 15–30 fps) and audio data (segmented in 1.0s chunks) into unified 1.0-second time bins.
- For each time bin $k$:
  - `active_faces_count`: Number of distinct tracked participants observed.
  - `mean_movement_displacement`: Average pixel displacement across consecutive centroids.
  - `audio_rms_energy`: Mean acoustic RMS energy across overlapping audio windows.
  - `audio_is_speech`: Boolean indicating whether vocal activity occurred.
  - Explicit boolean flags `has_vision` and `has_audio`.
- Missing modalities are never fabricated: if camera data is unavailable, vision attributes are marked `None` / `0` and `has_vision = False`.

---

## 17. Engagement-Scoring Formula

Located in [`agent/engagement.py`](file:///C:/Projects/ClassPulseAI/agent/engagement.py):

The heuristic engagement index produces an explainable value between $0.0$ and $100.0$:

### Component Scores

1. **Visual Presence Score ($S_{pres} \in [0, 100]$):**
   $$S_{pres} = \min\left(100.0, \frac{\text{active\_faces}}{\text{expected\_participants}} \times 100.0\right)$$
2. **Movement Dynamics Score ($S_{mov} \in [0, 100]$):**
   - Natural micro-movement (nodding, taking notes, looking at board) yields 95.
   - Frozen stillness (< 1 px) yields 35.
   - Violent erratic motion (> 80 px) yields 55.
3. **Audio Participation Score ($S_{aud} \in [0, 100]$):**
   - Active vocal speech with healthy SNR yields 75–100.
   - Attentive quiet background yields 60.
   - High distracting background noise without speech yields 30.

### Fused Scoring

- **Bimodal Session:**
  $$\text{Score} = \frac{w_v S_{pres} + w_m S_{mov} + w_a S_{aud}}{w_v + w_m + w_a} \quad (\text{default weights: } w_v=0.40, w_m=0.25, w_a=0.35)$$
- **Vision-Only Session:**
  $$\text{Score} = \frac{w_v S_{pres} + w_m S_{mov}}{w_v + w_m} \quad (\text{confidence: } 0.75)$$
- **Audio-Only Session:**
  $$\text{Score} = S_{aud} \quad (\text{confidence: } 0.70)$$

*Ethical Disclaimer: This index is a heuristic indicator of physical and acoustic presence. It does not measure cognitive understanding, learning capacity, or student intelligence.*

---

## 18. Recommendation Logic

Located in [`agent/recommendations.py`](file:///C:/Projects/ClassPulseAI/agent/recommendations.py):

| Trigger Pattern | Threshold Condition | Pedagogical Suggestion | Exact Metrics Captured |
| :--- | :--- | :--- | :--- |
| **Sustained Low Engagement** | Score < 38 for $\ge$ 6.0s | Introduce think-pair-share or concept check poll. | Mean Score, Threshold, Duration (s) |
| **Elevated Ambient Noise** | RMS $\ge$ 0.08 without speech for $\ge$ 6.0s | Inspect microphone placement and room acoustics. | Avg Noise RMS, Threshold RMS, Duration |
| **Sudden Engagement Drop** | Score drop $\ge$ 25.0 points in one bin | Check if topic transition was abrupt or slide was unclear. | Score Delta, Previous Score, Current Score |
| **Participation Surge** | Sustained score $\ge$ 80 for $\ge$ 4.0s | Note instructional strategy as an effective repeatable practice. | Mean High Score, Duration (s) |
| **Insufficient Data** | Session duration < 5.0 seconds | Allow session to run longer for meaningful trend analysis. | Session Duration (s), Min Required (s) |

*Contradiction Resolution: Overlapping contradictory suggestions (e.g. low engagement vs. participation surge) are automatically resolved and deduplicated.*

---

## 19. Analytics, Archive & Export Instructions

From the **📥 Reports & Exports** view in the Streamlit dashboard or the CLI:

1. **Printable HTML/PDF Report:** Downloads a self-contained, beautifully styled HTML report with complete tables, KPI cards, and print styling.
2. **Attendance CSV:** Exports tracking IDs, presence seconds, required seconds, session percentage, first seen, last seen, absence reason, and attendance status.
3. **Engagement Timeline CSV:** Exports timestamped scores, level ratings, confidence values, and formula strings.
4. **Suggestions CSV:** Exports all triggered recommendations, observations, rationale, and exact metric dictionaries.
5. **Full JSON (Schema v1.1.0):** Standardized JSON session summary including schema version and metadata.
6. **Session Comparer:** The **📊 Session Archive & Compare** view enables side-by-side metric comparison of any two past sessions stored in `data/sessions/`.

---

## 20. Testing & Self-Test Instructions

Run the 25 automated pytest tests:

```powershell
.\venv\Scripts\python -m pytest tests -v
```

Run the system diagnostic self-test:

```powershell
.\venv\Scripts\python cli.py --selftest
```

---

## 21. Privacy & Ethical Safeguards

1. **Zero External API Dependency:** No recordings or student data are ever sent over the internet or uploaded to cloud APIs.
2. **Anonymous Ephemeral Tracking:** No facial embeddings, biometric templates, or real-world names are created or stored. Identifiers are discarded after session termination.
3. **Transparent Heuristics:** Engagement scores are explainable mathematical combinations of observable activity. The system makes no psychological or intelligence claims.
4. **Teacher-Centric Guidance:** Recommendations are non-prescriptive suggestions meant to assist educator reflection, not evaluate teacher competency.

---

## 22. Known Limitations

- **Lighting & Angles:** Extreme backlighting or steep camera angles can reduce face detection recall (mitigated by CLAHE contrast enhancement).
- **Microphone Placement:** Microphones positioned next to laptop cooling fans will experience an elevated ambient noise floor.
- **Passive Attentiveness:** Students quietly reading textbooks or writing notes may exhibit low physical movement and silence, resulting in low movement scores despite genuine cognitive focus. Instructors should consider this context when reviewing suggestions.

---

## 23. Troubleshooting Instructions

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`PermissionError` on Matplotlib cache** | Default Windows AppData directory restricted | Fixed automatically: `config.py` sets `MPLCONFIGDIR = data/.mpl_cache`. |
| **Webcam fails to open** | Camera 0 busy or blocked by Windows Privacy settings | Check Windows Settings > Privacy > Camera; allow desktop apps access. |
| **Audio extraction fails on video** | Unsupported audio codec in video container | Ensure `imageio-ffmpeg` is installed (`pip install imageio-ffmpeg`). |
| **`Execution_Policies` warning on venv activation** | PowerShell script execution disabled | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your terminal. |
| **Threshold Warning in UI** | Required seconds exceed session length | Adjust required presence ratio slider or decrease absolute seconds override. |

---

*ClassPulse AI — Developed independently for educational technology and multimodal AI research.*
