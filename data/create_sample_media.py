"""
Generates synthetic sample audio and video files for offline testing and demonstration.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import soundfile as sf

from config import SAMPLE_MEDIA_DIR, AUDIO_CONFIG


def generate_synthetic_audio(
    output_path: Path = SAMPLE_MEDIA_DIR / "sample_classroom_audio.wav",
    duration_seconds: float = 12.0,
    sample_rate: int = 16000,
) -> Path:
    """
    Generates a 12-second synthetic classroom audio clip:
    - 0-3s: Quiet ambient classroom murmur
    - 3-7s: Simulated speech burst (voiced fundamental 180Hz + formants + syllable modulation)
    - 7-9s: Quiet pause
    - 9-12s: Second speech activity segment
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

    # Ambient low-level pink/white noise (RMS ~ 0.003)
    ambient_noise = np.random.normal(0, 0.003, num_samples).astype(np.float32)

    # Simulated speech carrier (harmonics at 180Hz, 360Hz, 540Hz, 720Hz)
    f0 = 180.0
    carrier = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.15 * np.sin(2 * np.pi * 3 * f0 * t) +
        0.05 * np.sin(2 * np.pi * 4 * f0 * t)
    )

    # Syllabic envelope modulation (3 Hz syllable rate)
    syllable_mod = np.clip(np.sin(2 * np.pi * 3.5 * t) ** 2, 0, 1)

    # Activity gates: speech active during [3.0s, 7.0s] and [9.0s, 11.5s]
    speech_gate = np.zeros(num_samples, dtype=np.float32)
    mask1 = (t >= 3.0) & (t <= 7.0)
    mask2 = (t >= 9.0) & (t <= 11.5)
    speech_gate[mask1] = 0.35
    speech_gate[mask2] = 0.40

    speech_signal = carrier * syllable_mod * speech_gate

    # Combine ambient noise and speech signal
    audio = (ambient_noise + speech_signal).astype(np.float32)
    # Normalize
    audio = audio / np.max(np.abs(audio) + 1e-6) * 0.85

    sf.write(str(output_path), audio, sample_rate)
    print(f"Generated synthetic audio: {output_path} ({duration_seconds}s)")
    return output_path


def generate_synthetic_video(
    output_path: Path = SAMPLE_MEDIA_DIR / "sample_classroom_video.mp4",
    duration_seconds: float = 8.0,
    fps: int = 15,
    width: int = 480,
    height: int = 360,
) -> Path:
    """
    Generates an 8-second synthetic MP4 video containing simulated participant activity.
    Uses realistic facial rendering to allow Haar cascade / OpenCV face detection.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(duration_seconds * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    def draw_realistic_face(img, cx, cy, radius=45):
        # Skin tone base
        skin_color = (185, 215, 240)  # BGR skin tone
        cv2.circle(img, (cx, cy), radius, skin_color, -1)
        # Hair
        hair_color = (40, 45, 60)
        cv2.ellipse(img, (cx, cy - 15), (radius, int(radius * 0.7)), 0, 180, 360, hair_color, -1)
        # Eyebrows
        cv2.line(img, (cx - 24, cy - 16), (cx - 8, cy - 18), (30, 30, 30), 2)
        cv2.line(img, (cx + 8, cy - 18), (cx + 24, cy - 16), (30, 30, 30), 2)
        # Eyes
        cv2.circle(img, (cx - 16, cy - 8), 6, (255, 255, 255), -1)
        cv2.circle(img, (cx + 16, cy - 8), 6, (255, 255, 255), -1)
        cv2.circle(img, (cx - 16, cy - 8), 3, (60, 40, 20), -1)
        cv2.circle(img, (cx + 16, cy - 8), 3, (60, 40, 20), -1)
        # Nose
        cv2.line(img, (cx, cy - 4), (cx - 2, cy + 8), (140, 170, 200), 2)
        cv2.line(img, (cx - 2, cy + 8), (cx + 4, cy + 8), (140, 170, 200), 2)
        # Mouth
        cv2.ellipse(img, (cx, cy + 20), (14, 6), 0, 0, 180, (50, 50, 180), 2)

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        # Background: classroom blackboard style
        frame = np.full((height, width, 3), (45, 55, 45), dtype=np.uint8)

        # Classroom border
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (70, 80, 70), 2)
        cv2.putText(
            frame,
            "ClassPulse AI - Classroom Simulation",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 220, 200),
            1,
            cv2.LINE_AA,
        )

        # Participant 1: Present throughout the session with slight natural sway
        p1_x = int(140 + 8 * np.sin(2 * np.pi * 0.4 * t))
        p1_y = int(200 + 4 * np.cos(2 * np.pi * 0.4 * t))
        draw_realistic_face(frame, p1_x, p1_y, radius=42)

        # Participant 2: Present only between 1.5s and 6.5s (leaves early/arrives late)
        if 1.5 <= t <= 6.5:
            p2_x = int(340 + 6 * np.cos(2 * np.pi * 0.3 * t))
            p2_y = int(210 + 3 * np.sin(2 * np.pi * 0.3 * t))
            draw_realistic_face(frame, p2_x, p2_y, radius=40)

        out.write(frame)

    out.release()
    print(f"Generated synthetic video: {output_path} ({duration_seconds}s)")
    return output_path


if __name__ == "__main__":
    generate_synthetic_audio()
    generate_synthetic_video()
