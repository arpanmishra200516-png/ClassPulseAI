"""
Audio Processor for ClassPulse AI.
Handles loading audio files (WAV, MP3, etc.), extracting audio tracks from video files,
resampling, mono conversion, segmentation into timestamped analysis windows,
and defensive handling of missing, silent, or invalid audio data.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger("ClassPulseAI.audio")

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = None

if not FFMPEG_EXE or not os.path.exists(FFMPEG_EXE):
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        FFMPEG_EXE = system_ffmpeg

from config import AUDIO_CONFIG


@dataclass
class AudioSegmentData:
    """Represents a discrete, timestamped audio window."""
    window_idx: int
    start_time: float
    end_time: float
    samples: np.ndarray
    sample_rate: int
    duration: float
    is_silent: bool


class AudioProcessor:
    """
    Extracts and standardizes audio from audio or video files into discrete timestamped windows.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        window_duration: float = 1.0,
        hop_duration: float = 0.5,
    ):
        self.target_sample_rate = target_sample_rate
        self.window_duration = window_duration
        self.hop_duration = hop_duration

    def check_has_audio_stream(self, file_path: str) -> Tuple[bool, str]:
        """
        Inspects media file to determine if an audio stream exists.
        Returns (has_audio_stream, reason_or_stream_info).
        """
        path = Path(file_path)
        if not path.exists() or path.stat().st_size == 0:
            return False, "File does not exist or is empty"

        audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
        if path.suffix.lower() in audio_extensions:
            return True, "Audio file container"

        if FFMPEG_EXE and os.path.exists(FFMPEG_EXE):
            try:
                proc = subprocess.run(
                    [FFMPEG_EXE, "-i", str(path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if "Audio:" in proc.stderr:
                    return True, "Audio stream detected in video container"
                else:
                    return False, "This video has no audio track - audio metrics skipped"
            except Exception as e:
                logger.warning("[AudioProcessor] Error probing audio stream with ffmpeg: %s", e)

        return True, "Assuming audio stream present (probe unavailable)"

    def extract_audio_from_video(self, video_path: str, output_wav_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts the audio track from a video file into a 16kHz mono WAV file using local FFMPEG.
        Returns (output_wav_path, error_message).
        """
        video_file = Path(video_path)
        if not video_file.exists() or video_file.stat().st_size == 0:
            return None, "Video file does not exist or is empty"

        if FFMPEG_EXE is None or not os.path.exists(FFMPEG_EXE):
            msg = "FFmpeg executable not found (imageio-ffmpeg / system ffmpeg unavailable)."
            logger.warning("[AudioProcessor] %s", msg)
            return None, msg

        if output_wav_path is None:
            temp_dir = tempfile.gettempdir()
            output_wav_path = str(Path(temp_dir) / f"{video_file.stem}_extracted_{os.getpid()}.wav")

        command = [
            FFMPEG_EXE,
            "-y",                     # Overwrite output
            "-i", str(video_file),     # Input video
            "-vn",                    # Discard video track
            "-acodec", "pcm_s16le",   # 16-bit PCM WAV
            "-ar", str(self.target_sample_rate), # Resample
            "-ac", "1",               # Mono
            str(output_wav_path),
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
                text=True,
            )
            if result.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100:
                return output_wav_path, None
            else:
                err_detail = result.stderr.strip() if result.stderr else f"Exit code {result.returncode}"
                return None, f"FFmpeg extraction failed: {err_detail}"
        except Exception as e:
            return None, f"FFmpeg extraction exception: {e}"

    def load_audio_with_status(
        self,
        file_path: str,
    ) -> Tuple[Optional[np.ndarray], int, float, str, Optional[str], Optional[str]]:
        """
        Loads audio from audio or video file with detailed extraction status reporting.
        Tries extraction decoders in order:
        1. FFmpeg via imageio-ffmpeg or subprocess
        2. Librosa / Soundfile direct
        3. Audioread fallback

        Returns (samples, sample_rate, duration_seconds, status, message, path_worked).
        status: 'ANALYZED', 'NO_AUDIO_TRACK', 'EXTRACTION_FAILED', or 'UNAVAILABLE'
        """
        path = Path(file_path)
        if not path.exists() or path.stat().st_size == 0:
            return None, self.target_sample_rate, 0.0, "UNAVAILABLE", "File not found or empty", None

        video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm", ".wmv"}
        is_video = path.suffix.lower() in video_extensions

        if is_video:
            has_stream, stream_msg = self.check_has_audio_stream(str(path))
            if not has_stream:
                logger.info("[AudioProcessor] %s", stream_msg)
                return None, self.target_sample_rate, 0.0, "NO_AUDIO_TRACK", stream_msg, None

        actual_path = str(path)
        extracted_temp_file = None
        path_worked = None
        data = None
        sr = self.target_sample_rate

        # -------------------------------------------------------------
        # 1. Try ffmpeg extraction (primary for video)
        # -------------------------------------------------------------
        ffmpeg_err = None
        if is_video:
            extracted_path, ffmpeg_err = self.extract_audio_from_video(str(path))
            if extracted_path:
                try:
                    data, sr = sf.read(extracted_path, dtype="float32")
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    if sr != self.target_sample_rate:
                        data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sample_rate)
                        sr = self.target_sample_rate
                    path_worked = "ffmpeg (imageio-ffmpeg / subprocess)"
                    logger.info("[AudioProcessor] Audio extracted successfully via %s", path_worked)
                except Exception as e:
                    ffmpeg_err = str(e)
                finally:
                    extracted_temp_file = extracted_path

        # -------------------------------------------------------------
        # 2. Try librosa / soundfile direct load
        # -------------------------------------------------------------
        librosa_err = None
        if data is None:
            try:
                data, sr = sf.read(actual_path, dtype="float32")
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if sr != self.target_sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sample_rate)
                    sr = self.target_sample_rate
                path_worked = "soundfile direct"
                logger.info("[AudioProcessor] Audio loaded successfully via %s", path_worked)
            except Exception as e_sf:
                try:
                    data, sr = librosa.load(actual_path, sr=self.target_sample_rate, mono=True)
                    path_worked = "librosa direct"
                    logger.info("[AudioProcessor] Audio loaded successfully via %s", path_worked)
                except Exception as e_lib:
                    librosa_err = f"soundfile error: {e_sf}; librosa error: {e_lib}"

        # -------------------------------------------------------------
        # 3. Try audioread fallback
        # -------------------------------------------------------------
        audioread_err = None
        if data is None:
            try:
                import audioread
                with audioread.audio_open(actual_path) as f:
                    sr_native = f.samplerate
                    nchannels = f.channels
                    buf = []
                    for chunk in f:
                        buf.append(chunk)
                    if buf:
                        raw = b"".join(buf)
                        data_arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        if nchannels > 1:
                            data_arr = data_arr.reshape(-1, nchannels).mean(axis=1)
                        if sr_native != self.target_sample_rate:
                            data = librosa.resample(data_arr, orig_sr=sr_native, target_sr=self.target_sample_rate)
                        else:
                            data = data_arr
                        sr = self.target_sample_rate
                        path_worked = "audioread fallback"
                        logger.info("[AudioProcessor] Audio loaded successfully via %s", path_worked)
            except Exception as e_ar:
                audioread_err = str(e_ar)

        # Clean up temporary extracted WAV
        if extracted_temp_file and os.path.exists(extracted_temp_file):
            try:
                os.remove(extracted_temp_file)
            except Exception:
                pass

        if data is not None and len(data) > 0:
            duration = float(len(data)) / float(sr)
            return data, sr, duration, "ANALYZED", f"Loaded via {path_worked}", path_worked

        if is_video:
            err_msg = (
                f"Audio extraction failed. Errors: ffmpeg: {ffmpeg_err}; "
                f"librosa: {librosa_err}; audioread: {audioread_err}"
            )
            logger.error("[AudioProcessor] %s", err_msg)
            return None, self.target_sample_rate, 0.0, "EXTRACTION_FAILED", err_msg, None

        return None, self.target_sample_rate, 0.0, "UNAVAILABLE", "Failed to decode audio file", None

    def load_audio(
        self,
        file_path: str,
    ) -> Tuple[Optional[np.ndarray], int, float]:
        """
        Loads an audio or video file, converts to mono, resamples to target_sample_rate,
        and normalizes amplitudes to [-1.0, 1.0].
        Returns (samples, sample_rate, duration_seconds).
        """
        samples, sr, duration, status, msg, path_worked = self.load_audio_with_status(file_path)
        return samples, sr, duration

    def segment_audio(
        self,
        samples: Optional[np.ndarray],
        sample_rate: int,
    ) -> List[AudioSegmentData]:
        """
        Segments a continuous 1D audio array into fixed timestamped windows.
        Handles empty or silent inputs defensively.
        """
        if samples is None or len(samples) == 0 or sample_rate <= 0:
            return []

        window_samples = int(self.window_duration * sample_rate)
        hop_samples = int(self.hop_duration * sample_rate)

        if window_samples <= 0 or hop_samples <= 0:
            return []

        total_samples = len(samples)
        segments: List[AudioSegmentData] = []
        window_idx = 0

        for start_idx in range(0, total_samples, hop_samples):
            end_idx = min(start_idx + window_samples, total_samples)
            chunk = samples[start_idx:end_idx]

            # If the last chunk is too short (less than 20% of window), break
            if len(chunk) < int(0.2 * window_samples):
                break

            # Pad chunk to window_samples if slightly short
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")

            start_sec = float(start_idx) / float(sample_rate)
            end_sec = start_sec + self.window_duration
            actual_dur = float(len(chunk)) / float(sample_rate)

            # Check if chunk is effectively silent
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_silent = (rms < AUDIO_CONFIG["min_energy_threshold"])

            segments.append(
                AudioSegmentData(
                    window_idx=window_idx,
                    start_time=round(start_sec, 3),
                    end_time=round(end_sec, 3),
                    samples=chunk,
                    sample_rate=sample_rate,
                    duration=actual_dur,
                    is_silent=is_silent,
                )
            )
            window_idx += 1

        return segments
