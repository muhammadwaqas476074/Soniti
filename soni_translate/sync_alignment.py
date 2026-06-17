"""
Anchor-based audio sync alignment.
Detects exact speech boundaries in original vocal segments
and aligns translated TTS to match precisely.
"""

import numpy as np
import soundfile as sf
import subprocess
import os
from .logging_setup import logger


def detect_speech_boundaries(audio_path, sr=16000, energy_threshold=0.02):
    """
    Finds the EXACT sample where speech starts and ends
    in an original vocal segment — not the subtitle timestamp,
    the actual audio activity.
    """
    try:
        import librosa
        y, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        logger.error(f"[Sync] Failed to load {audio_path}: {e}")
        return 0.0, None

    # RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    hop_duration = 128 / sr

    # Find first frame above threshold
    speech_frames = np.where(rms > energy_threshold)[0]
    if len(speech_frames) == 0:
        return 0.0, librosa.get_duration(y=y, sr=sr)

    speech_start = float(speech_frames[0] * hop_duration)
    speech_end = float(speech_frames[-1] * hop_duration)

    # Add small margin (20ms) to avoid clipping
    speech_start = max(0.0, speech_start - 0.02)
    speech_end = min(librosa.get_duration(y=y, sr=sr), speech_end + 0.02)

    return speech_start, speech_end


def align_tts_to_original(
    tts_path,
    original_vocals_path,
    segment_start,
    segment_end,
    output_path,
    temp_dir="audio/sync_tmp",
    sr=22050,
):
    """
    Places TTS audio so it starts exactly when the original
    speaker starts, not just at the subtitle timestamp.

    Steps:
      1. Extract original segment
      2. Detect exact speech start offset within segment
      3. Build output: [pre-speech silence] + [TTS] + [post-speech silence]
    """
    try:
        import librosa
        import pyrubberband as pyrb
    except ImportError:
        raise ImportError("pip install librosa pyrubberband")

    os.makedirs(temp_dir, exist_ok=True)
    seg_duration = segment_end - segment_start

    # 1. Extract original segment
    orig_seg_path = os.path.join(
        temp_dir, f"orig_{segment_start:.3f}.wav"
    )
    cmd = (
        f'ffmpeg -y -i "{original_vocals_path}" '
        f"-ss {segment_start:.3f} -t {seg_duration:.3f} "
        f'-ar {sr} -ac 1 "{orig_seg_path}" -loglevel error'
    )
    subprocess.run(cmd, shell=True)

    if not os.path.exists(orig_seg_path):
        import shutil
        shutil.copy(tts_path, output_path)
        return output_path

    # 2. Detect exact speech start/end in original
    speech_start_offset, speech_end_offset = detect_speech_boundaries(
        orig_seg_path, sr=sr
    )
    speech_duration = speech_end_offset - speech_start_offset

    logger.debug(
        f"[Sync] Seg {segment_start:.2f}s: "
        f"speech at +{speech_start_offset:.3f}s "
        f"(duration {speech_duration:.3f}s)"
    )

    # 3. Load and time-stretch TTS to fit speech window
    y_tts, tts_sr = librosa.load(tts_path, sr=sr)
    tts_duration = librosa.get_duration(y=y_tts, sr=sr)

    if speech_duration > 0.1 and abs(tts_duration - speech_duration) > 0.05:
        stretch_ratio = speech_duration / tts_duration
        stretch_ratio = float(np.clip(stretch_ratio, 0.7, 1.4))
        y_tts = pyrb.time_stretch(y_tts, sr, 1.0 / stretch_ratio)
        logger.debug(
            f"[Sync] Stretch x{stretch_ratio:.2f} to fit {speech_duration:.2f}s window"
        )

    # 4. Build output with correct silence padding
    total_samples = int(seg_duration * sr)
    pre_silence_samples = int(speech_start_offset * sr)
    tts_samples = len(y_tts)
    post_silence_samples = max(
        0, total_samples - pre_silence_samples - tts_samples
    )

    pre_silence = np.zeros(pre_silence_samples)
    post_silence = np.zeros(post_silence_samples)
    y_out = np.concatenate([pre_silence, y_tts, post_silence])

    # Trim/pad to exact segment duration
    if len(y_out) > total_samples:
        fade = int(0.03 * sr)
        if fade > 0 and total_samples > fade:
            y_out[total_samples - fade:total_samples] *= np.linspace(1, 0, fade)
        y_out = y_out[:total_samples]
    elif len(y_out) < total_samples:
        y_out = np.concatenate([y_out, np.zeros(total_samples - len(y_out))])

    sf.write(output_path, y_out, sr)

    # Cleanup
    try:
        os.remove(orig_seg_path)
    except OSError:
        pass

    return output_path
