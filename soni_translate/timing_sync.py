"""
Timing and sync fixes for TTS segments.
Time-stretches, compresses, and validates TTS to fit original durations.
"""

import os
import re
import numpy as np
import soundfile as sf
from .logging_setup import logger


MAX_TIME_STRETCH_RATIO = 1.35
MAX_TIME_COMPRESS_RATIO = 0.75
HINDI_CHARS_PER_SECOND = 14


def fit_audio_to_duration(
    tts_path, target_duration_sec, output_path,
    max_stretch=MAX_TIME_STRETCH_RATIO,
):
    """
    Stretches or compresses TTS audio to fit the original segment duration.
    max_stretch=1.35 means max 35% slower — beyond this it sounds unnatural.
    """
    try:
        import pyrubberband as pyrb
        has_pyrubberband = True
    except ImportError:
        has_pyrubberband = False

    try:
        audio, sr = sf.read(tts_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        current_duration = len(audio) / sr
        ratio = target_duration_sec / current_duration

        # Cap stretch ratio to avoid unnatural speech
        ratio = min(ratio, max_stretch)
        ratio = max(ratio, MAX_TIME_COMPRESS_RATIO)

        if abs(ratio - 1.0) > 0.03:  # only stretch if difference > 3%
            if has_pyrubberband:
                audio = pyrb.time_stretch(audio, sr, 1.0 / ratio)
                logger.debug(
                    f"Time-stretched by {ratio:.2f}x using pyrubberband"
                )
            else:
                # Fallback: simple resampling
                target_samples = int(len(audio) / ratio)
                indices = np.linspace(0, len(audio) - 1, target_samples)
                audio = np.interp(
                    np.arange(target_samples),
                    np.arange(len(audio)),
                    audio,
                )
                logger.debug(
                    f"Time-stretched by {ratio:.2f}x using resampling"
                )

        # Pad with silence if still shorter than target
        target_samples = int(target_duration_sec * sr)
        if len(audio) < target_samples:
            silence = np.zeros(target_samples - len(audio))
            audio = np.concatenate([audio, silence])
        elif len(audio) > target_samples:
            audio = audio[:target_samples]

        sf.write(output_path, audio, sr)
        return output_path
    except Exception as e:
        logger.error(f"Time-stretch failed: {e}")
        import shutil
        shutil.copy2(tts_path, output_path)
        return output_path


def check_and_split_segment(
    hindi_text, duration_sec, chars_per_second=HINDI_CHARS_PER_SECOND
):
    """
    Estimates if Hindi text will fit in the given duration.
    Hindi TTS averages ~14 chars/sec. If too long, split at natural
    punctuation or clause boundaries.

    Returns:
        list: list of text parts (1 if fits, 2+ if split needed)
    """
    if not hindi_text or not hindi_text.strip():
        return [hindi_text]

    estimated_duration = len(hindi_text) / chars_per_second

    if estimated_duration <= duration_sec * 1.3:  # 30% tolerance
        return [hindi_text]

    # Try to split at Hindi/Urdu punctuation
    parts = re.split(r'[,،؛!?\n]', hindi_text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        # Merge short parts to avoid too many tiny segments
        merged = []
        current = ""
        for part in parts:
            if current and len(current + part) > 80:
                merged.append(current)
                current = part
            else:
                current = current + " " + part if current else part
        if current:
            merged.append(current)
        return merged

    # Hard split at midpoint word boundary
    words = hindi_text.split()
    mid = len(words) // 2
    if mid == 0:
        return [hindi_text]
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def verify_alignment(
    dubbed_audio_path, expected_segments, language="hi"
):
    """
    Run WhisperX forced alignment on the dubbed audio to verify timing.
    Logs any segments that drifted more than 0.3 seconds.
    """
    try:
        import whisperx
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = whisperx.load_model(
            "base", device=device, language=language
        )
        result = model.transcribe(dubbed_audio_path)

        alignment_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        aligned = whisperx.align(
            result["segments"], alignment_model, metadata,
            dubbed_audio_path, device=device,
        )

        drift_log = []
        aligned_segs = aligned.get("segments", [])

        for i, expected in enumerate(expected_segments):
            if i >= len(aligned_segs):
                break
            actual = aligned_segs[i]
            expected_start = expected.get("start", 0)
            actual_start = actual.get("start", expected_start)
            drift = abs(expected_start - actual_start)

            if drift > 0.3:
                drift_log.append({
                    "segment": i,
                    "drift_seconds": round(drift, 2),
                    "text": expected.get("text", "")[:50],
                })

        if drift_log:
            logger.warning(
                f"[Alignment] {len(drift_log)} segments drifted >0.3s:"
            )
            for entry in drift_log:
                logger.warning(
                    f"  Seg {entry['segment']}: "
                    f"{entry['drift_seconds']}s — {entry['text']}"
                )
        else:
            logger.info("[Alignment] All segments within 0.3s tolerance")

        return drift_log
    except ImportError:
        logger.warning("whisperx not available, skipping alignment verification")
        return []
    except Exception as e:
        logger.error(f"Alignment verification failed: {e}")
        return []
