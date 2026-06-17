"""
Per-speaker voice extraction and cloning.
Extracts reference audio samples per speaker for XTTS/OpenVoice cloning.
"""

import os
import numpy as np
import soundfile as sf
from .logging_setup import logger


MIN_SPEAKER_SAMPLE_DURATION = 4.0
MAX_REFERENCE_DURATION = 12.0


def extract_speaker_samples(
    diarization_segments, vocals_path,
    output_dir="speaker_samples", min_duration=MIN_SPEAKER_SAMPLE_DURATION,
):
    """
    For each unique speaker in diarization output, collect their longest
    clean audio segments and concatenate to form a reference sample.

    Args:
        diarization_segments: list of dicts with 'speaker', 'start', 'end'
        vocals_path: path to isolated vocals WAV
        output_dir: where to save reference samples
        min_duration: minimum seconds of audio needed for cloning

    Returns:
        dict: { "SPEAKER_00": "/path/to/sample.wav", ... }
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(vocals_path):
        logger.error(f"Vocals file not found: {vocals_path}")
        return {}

    audio_data, sr = sf.read(vocals_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)  # mono

    speaker_audio = {}  # speaker_id -> list of (audio_chunk, duration)

    for segment in diarization_segments:
        spk = segment.get("speaker", "SPEAKER_00")
        start = float(segment.get("start", 0))
        end = float(segment.get("end", 0))
        duration = end - start

        if duration < 1.0:
            continue

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        end_sample = min(end_sample, len(audio_data))

        if start_sample >= end_sample:
            continue

        chunk = audio_data[start_sample:end_sample]

        if spk not in speaker_audio:
            speaker_audio[spk] = []
        speaker_audio[spk].append((chunk, duration))

    speaker_sample_paths = {}

    for spk, chunks in speaker_audio.items():
        # Sort by duration (longest first) and pick best segments
        chunks.sort(key=lambda x: x[1], reverse=True)
        total_duration = 0
        selected_chunks = []

        for chunk, dur in chunks:
            if total_duration >= MAX_REFERENCE_DURATION:
                break
            selected_chunks.append(chunk)
            total_duration += dur

        if not selected_chunks:
            continue

        combined = np.concatenate(selected_chunks)
        max_samples = int(MAX_REFERENCE_DURATION * sr)
        combined = combined[:max_samples]

        # Normalize audio
        peak = np.max(np.abs(combined))
        if peak > 0:
            combined = combined / peak * 0.95

        out_path = os.path.join(output_dir, f"{spk}_reference.wav")
        sf.write(out_path, combined, sr)
        speaker_sample_paths[spk] = out_path

        logger.info(
            f"[Speaker Clone] {spk}: {len(combined)/sr:.1f}s "
            f"reference saved -> {out_path}"
        )

    # Log warning for speakers with insufficient audio
    for spk, chunks in speaker_audio.items():
        total = sum(d for _, d in chunks)
        if total < min_duration:
            logger.warning(
                f"[Speaker Clone] {spk}: only {total:.1f}s audio "
                f"(need {min_duration}s for reliable cloning). "
                f"Voice cloning may be less accurate."
            )

    return speaker_sample_paths


def get_speaker_reference(
    speaker_id, speaker_sample_paths, fallback_path=None
):
    """
    Get the reference wav path for a specific speaker.
    Falls back to fallback_path if speaker sample not found.
    """
    ref = speaker_sample_paths.get(speaker_id)
    if ref and os.path.exists(ref):
        return ref
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path
    return None


def apply_tone_color(
    xtts_output_path, speaker_ref_path, output_path,
    ckpt_converter_path=None, tau=0.7, device="cuda",
):
    """
    Apply OpenVoice tone color transfer to match original speaker timbre.

    Args:
        xtts_output_path: path to TTS-generated audio
        speaker_ref_path: path to original speaker reference audio
        output_path: path to save tone-matched output
        ckpt_converter_path: path to OpenVoice checkpoint
        tau: transfer strength (0.7 = balanced, lower = more like original)
    """
    try:
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
    except ImportError:
        logger.warning(
            "OpenVoice not available, skipping tone color transfer"
        )
        import shutil
        shutil.copy2(xtts_output_path, output_path)
        return output_path

    if ckpt_converter_path is None:
        # Try default location
        ckpt_converter_path = os.path.join(
            "checkpoints_v2", "converter"
        )
        if not os.path.exists(ckpt_converter_path):
            logger.warning(
                "OpenVoice checkpoint not found, "
                "skipping tone color transfer"
            )
            import shutil
            shutil.copy2(xtts_output_path, output_path)
            return output_path

    try:
        converter = ToneColorConverter(
            ckpt_converter_path, device=device
        )
        target_se, _ = se_extractor.get_se(
            speaker_ref_path, converter, vad=True
        )
        source_se, _ = se_extractor.get_se(
            xtts_output_path, converter, vad=True
        )
        converter.convert(
            audio_src_path=xtts_output_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            tau=tau,
        )
        logger.info(
            f"Tone color applied: {output_path} (tau={tau})"
        )
        return output_path
    except Exception as e:
        logger.error(f"Tone color transfer failed: {e}")
        import shutil
        shutil.copy2(xtts_output_path, output_path)
        return output_path
