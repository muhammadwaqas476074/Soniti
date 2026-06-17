"""
Audio enhancement: loudness normalization and room tone matching.
Makes TTS segments match original speaker loudness and environment.
"""

import os
import numpy as np
import soundfile as sf
from .logging_setup import logger


ROOM_TONE_MIX = 0.15  # 15% room ambience blend


def normalize_segment_loudness(
    original_wav_path, tts_wav_path, output_path
):
    """
    Measures integrated loudness (LUFS) of original speaker segment,
    then adjusts TTS segment to match it exactly.
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning("pyloudnorm not available, skipping loudness normalization")
        import shutil
        shutil.copy2(tts_wav_path, output_path)
        return output_path

    try:
        orig_audio, orig_sr = sf.read(original_wav_path)
        tts_audio, tts_sr = sf.read(tts_wav_path)

        # Ensure mono for loudness measurement
        if len(orig_audio.shape) > 1:
            orig_audio = orig_audio.mean(axis=1)
        if len(tts_audio.shape) > 1:
            tts_audio = tts_audio.mean(axis=1)

        meter = pyln.Meter(orig_sr)

        orig_loudness = meter.integrated_loudness(orig_audio)
        tts_loudness = meter.integrated_loudness(tts_audio)

        if np.isinf(orig_loudness) or np.isinf(tts_loudness):
            logger.warning("Could not measure loudness, skipping normalization")
            import shutil
            shutil.copy2(tts_wav_path, output_path)
            return output_path

        # Normalize TTS to match original loudness
        normalized_tts = pyln.normalize.loudness(
            tts_audio, tts_loudness, orig_loudness
        )

        # Clip to prevent distortion
        normalized_tts = np.clip(normalized_tts, -1.0, 1.0)

        sf.write(output_path, normalized_tts, tts_sr)
        logger.debug(
            f"Loudness matched: orig={orig_loudness:.1f} LUFS, "
            f"tts={tts_loudness:.1f} -> {orig_loudness:.1f} LUFS"
        )
        return output_path
    except Exception as e:
        logger.error(f"Loudness normalization failed: {e}")
        import shutil
        shutil.copy2(tts_wav_path, output_path)
        return output_path


def extract_room_sample(
    vocals_path, diarization_segments, output_path, duration=0.5
):
    """
    Finds a gap between speech segments to capture room tone.
    Returns path to room sample or None if no suitable gap found.
    """
    try:
        audio, sr = sf.read(vocals_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Build set of speech sample indices (for fast lookup)
        speech_ranges = []
        for seg in diarization_segments:
            start = int(float(seg.get("start", 0)) * sr)
            end = int(float(seg.get("end", 0)) * sr)
            speech_ranges.append((start, end))

        # Find first 0.5s gap with no speech
        gap_len = int(duration * sr)
        total_samples = len(audio)

        for gap_start in range(0, total_samples - gap_len, int(0.1 * sr)):
            gap_end = gap_start + gap_len
            is_speech = False
            for s_start, s_end in speech_ranges:
                if gap_start < s_end and gap_end > s_start:
                    is_speech = True
                    break
            if not is_speech:
                # Check if this region is quiet (near-silent)
                region = audio[gap_start:gap_end]
                rms = np.sqrt(np.mean(region ** 2))
                if rms < 0.1:  # relatively quiet
                    sf.write(output_path, region, sr)
                    logger.info(
                        f"Room sample extracted: {duration}s "
                        f"at {gap_start/sr:.1f}s"
                    )
                    return output_path

        logger.warning("No suitable room tone gap found")
        return None
    except Exception as e:
        logger.error(f"Room sample extraction failed: {e}")
        return None


def apply_room_tone(
    tts_wav_path, room_sample_path, output_path, mix=ROOM_TONE_MIX
):
    """
    Applies subtle room ambience from original recording to TTS audio.
    mix=0.15 means 15% room tone blended in — keeps it subtle.
    """
    if room_sample_path is None or not os.path.exists(room_sample_path):
        import shutil
        shutil.copy2(tts_wav_path, output_path)
        return output_path

    try:
        from scipy.signal import fftconvolve

        tts_audio, sr = sf.read(tts_wav_path)
        room_audio, _ = sf.read(room_sample_path)

        if len(tts_audio.shape) > 1:
            tts_audio = tts_audio.mean(axis=1)
        if len(room_audio.shape) > 1:
            room_audio = room_audio.mean(axis=1)

        # Use max 0.5s of room audio as impulse response
        max_room_samples = int(0.5 * sr)
        if len(room_audio) > max_room_samples:
            room_audio = room_audio[:max_room_samples]

        # Normalize room audio
        room_peak = np.max(np.abs(room_audio))
        if room_peak > 0:
            room_audio = room_audio / room_peak

        # Convolve for reverb character
        convolved = fftconvolve(tts_audio, room_audio, mode='full')[:len(tts_audio)]
        conv_max = np.max(np.abs(convolved))
        if conv_max > 0:
            convolved = convolved / conv_max

        # Blend: mostly dry TTS + subtle room
        blended = (1 - mix) * tts_audio + mix * convolved
        blended = np.clip(blended, -1.0, 1.0)

        sf.write(output_path, blended, sr)
        logger.debug(f"Room tone applied (mix={mix}): {output_path}")
        return output_path
    except ImportError:
        logger.warning("scipy not available, skipping room tone")
        import shutil
        shutil.copy2(tts_wav_path, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Room tone application failed: {e}")
        import shutil
        shutil.copy2(tts_wav_path, output_path)
        return output_path
