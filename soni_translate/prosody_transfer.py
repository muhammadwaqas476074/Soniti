"""
Per-segment prosody transfer.
Adapts TTS output to match the original speaker's
pitch contour, energy envelope, and speaking rate
for each individual segment.
"""

import numpy as np
import soundfile as sf
import subprocess
import os
from .logging_setup import logger


def analyze_segment(audio_path, sr=16000):
    """
    Extract prosodic features from an original vocal segment.
    Returns a dict of features to transfer to TTS output.
    """
    try:
        import librosa
        from scipy.signal import find_peaks
    except ImportError:
        raise ImportError("pip install librosa scipy")

    y, _ = librosa.load(audio_path, sr=sr)

    # Pitch (F0) contour
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr,
        frame_length=1024, hop_length=256,
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    f0_median = float(np.median(f0_valid)) if len(f0_valid) else None
    f0_std = float(np.std(f0_valid)) if len(f0_valid) else 0.0

    # Energy envelope
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    energy_peak = float(np.max(rms))

    # Speaking rate estimate
    duration = librosa.get_duration(y=y, sr=sr)
    peaks, _ = find_peaks(rms, height=energy_mean * 0.5, distance=4)
    syllable_rate = len(peaks) / duration if duration > 0 else 3.0

    # Emotion proxy from pitch variance + energy
    emotion_score = (f0_std / 50.0) * (energy_mean / 0.05) if energy_mean > 0 else 0.0
    if emotion_score > 1.5:
        emotion = "high"
    elif emotion_score < 0.4:
        emotion = "low"
    else:
        emotion = "neutral"

    # Pause pattern
    silence_mask = rms < (energy_mean * 0.15)
    pause_ratio = float(np.mean(silence_mask))

    return {
        "duration": duration,
        "f0_median": f0_median,
        "f0_std": f0_std,
        "f0_contour": f0,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "energy_peak": energy_peak,
        "syllable_rate": syllable_rate,
        "emotion": emotion,
        "emotion_score": emotion_score,
        "pause_ratio": pause_ratio,
    }


def _match_energy(y_tts, target_energy_mean, sr):
    """Scale TTS volume to match original segment energy."""
    try:
        import librosa
        rms_tts = librosa.feature.rms(y=y_tts, frame_length=1024, hop_length=256)[0]
        tts_energy = float(np.mean(rms_tts))
        if tts_energy > 1e-6:
            gain = target_energy_mean / tts_energy
            gain = np.clip(gain, 0.25, 4.0)
            y_tts = y_tts * gain
        return np.clip(y_tts, -1.0, 1.0)
    except Exception as e:
        logger.warning(f"[Prosody] Energy match failed: {e}")
        return y_tts


def _apply_emotion_dynamics(y_tts, sr, emotion, f0_std):
    """
    Apply dynamic range compression/expansion based on emotion.
    - high emotion: expand dynamics, slight overdrive
    - low emotion: compress dynamics, soften
    - neutral: light compression only
    """
    try:
        if emotion == "high":
            threshold = 0.3
            y_out = np.where(
                np.abs(y_tts) > threshold,
                np.sign(y_tts) * (threshold + (np.abs(y_tts) - threshold) * 1.4),
                y_tts * 0.9,
            )
            return np.clip(y_out, -1.0, 1.0)
        elif emotion == "low":
            rms = np.sqrt(np.mean(y_tts ** 2))
            if rms > 1e-6:
                target_rms = 0.08
                y_tts = y_tts * (target_rms / rms)
            return np.clip(y_tts, -0.6, 0.6)
        else:
            threshold = 0.5
            ratio = 0.7
            y_out = np.where(
                np.abs(y_tts) > threshold,
                np.sign(y_tts) * (threshold + (np.abs(y_tts) - threshold) * ratio),
                y_tts,
            )
            return np.clip(y_out, -1.0, 1.0)
    except Exception as e:
        logger.warning(f"[Prosody] Emotion dynamics failed: {e}")
        return y_tts


def adapt_tts_to_original(
    tts_path,
    original_features,
    output_path,
    sr=22050,
    max_stretch=1.4,
    max_compress=0.7,
):
    """
    Adapts a TTS audio file to match the prosodic features
    of the original segment.
    """
    try:
        import librosa
        import pyrubberband as pyrb
    except ImportError:
        raise ImportError("pip install librosa pyrubberband")

    y_tts, tts_sr = librosa.load(tts_path, sr=sr)
    orig_duration = original_features["duration"]
    orig_f0 = original_features["f0_median"]
    orig_energy = original_features["energy_mean"]
    orig_emotion = original_features["emotion"]
    orig_f0_std = original_features["f0_std"]

    # 1. Time-stretch to match original duration
    tts_duration = librosa.get_duration(y=y_tts, sr=sr)
    stretch_ratio = orig_duration / tts_duration if tts_duration > 0 else 1.0
    stretch_ratio = float(np.clip(stretch_ratio, max_compress, max_stretch))

    if abs(stretch_ratio - 1.0) > 0.03:
        y_tts = pyrb.time_stretch(y_tts, sr, 1.0 / stretch_ratio)
        logger.debug(
            f"[Prosody] Time-stretch x{stretch_ratio:.2f} "
            f"({tts_duration:.2f}s -> {orig_duration:.2f}s)"
        )

    # 2. Pitch-shift toward original F0
    if orig_f0 and orig_f0 > 50:
        f0_tts, _, _ = librosa.pyin(
            y_tts, fmin=50, fmax=500, sr=sr,
            frame_length=1024, hop_length=256,
        )
        f0_tts_valid = f0_tts[~np.isnan(f0_tts)]
        if len(f0_tts_valid) > 0:
            tts_f0_median = float(np.median(f0_tts_valid))
            if tts_f0_median > 0:
                semitone_shift = 12 * np.log2(orig_f0 / tts_f0_median)
                semitone_shift = float(np.clip(semitone_shift, -6, 6))
                if abs(semitone_shift) > 0.5:
                    y_tts = pyrb.pitch_shift(y_tts, sr, semitone_shift)
                    logger.debug(
                        f"[Prosody] Pitch shift {semitone_shift:+.1f} semitones "
                        f"(TTS={tts_f0_median:.1f}Hz -> target={orig_f0:.1f}Hz)"
                    )

    # 3. Energy/volume envelope matching
    y_tts = _match_energy(y_tts, orig_energy, sr)

    # 4. Emotion-based dynamic range scaling
    y_tts = _apply_emotion_dynamics(y_tts, sr, orig_emotion, orig_f0_std)

    # 5. Pad/trim to exact original duration
    target_samples = int(orig_duration * sr)
    if len(y_tts) < target_samples:
        y_tts = np.concatenate([y_tts, np.zeros(target_samples - len(y_tts))])
    else:
        fade_samples = int(0.05 * sr)
        if fade_samples > 0 and target_samples > fade_samples:
            y_tts[target_samples - fade_samples:target_samples] *= np.linspace(
                1, 0, fade_samples
            )
        y_tts = y_tts[:target_samples]

    sf.write(output_path, y_tts, sr)
    return output_path


def extract_original_segment(vocals_path, start_sec, end_sec, output_path, sr=16000):
    """
    Cuts the exact segment from original vocals track.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return None

    cmd = (
        f'ffmpeg -y -i "{vocals_path}" '
        f"-ss {start_sec:.3f} -t {duration:.3f} "
        f'-ar {sr} -ac 1 "{output_path}" -loglevel error'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.returncode == 0 and os.path.exists(output_path):
        return output_path
    return None


def process_segment_with_prosody(
    tts_audio_path,
    original_vocals_path,
    segment_start,
    segment_end,
    output_path,
    temp_dir="audio/prosody_tmp",
):
    """
    Full pipeline for one segment:
      1. Extract original vocal segment
      2. Analyze its prosody
      3. Adapt TTS output to match
      4. Save result
    """
    import shutil

    os.makedirs(temp_dir, exist_ok=True)
    seg_id = f"{segment_start:.3f}_{segment_end:.3f}"

    # Step 1: Extract original vocal segment
    orig_seg_path = os.path.join(temp_dir, f"orig_{seg_id}.wav")
    orig_seg = extract_original_segment(
        original_vocals_path, segment_start, segment_end, orig_seg_path,
    )

    if orig_seg is None:
        logger.warning(
            f"[Prosody] No original audio for segment "
            f"{segment_start:.1f}-{segment_end:.1f}s, skipping transfer"
        )
        shutil.copy(tts_audio_path, output_path)
        return output_path

    # Step 2: Analyze original
    try:
        features = analyze_segment(orig_seg_path)
        f0_str = f"{features['f0_median']:.1f}" if features["f0_median"] else "N/A"
        logger.info(
            f"[Prosody] Seg {segment_start:.1f}s: "
            f"emotion={features['emotion']}, "
            f"F0={f0_str}Hz, "
            f"energy={features['energy_mean']:.4f}"
        )
    except Exception as e:
        logger.error(f"[Prosody] Analysis failed for seg {seg_id}: {e}")
        shutil.copy(tts_audio_path, output_path)
        return output_path

    # Step 3: Adapt TTS to original prosody
    try:
        adapt_tts_to_original(
            tts_path=tts_audio_path,
            original_features=features,
            output_path=output_path,
        )
    except Exception as e:
        logger.error(f"[Prosody] Adaptation failed for seg {seg_id}: {e}")
        shutil.copy(tts_audio_path, output_path)
        return output_path

    # Cleanup temp
    try:
        os.remove(orig_seg_path)
    except OSError:
        pass

    return output_path
