"""
Preview mode and segment-level retry system.
Enables quick QA previews and selective segment reprocessing.
"""

import os
import json
import shutil
from .logging_setup import logger


PREVIEW_DURATION_SEC = 60.0
MANIFEST_PATH = "output/segment_manifest.json"


def filter_segments_preview(segments, max_duration=PREVIEW_DURATION_SEC, start_time=0.0):
    """
    Filter segments to only include those within the preview window [start_time, start_time + max_duration].
    If start_time is 0, keeps first N seconds (legacy behavior).
    If start_time > 0, clips to the window starting at start_time.
    """
    end_time = start_time + max_duration
    if start_time > 0:
        filtered = [
            s for s in segments
            if float(s.get("start", 0)) >= start_time
            and float(s.get("start", 0)) < end_time
        ]
        # Shift timestamps so preview starts from 0
        for s in filtered:
            s["start"] = float(s["start"]) - start_time
            s["end"] = float(s["end"]) - start_time
        logger.info(
            f"Preview mode: {len(filtered)} segments "
            f"({start_time}s-{end_time}s of {len(segments)} total)"
        )
    else:
        filtered = [s for s in segments if float(s.get("start", 0)) < max_duration]
        logger.info(
            f"Preview mode: {len(filtered)} segments "
            f"(first {max_duration}s of {len(segments)} total)"
        )
    return filtered


def save_segment_manifest(segments_data, manifest_path=MANIFEST_PATH):
    """
    Save all processed segment data for later retry.
    Each entry contains: index, speaker, start, end, original_text,
    translated_text, tts_audio_path, status.
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Segment manifest saved: {manifest_path}")


def load_segment_manifest(manifest_path=MANIFEST_PATH):
    """Load the segment manifest from disk."""
    if not os.path.exists(manifest_path):
        logger.warning(f"No manifest found at {manifest_path}")
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retry_segment(
    segment_index, manifest_path=MANIFEST_PATH,
    translate_fn=None, tts_fn=None, normalize_fn=None,
    time_fit_fn=None, speaker_sample_paths=None,
):
    """
    Re-process a single segment by index without touching the rest.

    Args:
        segment_index: 0-indexed segment to retry
        manifest_path: path to segment manifest
        translate_fn: function(text, target_lang) -> translated_text
        tts_fn: function(text, speaker_ref, output_path) -> tts_path
        normalize_fn: function(original_path, tts_path, output_path) -> path
        time_fit_fn: function(tts_path, duration, output_path) -> path
        speaker_sample_paths: dict of speaker_id -> reference wav path

    Returns:
        dict: updated segment data
    """
    manifest = load_segment_manifest(manifest_path)
    if segment_index >= len(manifest):
        logger.error(
            f"Segment index {segment_index} out of range "
            f"(manifest has {len(segments)} segments)"
        )
        return None

    seg = manifest[segment_index]
    logger.info(
        f"[Retry] Reprocessing segment {segment_index}: "
        f"'{seg.get('original_text', '')[:50]}'"
    )

    try:
        # Re-translate
        if translate_fn:
            new_text = translate_fn(
                seg["original_text"], seg.get("target_lang", "hi")
            )
            seg["translated_text"] = new_text
        else:
            new_text = seg.get("translated_text", "")

        # Re-TTS with speaker reference
        if tts_fn:
            spk = seg.get("speaker", "SPEAKER_00")
            ref_path = None
            if speaker_sample_paths:
                ref_path = speaker_sample_paths.get(spk)
            tts_output = f"audio/seg_{segment_index:04d}_retry.wav"
            tts_fn(new_text, ref_path, tts_output)
            seg["tts_audio_path"] = tts_output

        # Re-normalize loudness
        if normalize_fn and "original_segment_path" in seg:
            norm_output = f"audio/seg_{segment_index:04d}_norm.wav"
            normalize_fn(
                seg["original_segment_path"],
                seg["tts_audio_path"],
                norm_output,
            )
            seg["tts_audio_path"] = norm_output

        # Re-fit timing
        if time_fit_fn:
            duration = seg.get("end", 0) - seg.get("start", 0)
            fit_output = f"audio/seg_{segment_index:04d}_fit.wav"
            time_fit_fn(
                seg["tts_audio_path"], duration, fit_output
            )
            seg["tts_audio_path"] = fit_output

        seg["status"] = "retried"
        manifest[segment_index] = seg
        save_segment_manifest(manifest, manifest_path)

        logger.info(f"[Retry] Segment {segment_index} reprocessed successfully")
        return seg
    except Exception as e:
        logger.error(f"[Retry] Segment {segment_index} failed: {e}")
        seg["status"] = "failed"
        manifest[segment_index] = seg
        save_segment_manifest(manifest, manifest_path)
        return None


def assemble_from_manifest(manifest_path=MANIFEST_PATH, output_dir="audio"):
    """
    Re-assemble the final dubbed audio track from the segment manifest.
    Overlays each segment's TTS audio at its original timestamp.
    """
    from pydub import AudioSegment
    from .audio_segments import Mixer

    manifest = load_segment_manifest(manifest_path)
    if not manifest:
        logger.error("Cannot assemble: manifest is empty")
        return None

    # Find total duration from last segment
    last_seg = max(manifest, key=lambda s: s.get("end", 0))
    total_duration = float(last_seg.get("end", 0)) + 5.0  # 5s buffer

    base_audio = AudioSegment.silent(
        duration=int(total_duration * 1000), frame_rate=41000
    )
    combined = Mixer()
    combined.overlay(base_audio)

    for seg in manifest:
        tts_path = seg.get("tts_audio_path")
        if not tts_path or not os.path.exists(tts_path):
            continue
        start = float(seg.get("start", 0))
        try:
            tts_audio = AudioSegment.from_file(tts_path)
            combined = combined.overlay(tts_audio, position=int(start * 1000))
        except Exception as e:
            logger.warning(
                f"Failed to add segment at {start}s: {e}"
            )

    output_path = os.path.join(output_dir, "assembled_dubbed.wav")
    result = combined.to_audio_segment()
    result.export(output_path, format="wav")
    logger.info(f"Assembled dubbed audio: {output_path}")
    return output_path
