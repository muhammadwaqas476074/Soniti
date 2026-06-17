"""
Speaker gender detection and intelligent TTS voice assignment.

Analyzes audio pitch (fundamental frequency) of each diarized speaker
to determine if they are male or female, then auto-assigns appropriate
TTS voices from the target language.
"""

from .logging_setup import logger
import os
import re
import json
import subprocess
import numpy as np
from collections import defaultdict


def parse_voice_sample_filename(filepath):
    """
    Parse a voice sample filename to extract identity and gender.
    
    Expected formats:
      - "Abhinav-male.wav" → identity="Abhinav", gender="male"
      - "priya-female.mp3" → identity="Priya", gender="female"
      - "RAJ-male.wav" → identity="Raj", gender="male"
    
    Gender suffix is case-insensitive. Identity is title-cased.
    
    Returns:
        dict: {"identity": str, "gender": str, "path": str} or None if invalid
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Try to split by last hyphen
    parts = basename.rsplit("-", 1)
    if len(parts) != 2:
        # Try underscore as separator
        parts = basename.rsplit("_", 1)
    
    if len(parts) != 2:
        logger.warning(f"Cannot parse voice sample filename: {basename} (expected 'Name-gender.ext')")
        return None
    
    identity_raw, gender_raw = parts
    gender_lower = gender_raw.strip().lower()
    
    if gender_lower not in ("male", "female"):
        logger.warning(f"Unknown gender '{gender_raw}' in filename: {basename} (expected 'male' or 'female')")
        return None
    
    identity = identity_raw.strip().title()
    return {
        "identity": identity,
        "gender": gender_lower,
        "path": filepath,
    }


def parse_uploaded_voice_samples(file_paths):
    """
    Parse a list of uploaded voice sample file paths.
    
    Returns:
        list: [{"identity": str, "gender": str, "path": str}, ...]
        dict: {"male": [...], "female": [...]} grouped by gender
    """
    samples = []
    for fp in file_paths:
        if fp is None:
            continue
        parsed = parse_voice_sample_filename(fp)
        if parsed:
            samples.append(parsed)
    
    male_samples = [s for s in samples if s["gender"] == "male"]
    female_samples = [s for s in samples if s["gender"] == "female"]
    
    logger.info(f"Parsed {len(samples)} voice samples: {len(male_samples)} male, {len(female_samples)} female")
    for s in samples:
        logger.info(f"  Sample: {s['identity']} ({s['gender']}) -> {s['path']}")
    
    return samples, {"male": male_samples, "female": female_samples}


def auto_map_speakers_to_samples(speaker_info, voice_samples_by_gender):
    """
    Auto-map diarized speakers to uploaded voice samples by gender.
    
    Uses F0 pitch analysis from Demucs-separated vocals to determine speaker gender,
    then matches to available voice samples of the same gender.
    
    Args:
        speaker_info: { "SPEAKER_00": {"gender": "male", "f0": 120.5, ...}, ... }
        voice_samples_by_gender: {"male": [{"identity": "Abhinav", ...}], "female": [...]}
    
    Returns:
        dict: { "SPEAKER_00": "Abhinav-male", ... }
    """
    assignments = {}
    
    male_samples = voice_samples_by_gender.get("male", [])
    female_samples = voice_samples_by_gender.get("female", [])
    
    male_speakers = sorted([s for s, i in speaker_info.items() if i.get("gender") == "male"])
    female_speakers = sorted([s for s, i in speaker_info.items() if i.get("gender") == "female"])
    unknown_speakers = sorted([s for s, i in speaker_info.items() if i.get("gender") == "unknown"])
    
    # Map male speakers to male samples (cycling if more speakers than samples)
    for idx, spk in enumerate(male_speakers):
        if male_samples:
            sample = male_samples[idx % len(male_samples)]
            assignments[spk] = f"{sample['identity']}-{sample['gender']}"
            f0 = speaker_info[spk].get("f0")
            f0_str = f"{f0:.1f} Hz" if f0 else "N/A"
            logger.info(f"[AutoMap] {spk} ({f0_str}) -> {assignments[spk]}")
        else:
            assignments[spk] = ""
            logger.warning(f"[AutoMap] {spk}: no male voice samples available")
    
    # Map female speakers to female samples
    for idx, spk in enumerate(female_speakers):
        if female_samples:
            sample = female_samples[idx % len(female_samples)]
            assignments[spk] = f"{sample['identity']}-{sample['gender']}"
            f0 = speaker_info[spk].get("f0")
            f0_str = f"{f0:.1f} Hz" if f0 else "N/A"
            logger.info(f"[AutoMap] {spk} ({f0_str}) -> {assignments[spk]}")
        else:
            assignments[spk] = ""
            logger.warning(f"[AutoMap] {spk}: no female voice samples available")
    
    # Map unknown speakers to any available sample
    all_samples = male_samples + female_samples
    for idx, spk in enumerate(unknown_speakers):
        if all_samples:
            sample = all_samples[idx % len(all_samples)]
            assignments[spk] = f"{sample['identity']}-{sample['gender']}"
            logger.info(f"[AutoMap] {spk} (unknown) -> {assignments[spk]}")
        else:
            assignments[spk] = ""
    
    return assignments


def get_sample_path_by_identity(identity_key, voice_samples):
    """
    Find the file path for a voice sample by its identity key.
    
    Args:
        identity_key: e.g. "Abhinav-male" or "Abhinav"
        voice_samples: list of {"identity": str, "gender": str, "path": str}
    
    Returns:
        str: file path or None
    """
    key_lower = identity_key.lower().strip()
    
    # Try exact match first
    for s in voice_samples:
        key = f"{s['identity']}-{s['gender']}"
        if key == identity_key or s["identity"] == identity_key:
            return s["path"]
    
    # Try case-insensitive match
    for s in voice_samples:
        key = f"{s['identity']}-{s['gender']}".lower()
        if key == key_lower or s["identity"].lower() == key_lower:
            return s["path"]
    
    return None


# Pitch ranges for gender classification (Hz)
MALE_F0_RANGE = (60, 180)      # lowered from 85 to catch deep voices
FEMALE_F0_RANGE = (165, 300)
AMBIGUITY_OVERLAP = (155, 185)  # tightened overlap zone
MIN_VALID_F0 = 55  # below this is likely noise, not a real voice

# Predefined voice mappings per target language
# Format: { "lang_code": { "male": [...], "female": [...] } }
# Hindi pool: ALL Edge TTS voices with "IN" locale + Male/Female tag
TARGET_VOICE_MAP = {
    "hi": {
        "male": [
            "hi-IN-MadhurNeural-Male",
            "hi-IN-PrabhatNeural-Male",
            "en-IN-PrabhatNeural-Male",
            "bn-IN-BashkarNeural-Male",
            "gu-IN-NiranjanNeural-Male",
            "kn-IN-GaganNeural-Male",
            "ml-IN-MidhunNeural-Male",
            "mr-IN-ManoharNeural-Male",
            "ta-IN-ValluvarNeural-Male",
            "te-IN-MohanNeural-Male",
            "ur-IN-SalmanNeural-Male",
            "en-US-AndrewMultilingualNeural-Male",
            "en-US-BrianMultilingualNeural-Male",
        ],
        "female": [
            "hi-IN-SwaraNeural-Female",
            "en-IN-NeerjaNeural-Female",
            "en-IN-NeerjaExpressiveNeural-Female",
            "bn-IN-TanishaaNeural-Female",
            "gu-IN-DhwaniNeural-Female",
            "kn-IN-SapnaNeural-Female",
            "ml-IN-SobhanaNeural-Female",
            "mr-IN-AarohiNeural-Female",
            "ta-IN-PallaviNeural-Female",
            "te-IN-ShrutiNeural-Female",
            "ur-IN-GulNeural-Female",
            "en-US-EmmaMultilingualNeural-Female",
            "en-US-AvaMultilingualNeural-Female",
        ],
        "male_bark": [
            "hi_speaker_2-Male BARK",
            "hi_speaker_5-Male BARK",
            "hi_speaker_6-Male BARK",
            "hi_speaker_7-Male BARK",
            "hi_speaker_8-Male BARK",
            "hi_speaker_9-Male BARK",
        ],
        "female_bark": [
            "hi_speaker_0-Female BARK",
            "hi_speaker_1-Female BARK",
            "hi_speaker_3-Female BARK",
            "hi_speaker_4-Female BARK",
        ],
        "male_vits": [
            "hi-facebook-mms VITS",
        ],
        "female_vits": [
            "hi-facebook-mms VITS",
        ],
    },
    "ur": {
        "male": [
            "ur-PK-SalmanNeural-Male",
            "ur-PK-AsadNeural-Male",
            "ur-IN-SalmanNeural-Male",
        ],
        "female": [
            "ur-PK-UzmaNeural-Female",
            "ur-IN-GulNeural-Female",
        ],
        "male_bark": [],
        "female_bark": [],
        "male_vits": [
            "ur_devanagari-facebook-mms VITS",
            "ur_latin-facebook-mms VITS",
        ],
        "female_vits": [],
    },
    "hi-ur": {
        # Hindistani: ALL IN-locale voices that handle Devanagari script
        # Urdu voices (ur-PK-*) CANNOT render Hindi Devanagari text
        # But ur-IN-* CAN (Indian Urdu uses Devanagari sometimes)
        "male": [
            "hi-IN-MadhurNeural-Male",
            "hi-IN-PrabhatNeural-Male",
            "en-IN-PrabhatNeural-Male",
            "bn-IN-BashkarNeural-Male",
            "gu-IN-NiranjanNeural-Male",
            "kn-IN-GaganNeural-Male",
            "ml-IN-MidhunNeural-Male",
            "mr-IN-ManoharNeural-Male",
            "ta-IN-ValluvarNeural-Male",
            "te-IN-MohanNeural-Male",
            "ur-IN-SalmanNeural-Male",
            "en-US-AndrewMultilingualNeural-Male",
            "en-US-BrianMultilingualNeural-Male",
        ],
        "female": [
            "hi-IN-SwaraNeural-Female",
            "en-IN-NeerjaNeural-Female",
            "en-IN-NeerjaExpressiveNeural-Female",
            "bn-IN-TanishaaNeural-Female",
            "gu-IN-DhwaniNeural-Female",
            "kn-IN-SapnaNeural-Female",
            "ml-IN-SobhanaNeural-Female",
            "mr-IN-AarohiNeural-Female",
            "ta-IN-PallaviNeural-Female",
            "te-IN-ShrutiNeural-Female",
            "ur-IN-GulNeural-Female",
            "en-US-EmmaMultilingualNeural-Female",
            "en-US-AvaMultilingualNeural-Female",
        ],
        "male_bark": [
            "hi_speaker_2-Male BARK",
            "hi_speaker_5-Male BARK",
            "hi_speaker_6-Male BARK",
        ],
        "female_bark": [
            "hi_speaker_0-Female BARK",
            "hi_speaker_1-Female BARK",
            "hi_speaker_3-Female BARK",
        ],
        "male_vits": [
            "hi-facebook-mms VITS",
        ],
        "female_vits": [
            "hi-facebook-mms VITS",
        ],
    },
}

# Fallback generic voices for any target language
GENERIC_VOICE_MAP = {
    "male": [
        "en-US-AndrewMultilingualNeural-Male",
        "en-US-BrianMultilingualNeural-Male",
    ],
    "female": [
        "en-US-EmmaMultilingualNeural-Female",
        "en-US-AvaMultilingualNeural-Female",
    ],
}


def _extract_speaker_audio(audio_wav, segments, speaker_label, output_dir="audio"):
    """Extract concatenated audio segments for a specific speaker."""
    speaker_segments = [
        s for s in segments if s.get("speaker") == speaker_label
    ]
    
    # DEBUG: log how many segments found
    logger.info(f"{speaker_label}: found {len(speaker_segments)} segments in diarization")
    
    if not speaker_segments:
        return None

    os.makedirs(output_dir, exist_ok=True)
    part_files = []

    for i, seg in enumerate(speaker_segments[:10]):  # limit to 10 for speed
        start = float(seg["start"])
        end = float(seg["end"])
        duration = end - start
        if duration < 0.5:  # raised from 0.1 — too short gives bad pitch
            continue
        part_file = os.path.join(output_dir, f"_speaker_part_{speaker_label}_{i}.wav")
        cmd = (
            f'ffmpeg -y -i "{audio_wav}" -ss {start} -t {duration} '  # ← QUOTED
            f'-ar 16000 -ac 1 "{part_file}"'  # ← removed 2>/dev/null so errors show
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Log ffmpeg errors instead of swallowing them
        if result.returncode != 0:
            logger.warning(f"ffmpeg failed for {speaker_label} seg {i}: {result.stderr[-200:]}")
        
        if os.path.exists(part_file) and os.path.getsize(part_file) > 0:
            part_files.append(part_file)

    logger.info(f"{speaker_label}: extracted {len(part_files)} valid audio parts")
    
    if not part_files:
        return None

    # If only one part, use it directly
    if len(part_files) == 1:
        output_file = os.path.join(output_dir, f"_speaker_{speaker_label}.wav")
        os.rename(part_files[0], output_file)
        return output_file

    # Concatenate all parts
    concat_file = os.path.join(output_dir, f"_concat_{speaker_label}.txt")
    with open(concat_file, "w") as f:
        for pf in part_files:
            abs_path = os.path.abspath(pf)  # ← use absolute paths in concat
            f.write(f"file '{abs_path}'\n")

    output_file = os.path.join(output_dir, f"_speaker_{speaker_label}.wav")
    concat_cmd = (
        f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" '
        f'-c copy "{output_file}"'
    )
    result = subprocess.run(concat_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"Concat failed for {speaker_label}: {result.stderr[-200:]}")

    # Cleanup parts
    for pf in part_files:
        try:
            os.remove(pf)
        except OSError:
            pass
    try:
        os.remove(concat_file)
    except OSError:
        pass

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return output_file
    return None

def _compute_pitch(audio_file):
    """Compute the fundamental frequency (F0) using autocorrelation."""
    try:
        import librosa
        y, sr = librosa.load(audio_file, sr=16000)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr,
            frame_length=2048, hop_length=512,
        )
        # Filter out NaN values (unvoiced frames)
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) == 0:
            return None
        return float(np.median(f0_valid))
    except ImportError:
        logger.warning("librosa not available, using basic pitch detection")
        return _compute_pitch_basic(audio_file)
    except Exception as e:
        logger.error(f"Pitch computation failed: {e}")
        return None


def _compute_pitch_basic(audio_file):
    """Basic pitch detection using autocorrelation without librosa."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0

        frame_size = 2048
        hop_size = 512
        pitches = []

        for i in range(0, len(samples) - frame_size, hop_size):
            frame = samples[i:i + frame_size]
            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]
            # Find first peak after zero crossing
            d = np.diff(corr)
            start = 0
            # Skip initial rising part
            for j in range(1, len(d)):
                if d[j] < 0 and j > 10:
                    start = j
                    break
            if start > 0:
                peak_idx = np.argmax(corr[start:]) + start
                if peak_idx > 0:
                    pitch = 16000.0 / peak_idx
                    if 50 <= pitch <= 500:
                        pitches.append(pitch)

        if not pitches:
            return None
        return float(np.median(pitches))
    except Exception as e:
        logger.error(f"Basic pitch computation failed: {e}")
        return None


def classify_gender(f0_median):
    """Classify gender based on median fundamental frequency."""
    if f0_median is None or f0_median < MIN_VALID_F0:
        return "unknown"  # don't guess on bad readings

    if f0_median < AMBIGUITY_OVERLAP[0]:
        return "male"
    elif f0_median > AMBIGUITY_OVERLAP[1]:
        return "female"
    else:
        center = (AMBIGUITY_OVERLAP[0] + AMBIGUITY_OVERLAP[1]) / 2
        return "male" if f0_median < center else "female"


def detect_speakers_gender(audio_wav, result_diarize):
    """
    Detect the gender of each speaker in the diarization result.

    Returns:
        dict: { "SPEAKER_00": {"gender": "male", "f0": 120.5, "sample_audio": "path"}, ... }
    """
    segments = result_diarize.get("segments", [])
    speakers = list(set(s.get("speaker", "SPEAKER_00") for s in segments))
    speakers.sort()

    speaker_info = {}

    for spk in speakers:
        logger.info(f"Analyzing gender for {spk}...")
        audio_path = _extract_speaker_audio(audio_wav, segments, spk)
        if audio_path is None:
            logger.warning(f"No audio found for {spk}, marking as unknown")
            speaker_info[spk] = {
                "gender": "unknown",
                "f0": None,
                "sample_audio": None,
            }
            continue

        f0 = _compute_pitch(audio_path)
        gender = classify_gender(f0)

        f0_str = f"{f0:.1f} Hz" if f0 is not None else "N/A"
        logger.info(f"{spk}: F0={f0_str} -> {gender}")

        speaker_info[spk] = {
            "gender": gender,
            "f0": f0,
            "sample_audio": audio_path,
        }

    return speaker_info


def get_available_voices_for_target(target_lang_code, engine="edge"):
    """
    Get available TTS voices for a target language, separated by gender.

    Args:
        target_lang_code: e.g. "hi", "ur", "hi-ur"
        engine: "edge", "bark", or "vits"

    Returns:
        dict: { "male": [...], "female": [...] }
    """
    voice_data = TARGET_VOICE_MAP.get(target_lang_code, GENERIC_VOICE_MAP)

    if engine == "edge":
        return {
            "male": voice_data.get("male", []),
            "female": voice_data.get("female", []),
        }
    elif engine == "bark":
        return {
            "male": voice_data.get("male_bark", []),
            "female": voice_data.get("female_bark", []),
        }
    elif engine == "vits":
        return {
            "male": voice_data.get("male_vits", []),
            "female": voice_data.get("female_vits", []),
        }
    else:
        # Return all combined
        all_male = (
            voice_data.get("male", [])
            + voice_data.get("male_bark", [])
            + voice_data.get("male_vits", [])
        )
        all_female = (
            voice_data.get("female", [])
            + voice_data.get("female_bark", [])
            + voice_data.get("female_vits", [])
        )
        return {"male": all_male, "female": all_female}


def auto_assign_voices(speaker_info, target_lang_code):
    """
    Auto-assign TTS voices based on detected gender and target language.
    Distributes voices as evenly as possible across speakers.

    Args:
        speaker_info: output from detect_speakers_gender()
        target_lang_code: e.g. "hi", "ur", "hi-ur"

    Returns:
        dict: { "SPEAKER_00": "hi-IN-MadhurNeural-Male", ... }
        dict: { "SPEAKER_00": {"gender": "male", "f0": 120.5, ... }, ... } (updated)
    """
    edge_voices = get_available_voices_for_target(target_lang_code, engine="edge")

    male_pool = edge_voices["male"] or GENERIC_VOICE_MAP["male"]
    female_pool = edge_voices["female"] or GENERIC_VOICE_MAP["female"]

    male_speakers = [s for s, i in speaker_info.items() if i["gender"] == "male"]
    female_speakers = [s for s, i in speaker_info.items() if i["gender"] == "female"]
    unknown_speakers = [s for s, i in speaker_info.items() if i["gender"] == "unknown"]

    assignments = {}

    for idx, spk in enumerate(sorted(male_speakers)):
        voice = male_pool[idx % len(male_pool)]
        assignments[spk] = voice
        f0 = speaker_info[spk].get("f0")
        f0_str = f"{f0:.1f} Hz" if f0 else "N/A"
        logger.info(f"[Voice] {spk} -> {voice} (male, F0={f0_str})")

    for idx, spk in enumerate(sorted(female_speakers)):
        voice = female_pool[idx % len(female_pool)]
        assignments[spk] = voice
        f0 = speaker_info[spk].get("f0")
        f0_str = f"{f0:.1f} Hz" if f0 else "N/A"
        logger.info(f"[Voice] {spk} -> {voice} (female, F0={f0_str})")

    all_pool = male_pool + female_pool
    for idx, spk in enumerate(sorted(unknown_speakers)):
        voice = all_pool[idx % len(all_pool)] if all_pool else GENERIC_VOICE_MAP["male"][0]
        assignments[spk] = voice
        logger.info(f"[Voice] {spk} -> {voice} (unknown gender, fallback)")

    return assignments, speaker_info


def get_voice_sample_files():
    """Get the paths to extracted speaker sample audio files."""
    sample_dir = "audio"
    samples = {}
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.startswith("_speaker_SPEAKER_") and f.endswith(".wav"):
                spk = f.replace("_speaker_", "").replace(".wav", "")
                samples[spk] = os.path.join(sample_dir, f)
    return samples


def select_script_appropriate_voice(text, language_code, gender_key, voice_map=None):
    """
    Dynamically filters the voice list based on the characters found in the text.
    Ensures Hindi Devanagari script gets Hindi/Multilingual voices,
    English/Latin characters get English/Multilingual voices,
    and Urdu/Perso-Arabic script gets Urdu voices.

    Prevents Edge TTS NoAudioReceived crashes when script doesn't match voice locale.

    Args:
        text: The TTS text to analyze
        language_code: Target language code (e.g. "hi", "hi-ur", "en")
        gender_key: "male" or "female"
        voice_map: Optional custom voice map. Defaults to TARGET_VOICE_MAP.

    Returns:
        str: Best voice name for the text's script (with -Male/-Female suffix)
    """
    if voice_map is None:
        voice_map = TARGET_VOICE_MAP

    # Fallback to default list if language or gender isn't mapped
    if language_code not in voice_map or gender_key not in voice_map[language_code]:
        fallback = voice_map.get("hi", {}).get(gender_key, ["hi-IN-MadhurNeural-Male"])
        return fallback[0] if fallback else "hi-IN-MadhurNeural-Male"

    available_voices = voice_map[language_code][gender_key]
    if not available_voices:
        return "hi-IN-MadhurNeural-Male"

    # Detect character scripts using Unicode ranges
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    has_urdu_script = bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))

    filtered_voices = []

    if has_devanagari:
        # Hindi Devanagari: only voices that natively read Devanagari
        for voice in available_voices:
            if (voice.startswith("hi-") or
                "Multilingual" in voice or
                voice.startswith("mr-IN-") or  # Marathi uses Devanagari
                voice.startswith("ur-IN-")):   # Indian Urdu can read Devanagari
                filtered_voices.append(voice)

    elif has_urdu_script:
        # Urdu/Perso-Arabic: only Urdu voices
        for voice in available_voices:
            if voice.startswith("ur-"):
                filtered_voices.append(voice)

    elif has_latin:
        # Latin/English text: English or Multilingual voices
        for voice in available_voices:
            if (voice.startswith("en-") or
                "Multilingual" in voice or
                "latin" in voice):
                filtered_voices.append(voice)

    if filtered_voices:
        chosen = filtered_voices[0]
        logger.debug(f"[ScriptRouter] '{text[:30]}...' -> {chosen}")
        return chosen

    # Nothing matched — return first available voice as last resort
    logger.warning(f"[ScriptRouter] No script-matched voice for '{text[:30]}...' "
                   f"(devanagari={has_devanagari}, urdu={has_urdu_script}, latin={has_latin})")
    return available_voices[0]


def analyze_speaker_script(segments, speaker_info):
    """
    Analyze the translated text for each speaker to detect the dominant script.
    
    For each speaker, examines all their segments to determine if the text is:
    - "devanagari": Hindi/Devanagari script
    - "latin": English/Latin script
    - "mixed": Both scripts present
    - "unknown": No text or unable to determine
    
    Also extracts a character sample for display.
    
    Args:
        segments: List of segment dicts with 'speaker' and 'text' keys
        speaker_info: Dict of speaker info (will be updated in-place)
    
    Returns:
        dict: Updated speaker_info with script analysis added
    """
    # Collect all text per speaker
    speaker_texts = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker", "")
        text = seg.get("text", "")
        if spk and text:
            speaker_texts[spk].append(text)
    
    for spk, info in speaker_info.items():
        texts = speaker_texts.get(spk, [])
        all_text = " ".join(texts)
        
        if not all_text.strip():
            info["script"] = "unknown"
            info["script_sample"] = ""
            info["script_description"] = "No text"
            continue
        
        # Count characters by script type
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', all_text))
        latin_count = len(re.findall(r'[a-zA-Z]', all_text))
        urdu_count = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', all_text))
        total_alpha = devanagari_count + latin_count + urdu_count
        
        if total_alpha == 0:
            info["script"] = "unknown"
            info["script_sample"] = ""
            info["script_description"] = "No alphabetic characters"
            continue
        
        # Determine dominant script
        devanagari_pct = devanagari_count / total_alpha
        latin_pct = latin_count / total_alpha
        urdu_pct = urdu_count / total_alpha
        
        # Extract a character sample (first segment with text)
        sample_text = texts[0] if texts else ""
        if len(sample_text) > 30:
            sample_text = sample_text[:30] + "..."
        
        # Determine script classification
        if urdu_pct > 0.5:
            info["script"] = "urdu"
            info["script_sample"] = sample_text
            info["script_description"] = f"Urdu/Perso-Arabic ({urdu_pct*100:.0f}%)"
        elif devanagari_pct > 0.7:
            info["script"] = "devanagari"
            info["script_sample"] = sample_text
            info["script_description"] = f"Devanagari/Hindi ({devanagari_pct*100:.0f}%)"
        elif latin_pct > 0.7:
            info["script"] = "latin"
            info["script_sample"] = sample_text
            info["script_description"] = f"English/Latin ({latin_pct*100:.0f}%)"
        elif devanagari_pct > 0.2 and latin_pct > 0.2:
            info["script"] = "mixed"
            info["script_sample"] = sample_text
            info["script_description"] = f"Mixed Devanagari ({devanagari_pct*100:.0f}%) + Latin ({latin_pct*100:.0f}%)"
        else:
            # Majority rule
            if devanagari_pct > latin_pct:
                info["script"] = "devanagari"
                info["script_sample"] = sample_text
                info["script_description"] = f"Devanagari/Hindi ({devanagari_pct*100:.0f}%)"
            else:
                info["script"] = "latin"
                info["script_sample"] = sample_text
                info["script_description"] = f"English/Latin ({latin_pct*100:.0f}%)"
        
        logger.info(f"[ScriptAnalysis] {spk}: {info['script']} - {info['script_description']}")
    
    return speaker_info


def get_default_voice_for_script(script, gender, target_lang_code="hi"):
    """
    Get the default voice for a given script and gender combination.
    
    Args:
        script: "devanagari", "latin", "urdu", "mixed", or "unknown"
        gender: "male" or "female"
        target_lang_code: Target language code (e.g. "hi", "hi-ur")
    
    Returns:
        str: Default voice name
    """
    if script == "devanagari":
        # Hindi voices for Devanagari
        voice_data = TARGET_VOICE_MAP.get(target_lang_code, GENERIC_VOICE_MAP)
        voices = voice_data.get(gender, [])
        if voices:
            return voices[0]
        # Fallback to generic Hindi
        fallback = GENERIC_VOICE_MAP.get(gender, [])
        return fallback[0] if fallback else "hi-IN-MadhurNeural-Male"
    
    elif script == "latin":
        # English voices for Latin
        voice_data = TARGET_VOICE_MAP.get("en", GENERIC_VOICE_MAP)
        voices = voice_data.get(gender, [])
        if voices:
            return voices[0]
        # Fallback to generic English
        fallback = GENERIC_VOICE_MAP.get(gender, [])
        return fallback[0] if fallback else "en-US-GuyNeural-Male"
    
    elif script == "urdu":
        # Urdu voices
        voice_data = TARGET_VOICE_MAP.get("ur", GENERIC_VOICE_MAP)
        voices = voice_data.get(gender, [])
        if voices:
            return voices[0]
        # Fallback to generic Hindi
        fallback = GENERIC_VOICE_MAP.get(gender, [])
        return fallback[0] if fallback else "hi-IN-MadhurNeural-Male"
    
    else:
        # Mixed or unknown - use target language
        voice_data = TARGET_VOICE_MAP.get(target_lang_code, GENERIC_VOICE_MAP)
        voices = voice_data.get(gender, [])
        if voices:
            return voices[0]
        fallback = GENERIC_VOICE_MAP.get(gender, [])
        return fallback[0] if fallback else "hi-IN-MadhurNeural-Male"


def check_voice_script_match(voice_name, script):
    """
    Check if a voice name matches the detected script.
    
    Args:
        voice_name: Voice name (e.g. "hi-IN-MadhurNeural-Male")
        script: Detected script ("devanagari", "latin", "urdu", "mixed", "unknown")
    
    Returns:
        tuple: (is_match: bool, warning: str or None)
    """
    if script in ("unknown", "mixed"):
        return True, None  # No warning for unknown or mixed
    
    voice_lower = voice_name.lower()
    
    if script == "devanagari":
        # Check if voice is Hindi/Marathi/Urdu (Indian)
        if (voice_lower.startswith("hi-") or 
            voice_lower.startswith("mr-") or
            voice_lower.startswith("ur-") or
            "multilingual" in voice_lower):
            return True, None
        else:
            return True, f"⚠️ Voice '{voice_name}' may not support Devanagari script. Consider using a Hindi voice."
    
    elif script == "latin":
        # Check if voice is English
        if (voice_lower.startswith("en-") or
            "multilingual" in voice_lower or
            "latin" in voice_lower):
            return True, None
        else:
            return True, f"⚠️ Voice '{voice_name}' may not support English/Latin text. Consider using an English voice."
    
    elif script == "urdu":
        # Check if voice is Urdu
        if voice_lower.startswith("ur-"):
            return True, None
        else:
            return True, f"⚠️ Voice '{voice_name}' may not support Urdu script. Consider using a Urdu voice."
    
    return True, None


# =====================================
# Transliteration Functions
# =====================================

# Latin to Devanagari mapping (approximate phonetic)
_LATIN_TO_DEVANAGARI = {
    # Vowels
    'a': 'अ', 'A': 'आ', 'i': 'इ', 'I': 'ई', 'u': 'उ', 'U': 'ऊ',
    'e': 'ए', 'E': 'ऐ', 'o': 'ओ', 'O': 'औ',
    # Consonants
    'k': 'क', 'K': 'ख', 'g': 'ग', 'G': 'घ', 'ng': 'ङ',
    'ch': 'च', 'Ch': 'छ', 'j': 'ज', 'J': 'झ', 'ñ': 'ञ',
    't': 'ट', 'T': 'ठ', 'd': 'ड', 'D': 'ढ', 'N': 'ण',
    'th': 'त', 'Th': 'थ', 'dh': 'द', 'Dh': 'ध', 'n': 'न',
    'p': 'प', 'P': 'फ', 'b': 'ब', 'B': 'भ', 'm': 'म',
    'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व',
    'sh': 'श', 'Sh': 'ष', 's': 'स', 'S': 'श',
    'h': 'ह', 'f': 'फ़', 'Z': 'ज़',
    # Matras (vowel signs)
    'aa': 'ा', 'ii': 'ी', 'uu': 'ू', 'ai': 'ै', 'au': 'ौ', 'ee': 'ी', 'oo': 'ू',
    # Special
    'N ': 'ं', 'M ': 'ँ', 'H ': 'ः',
}

# Devanagari to Latin mapping (approximate phonetic)
_DEVANAGARI_TO_LATIN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha', 'फ़': 'fa', 'ज़': 'za',
    'ा': 'aa', 'ी': 'ee', 'ू': 'oo', 'ै': 'ai', 'ौ': 'au',
    'ं': 'n', 'ँ': 'm', 'ः': 'h',
}


def transliterate_latin_to_devanagari(text):
    """
    Transliterate Latin/English text to Devanagari script.
    Uses approximate phonetic mapping.
    
    Args:
        text: Latin script text (e.g., "namaste", "I am sorry")
    
    Returns:
        str: Devanagari transliteration (e.g., "नमस्ते", "आई एम सॉरी")
    """
    if not text:
        return text
    
    # Check if already Devanagari
    if re.search(r'[\u0900-\u097F]', text):
        return text
    
    result = []
    i = 0
    text_lower = text.lower()
    
    while i < len(text_lower):
        # Try two-char match first
        if i + 1 < len(text_lower):
            two_char = text_lower[i:i+2]
            if two_char in _LATIN_TO_DEVANAGARI:
                result.append(_LATIN_TO_DEVANAGARI[two_char])
                i += 2
                continue
        
        # Try single-char match
        char = text_lower[i]
        if char in _LATIN_TO_DEVANAGARI:
            result.append(_LATIN_TO_DEVANAGARI[char])
        elif char.isalpha():
            # Unknown Latin char, use placeholder
            result.append(' ')
        else:
            # Keep punctuation, numbers, spaces
            result.append(char)
        i += 1
    
    return ''.join(result)


def transliterate_devanagari_to_latin(text):
    """
    Transliterate Devanagari text to Latin/English script.
    Uses approximate phonetic mapping.
    
    Args:
        text: Devanagari script text (e.g., "नमस्ते")
    
    Returns:
        str: Latin transliteration (e.g., "namaste")
    """
    if not text:
        return text
    
    # Check if already Latin
    if re.search(r'[a-zA-Z]', text) and not re.search(r'[\u0900-\u097F]', text):
        return text
    
    result = []
    for char in text:
        if char in _DEVANAGARI_TO_LATIN:
            result.append(_DEVANAGARI_TO_LATIN[char])
        elif re.search(r'[\u0900-\u097F]', char):
            # Unknown Devanagari char, skip
            pass
        else:
            # Keep punctuation, numbers, spaces
            result.append(char)
    
    return ''.join(result)


# =====================================
# Voice Script Detection
# =====================================

def detect_voice_script(voice_name):
    """
    Detect the script type a voice is designed for.
    
    Args:
        voice_name: Voice name (e.g., "hi-IN-SwaraNeural-Female", "en-US-GuyNeural-Male")
    
    Returns:
        str: "devanagari", "latin", "multilingual", or "unknown"
    """
    if not voice_name:
        return "unknown"
    
    voice_lower = voice_name.lower()
    
    # Check for multilingual voices
    if "multilingual" in voice_lower:
        return "multilingual"
    
    # Check for Devanagari/Hindi/Indian voices
    if (voice_lower.startswith("hi-") or
        voice_lower.startswith("mr-") or
        voice_lower.startswith("bn-") or
        voice_lower.startswith("gu-") or
        voice_lower.startswith("kn-") or
        voice_lower.startswith("ml-") or
        voice_lower.startswith("ta-") or
        voice_lower.startswith("te-") or
        voice_lower.startswith("ur-")):
        return "devanagari"
    
    # Check for Latin/English voices
    if (voice_lower.startswith("en-") or
        voice_lower.startswith("es-") or
        voice_lower.startswith("fr-") or
        voice_lower.startswith("de-") or
        voice_lower.startswith("it-") or
        voice_lower.startswith("pt-") or
        voice_lower.startswith("ru-") or
        voice_lower.startswith("ja-") or
        voice_lower.startswith("ko-") or
        voice_lower.startswith("zh-")):
        return "latin"
    
    return "unknown"


def sanitize_text_for_voice(text, voice_name):
    """
    Sanitize text to match the voice's script type.
    This ensures voice consistency throughout all segments of a speaker.
    
    Priority:
    1. Use user-assigned voice exactly
    2. Transliterate text to match voice script
    3. Emergency fallback: swap to multilingual voice (same gender)
    
    Args:
        text: Text to sanitize
        voice_name: User-assigned voice name
    
    Returns:
        tuple: (sanitized_text, final_voice_name, was_modified: bool)
    """
    if not text or not voice_name:
        return text, voice_name, False
    
    voice_script = detect_voice_script(voice_name)
    
    # Multilingual voices can handle any script naturally
    if voice_script == "multilingual":
        return text, voice_name, False
    
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    
    # Case 1: Voice is Devanagari, but text has Latin
    if voice_script == "devanagari" and has_latin and not has_devanagari:
        # Transliterate Latin to Devanagari
        sanitized = transliterate_latin_to_devanagari(text)
        logger.info(f"[TextSanitize] Transliterated Latin→Devanagari: '{text[:30]}...' → '{sanitized[:30]}...'")
        return sanitized, voice_name, True
    
    # Case 2: Voice is Latin, but text has Devanagari
    if voice_script == "latin" and has_devanagari and not has_latin:
        # Transliterate Devanagari to Latin
        sanitized = transliterate_devanagari_to_latin(text)
        logger.info(f"[TextSanitize] Transliterated Devanagari→Latin: '{text[:30]}...' → '{sanitized[:30]}...'")
        return sanitized, voice_name, True
    
    # Case 3: Mixed scripts - transliterate to match voice
    if has_devanagari and has_latin:
        if voice_script == "devanagari":
            # Transliterate Latin parts to Devanagari
            # Simple approach: transliterate entire text, Devanagari stays
            sanitized = transliterate_latin_to_devanagari(text)
            logger.info(f"[TextSanitize] Mixed→Devanagari: '{text[:30]}...' → '{sanitized[:30]}...'")
            return sanitized, voice_name, True
        elif voice_script == "latin":
            # Transliterate Devanagari parts to Latin
            sanitized = transliterate_devanagari_to_latin(text)
            logger.info(f"[TextSanitize] Mixed→Latin: '{text[:30]}...' → '{sanitized[:30]}...'")
            return sanitized, voice_name, True
    
    # Text already matches voice script, no change needed
    return text, voice_name, False


def get_emergency_fallback_voice(voice_name, gender):
    """
    Get a multilingual voice of the same gender as emergency fallback.
    Only used when text cannot be safely transliterated.
    
    Args:
        voice_name: Original voice name (for logging)
        gender: "male" or "female"
    
    Returns:
        str: Multilingual voice name
    """
    # Find a multilingual voice
    multilingual_voices = [
        "en-US-AndrewMultilingualNeural-Male",
        "en-US-BrianMultilingualNeural-Male",
        "en-US-EmmaMultilingualNeural-Female",
        "en-US-AvaMultilingualNeural-Female",
    ]
    
    for voice in multilingual_voices:
        if gender.lower() in voice.lower():
            logger.warning(f"[EmergencyFallback] Swapping {voice_name} → {voice}")
            return voice
    
    # Ultimate fallback
    return "en-US-AndrewMultilingualNeural-Male" if gender.lower() == "male" else "en-US-EmmaMultilingualNeural-Female"
