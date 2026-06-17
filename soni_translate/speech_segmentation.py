from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_TORCH as DAMT,
    DEFAULT_ALIGN_MODELS_HF as DAMHF,
)
from whisperx.utils import TO_LANGUAGE_CODE
import whisperx
import torch
import gc
import os
import copy
import soundfile as sf
from IPython.utils import capture # noqa
from .language_configuration import EXTRA_ALIGN, INVERTED_LANGUAGES
from .logging_setup import logger
from .postprocessor import sanitize_file_name
from .utils import remove_directory_contents, run_command

ASR_MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "Systran/faster-distil-whisper-large-v3",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "distil-small.en",
    "distil-medium.en",
    "OpenAI_API_Whisper",
]

COMPUTE_TYPE_GPU = [
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "float16",
    "bfloat16",
    "float32"
]

COMPUTE_TYPE_CPU = [
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int16",
    "float32",
]

WHISPER_MODELS_PATH = './WHISPER_MODELS'


def openai_api_whisper(
    input_audio_file,
    source_lang=None,
    chunk_duration=1800
):

    info = sf.info(input_audio_file)
    duration = info.duration

    output_directory = "./whisper_api_audio_parts"
    os.makedirs(output_directory, exist_ok=True)
    remove_directory_contents(output_directory)

    if duration > chunk_duration:
        # Split the audio file into smaller chunks with 30-minute duration
        cm = f'ffmpeg -i "{input_audio_file}" -f segment -segment_time {chunk_duration} -c:a libvorbis "{output_directory}/output%03d.ogg"'
        run_command(cm)
        # Get list of generated chunk files
        chunk_files = sorted(
            [f"{output_directory}/{f}" for f in os.listdir(output_directory) if f.endswith('.ogg')]
        )
    else:
        one_file = f"{output_directory}/output000.ogg"
        cm = f'ffmpeg -i "{input_audio_file}" -c:a libvorbis {one_file}'
        run_command(cm)
        chunk_files = [one_file]

    # Transcript
    segments = []
    language = source_lang if source_lang else None
    for i, chunk in enumerate(chunk_files):
        from openai import OpenAI
        client = OpenAI()

        audio_file = open(chunk, "rb")
        transcription = client.audio.transcriptions.create(
          model="whisper-1",
          file=audio_file,
          language=language,
          response_format="verbose_json",
          timestamp_granularities=["segment"],
        )

        try:
            transcript_dict = transcription.model_dump()
        except: # noqa
            transcript_dict = transcription.to_dict()

        if language is None:
            logger.info(f'Language detected: {transcript_dict["language"]}')
            language = TO_LANGUAGE_CODE[transcript_dict["language"]]

        chunk_time = chunk_duration * (i)

        for seg in transcript_dict["segments"]:

            if "start" in seg.keys():
                segments.append(
                    {
                        "text": seg["text"],
                        "start": seg["start"] + chunk_time,
                        "end": seg["end"] + chunk_time,
                    }
                )

    audio = whisperx.load_audio(input_audio_file)
    result = {"segments": segments, "language": language}

    return audio, result


def find_whisper_models():
    path = WHISPER_MODELS_PATH
    folders = []

    if os.path.exists(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if (
                os.path.isdir(folder_path)
                and 'model.bin' in os.listdir(folder_path)
            ):
                folders.append(folder)
    return folders


def transcribe_speech(
    audio_wav,
    asr_model,
    compute_type,
    batch_size,
    SOURCE_LANGUAGE,
    literalize_numbers=True,
    segment_duration_limit=15,
):
    """
    Transcribe speech using a whisper model.

    Parameters:
    - audio_wav (str): Path to the audio file in WAV format.
    - asr_model (str): The whisper model to be loaded.
    - compute_type (str): Type of compute to be used (e.g., 'int8', 'float16').
    - batch_size (int): Batch size for transcription.
    - SOURCE_LANGUAGE (str): Source language for transcription.

    Returns:
    - Tuple containing:
        - audio: Loaded audio file.
        - result: Transcription result as a dictionary.
    """

    if asr_model == "OpenAI_API_Whisper":
        if literalize_numbers:
            logger.info(
                "OpenAI's API Whisper does not support "
                "the literalization of numbers."
            )
        return openai_api_whisper(audio_wav, SOURCE_LANGUAGE)

    # https://github.com/openai/whisper/discussions/277
    prompt = "以下是普通话的句子。" if SOURCE_LANGUAGE == "zh" else None
    SOURCE_LANGUAGE = (
        SOURCE_LANGUAGE if SOURCE_LANGUAGE != "zh-TW" else "zh"
    )
    asr_options = {
        "initial_prompt": prompt,
        "suppress_numerals": literalize_numbers
    }

    if asr_model not in ASR_MODEL_OPTIONS:

        base_dir = WHISPER_MODELS_PATH
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        model_dir = os.path.join(base_dir, sanitize_file_name(asr_model))

        if not os.path.exists(model_dir):
            from ctranslate2.converters import TransformersConverter

            quantization = "float32"
            # Download new model
            try:
                converter = TransformersConverter(
                    asr_model,
                    low_cpu_mem_usage=True,
                    copy_files=[
                        "tokenizer_config.json", "preprocessor_config.json"
                    ]
                )
                converter.convert(
                    model_dir,
                    quantization=quantization,
                    force=False
                )
            except Exception as error:
                if "File tokenizer_config.json does not exist" in str(error):
                    converter._copy_files = [
                        "tokenizer.json", "preprocessor_config.json"
                    ]
                    converter.convert(
                        model_dir,
                        quantization=quantization,
                        force=True
                    )
                else:
                    raise error

        asr_model = model_dir
        logger.info(f"ASR Model: {str(model_dir)}")

    model = whisperx.load_model(
        asr_model,
        os.environ.get("SONITR_DEVICE"),
        compute_type=compute_type,
        language=SOURCE_LANGUAGE,
        asr_options=asr_options,
    )

    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=segment_duration_limit,
        print_progress=True,
    )

    if result["language"] == "zh" and not prompt:
        result["language"] = "zh-TW"
        logger.info("Chinese - Traditional (zh-TW)")

    del model
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return audio, result


def align_speech(audio, result):
    """
    Aligns speech segments based on the provided audio and result metadata.

    Parameters:
    - audio (array): The audio data in a suitable format for alignment.
    - result (dict): Metadata containing information about the segments
         and language.

    Returns:
    - result (dict): Updated metadata after aligning the segments with
        the audio. This includes character-level alignments if
        'return_char_alignments' is set to True.

    Notes:
    - This function uses language-specific models to align speech segments.
    - It performs language compatibility checks and selects the
        appropriate alignment model.
    - Cleans up memory by releasing resources after alignment.
    """
    DAMHF.update(DAMT)  # lang align
    if (
        not result["language"] in DAMHF.keys()
        and not result["language"] in EXTRA_ALIGN.keys()
    ):
        logger.warning(
            "Automatic detection: Source language not compatible with align"
        )
        raise ValueError(
            f"Detected language {result['language']}  incompatible, "
            "you can select the source language to avoid this error."
        )
    if (
        result["language"] in EXTRA_ALIGN.keys()
        and EXTRA_ALIGN[result["language"]] == ""
    ):
        lang_name = (
            INVERTED_LANGUAGES[result["language"]]
            if result["language"] in INVERTED_LANGUAGES.keys()
            else result["language"]
        )
        logger.warning(
            "No compatible wav2vec2 model found "
            f"for the language '{lang_name}', skipping alignment."
        )
        return result

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=os.environ.get("SONITR_DEVICE"),
        model_name=None
        if result["language"] in DAMHF.keys()
        else EXTRA_ALIGN[result["language"]],
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        os.environ.get("SONITR_DEVICE"),
        return_char_alignments=True,
        print_progress=False,
    )
    del model_a
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return result


diarization_models = {
    "pyannote_3.1": "pyannote/speaker-diarization-3.1",
    "pyannote_3.1_precision": "pyannote/speaker-diarization-3.1",
    "pyannote_2.1": "pyannote/speaker-diarization@2.1",
    "pyannote_silero": "pyannote/speaker-diarization-3.1",
    "disable": "",
}


def reencode_speakers(result):

    if result["segments"][0]["speaker"] == "SPEAKER_00":
        return result

    speaker_mapping = {}
    counter = 0

    logger.debug("Reencode speakers")

    for segment in result["segments"]:
        old_speaker = segment["speaker"]
        if old_speaker not in speaker_mapping:
            speaker_mapping[old_speaker] = f"SPEAKER_{counter:02d}"
            counter += 1
        segment["speaker"] = speaker_mapping[old_speaker]

    return result


def diarize_speech(
    audio_wav,
    result,
    min_speakers,
    max_speakers,
    YOUR_HF_TOKEN,
    model_name="pyannote/speaker-diarization@2.1",
):
    """
    Performs speaker diarization on speech segments with intelligent fallback.

    Strategy:
    1. Try primary model with requested speaker range
    2. If too few speakers found, retry with wider range
    3. If model fails, try alternative model
    4. Last resort: assign based on segment gaps (silence = speaker change)
    """

    if max(min_speakers, max_speakers) <= 1 or not model_name:
        result_diarize = result
        result_diarize["segments"] = [
            {**item, "speaker": "SPEAKER_00"}
            for item in result_diarize["segments"]
        ]
        return reencode_speakers(result_diarize)

    # Try primary diarization
    diarize_segments = None
    used_model = model_name

    try:
        diarize_segments = _run_diarization(
            audio_wav, model_name, min_speakers, max_speakers, YOUR_HF_TOKEN
        )
    except Exception as e:
        logger.warning(f"Primary diarization failed ({model_name}): {e}")

        # Try fallback model
        fallback_models = [
            m for m in [
                "pyannote/speaker-diarization-3.1",
                "pyannote/speaker-diarization@2.1",
            ] if m != model_name
        ]
        for fallback in fallback_models:
            try:
                logger.info(f"Trying fallback diarization model: {fallback}")
                diarize_segments = _run_diarization(
                    audio_wav, fallback, min_speakers, max_speakers, YOUR_HF_TOKEN
                )
                used_model = fallback
                break
            except Exception as e2:
                logger.warning(f"Fallback {fallback} also failed: {e2}")

    if diarize_segments is None:
        logger.warning(
            "All diarization models failed. "
            "Using segment-gap based speaker assignment."
        )
        result_diarize = _assign_speakers_by_gaps(result, min_speakers, max_speakers)
        return reencode_speakers(result_diarize)

    # Assign speakers to transcription segments
    result_diarize = whisperx.assign_word_speakers(diarize_segments, result)

    # Count unique speakers found
    found_speakers = set()
    for seg in result_diarize["segments"]:
        if "speaker" in seg:
            found_speakers.add(seg["speaker"])

    segments_without_speaker = sum(
        1 for s in result_diarize["segments"] if "speaker" not in s
    )

    logger.info(
        f"Diarization ({used_model}): found {len(found_speakers)} speakers, "
        f"{segments_without_speaker} segments without speaker"
    )

    # Fill missing speakers with nearest speaker based on proximity
    if segments_without_speaker > 0:
        result_diarize = _fill_missing_speakers(result_diarize)

    # Validate: if too few speakers found but max_speakers > 1, retry
    if len(found_speakers) < 2 and max_speakers > 1:
        logger.warning(
            f"Only {len(found_speakers)} speaker(s) found but "
            f"max_speakers={max_speakers}. Retrying with wider range..."
        )
        try:
            wider_segments = _run_diarization(
                audio_wav, used_model,
                min_speakers=2, max_speakers=max(max_speakers + 2, 6),
                YOUR_HF_TOKEN=YOUR_HF_TOKEN,
            )
            wider_result = whisperx.assign_word_speakers(wider_segments, result)
            wider_speakers = set(
                s.get("speaker", "") for s in wider_result["segments"]
            )
            if len(wider_speakers) > len(found_speakers):
                logger.info(
                    f"Wider range found {len(wider_speakers)} speakers, "
                    f"using better result"
                )
                result_diarize = wider_result
                found_speakers = wider_speakers
        except Exception as e:
            logger.warning(f"Retry with wider range failed: {e}")

    # Final fallback: fill any remaining missing speakers
    for segment in result_diarize["segments"]:
        if "speaker" not in segment:
            segment["speaker"] = "SPEAKER_00"
            logger.debug(
                f"No speaker for segment at {segment.get('start', 0):.1f}s"
            )

    del diarize_segments
    gc.collect()
    torch.cuda.empty_cache()  # noqa

    return reencode_speakers(result_diarize)


def _run_diarization(audio_wav, model_name, min_speakers, max_speakers, hf_token):
    """Run a single diarization model and return segments."""
    diarize_model = whisperx.DiarizationPipeline(
        model_name=model_name,
        use_auth_token=hf_token,
        device=os.environ.get("SONITR_DEVICE"),
    )
    diarize_segments = diarize_model(
        audio_wav, min_speakers=min_speakers, max_speakers=max_speakers
    )
    del diarize_model
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return diarize_segments


def _assign_speakers_by_gaps(result, min_speakers, max_speakers):
    """
    Assign speakers based on segment gaps (silence between segments).
    When diarization fails completely, assume speaker changes at longer gaps.
    """
    segments = result.get("segments", [])
    if not segments:
        return result

    result_copy = copy.deepcopy(result)

    # Calculate typical gap between consecutive segments
    gaps = []
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i-1]["end"]
        gaps.append(gap)

    if not gaps:
        for seg in result_copy["segments"]:
            seg["speaker"] = "SPEAKER_00"
        return result_copy

    # Use median gap + 1 standard deviation as threshold for speaker change
    import numpy as np
    median_gap = np.median(gaps)
    std_gap = np.std(gaps)
    threshold = median_gap + std_gap

    # Ensure threshold is reasonable (0.3s to 3.0s)
    threshold = max(0.3, min(3.0, threshold))

    logger.info(
        f"Gap-based speaker assignment: threshold={threshold:.2f}s "
        f"(median={median_gap:.2f}s, std={std_gap:.2f}s)"
    )

    # Assign speakers
    current_speaker = 0
    for i, seg in enumerate(result_copy["segments"]):
        seg["speaker"] = f"SPEAKER_{current_speaker:02d}"
        if i > 0:
            gap = seg["start"] - result_copy["segments"][i-1]["end"]
            if gap > threshold:
                current_speaker = min(current_speaker + 1, max_speakers - 1)

    found_speakers = set(s["speaker"] for s in result_copy["segments"])
    logger.info(
        f"Gap-based assignment: {len(found_speakers)} speakers "
        f"(threshold: {threshold:.2f}s)"
    )

    return result_copy


def _fill_missing_speakers(result_diarize):
    """Fill segments without speakers by finding nearest speaker in time."""
    segments = result_diarize["segments"]
    missing_indices = [i for i, s in enumerate(segments) if "speaker" not in s]

    if not missing_indices:
        return result_diarize

    # Build list of (time, speaker) for segments that have speakers
    speaker_timeline = []
    for i, seg in enumerate(segments):
        if "speaker" in seg:
            mid_time = (seg["start"] + seg["end"]) / 2
            speaker_timeline.append((mid_time, seg["speaker"]))

    if not speaker_timeline:
        # No speakers at all, assign all to SPEAKER_00
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return result_diarize

    # For each missing segment, find nearest speaker
    for idx in missing_indices:
        seg = segments[idx]
        seg_mid = (seg["start"] + seg["end"]) / 2

        # Find nearest speaker by time
        nearest_speaker = min(
            speaker_timeline,
            key=lambda x: abs(x[0] - seg_mid)
        )[1]

        seg["speaker"] = nearest_speaker
        logger.debug(
            f"Filled missing speaker at {seg['start']:.1f}s "
            f"with {nearest_speaker}"
        )

    return result_diarize
