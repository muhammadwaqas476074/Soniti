from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time
import threading

# Global cancel event for translation loops
_translation_cancel_event = threading.Event()


def set_translation_cancel():
    """Signal translation to cancel at next batch boundary."""
    _translation_cancel_event.set()


def clear_translation_cancel():
    """Clear the translation cancel signal."""
    _translation_cancel_event.clear()


def _check_translation_cancelled():
    """Raise if translation cancel was requested."""
    if _translation_cancel_event.is_set():
        _translation_cancel_event.clear()
        raise InterruptedError("Translation cancelled by user")


# =====================================
# Translation Checkpoint System
# =====================================
import os as _os
_CHECKPOINT_DIR = "translation_checkpoints"


def _save_checkpoint(translated_lines, total_segments, target, source):
    """Save translation progress to disk after each batch."""
    try:
        _os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
        key = f"{target}_{source or 'auto'}"
        path = _os.path.join(_CHECKPOINT_DIR, f"{key}.json")
        data = {
            "translated_lines": translated_lines,
            "total_segments": total_segments,
            "target": target,
            "source": source,
            "count": len(translated_lines),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Could not save checkpoint: {e}")


def _load_checkpoint(target, source):
    """Load translation checkpoint if available. Returns (lines, total) or (None, 0)."""
    try:
        key = f"{target}_{source or 'auto'}"
        path = _os.path.join(_CHECKPOINT_DIR, f"{key}.json")
        if not _os.path.exists(path):
            return None, 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lines = data.get("translated_lines", [])
        total = data.get("total_segments", 0)
        if lines and total:
            logger.info(
                f"Loaded checkpoint: {len(lines)}/{total} segments translated"
            )
            return lines, total
        return None, 0
    except Exception as e:
        logger.debug(f"Could not load checkpoint: {e}")
        return None, 0


def _clear_checkpoint(target, source):
    """Clear checkpoint after successful translation."""
    try:
        key = f"{target}_{source or 'auto'}"
        path = _os.path.join(_CHECKPOINT_DIR, f"{key}.json")
        if _os.path.exists(path):
            _os.remove(path)
            logger.debug("Translation checkpoint cleared")
    except Exception:
        pass


TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    "openrouter_batch",
    "openrouter_sequential",
    "disable_translation",
]
DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "openrouter_sequential",
    "disable_translation",
]


def translate_iterative(segments, target, source=None):
    """
    Translate text segments individually to the specified language.

    Parameters:
    - segments (list): A list of dictionaries with 'text' as a key for
        segment text.
    - target (str): Target language code.
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Translates each segment using Google Translate.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated_segments = translate_iterative(segments, 'es')
    """

    segments_ = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    translator = GoogleTranslator(source=source, target=target)

    for line in tqdm(range(len(segments_))):
        text = segments_[line]["text"]
        translated_line = translator.translate(text.strip())
        segments_[line]["text"] = translated_line

    return segments_


def verify_translate(
    segments,
    segments_copy,
    translated_lines,
    target,
    source
):
    """
    Verify integrity and translate segments if lengths match, otherwise
    recover gracefully.
    """
    expected = len(segments)
    got = len(translated_lines)

    if expected == got:
        for line in range(len(segments_copy)):
            logger.debug(
                f"{segments_copy[line]['text']} >> "
                f"{translated_lines[line].strip()}"
            )
            segments_copy[line]["text"] = translated_lines[
                line].replace("\t", "").replace("\n", "").strip()
        return segments_copy

    elif got >= expected - 5:
        # Close match — pad or trim to fix
        logger.warning(
            f"Translation count mismatch: expected {expected}, got {got}. "
            f"{'Padding' if got < expected else 'Trimming'} to match."
        )
        if got < expected:
            # Pad with originals
            for i in range(got, expected):
                translated_lines.append(segments[i]["text"].strip())
        else:
            # Trim extras
            translated_lines = translated_lines[:expected]

        for line in range(len(segments_copy)):
            segments_copy[line]["text"] = translated_lines[
                line].replace("\t", "").replace("\n", "").strip()
        return segments_copy

    else:
        # Way off — translate only the missing segments
        logger.error(
            f"Translation count mismatch: expected {expected}, got {got}. "
            "Translating missing segments with Google."
        )
        # Figure out which segments are missing (assume first N are translated)
        fixed_target = fix_code_language(target)
        fixed_source = fix_code_language(source) if source else "auto"

        if got < expected:
            # Translate the tail end that's missing
            for i in range(got, expected):
                try:
                    translator = GoogleTranslator(
                        source=fixed_source, target=fixed_target
                    )
                    result = translator.translate(segments[i]["text"].strip())
                    translated_lines.append(result if result else segments[i]["text"].strip())
                except Exception:
                    translated_lines.append(segments[i]["text"].strip())

        translated_lines = translated_lines[:expected]
        for line in range(len(segments_copy)):
            segments_copy[line]["text"] = translated_lines[
                line].replace("\t", "").replace("\n", "").strip()
        return segments_copy


def translate_batch(segments, target, chunk_size=2000, source=None):
    """
    Translate a batch of text segments into the specified language in chunks,
        respecting the character limit.

    Parameters:
    - segments (list): List of dictionaries with 'text' as a key for segment
        text.
    - target (str): Target language code.
    - chunk_size (int, optional): Maximum character limit for each translation
        chunk (default is 2000; max 5000).
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Splits input segments into chunks respecting the character limit for
        translation.
    - Translates the chunks using Google Translate.
    - If chunked translation fails, switches to iterative translation using
        `translate_iterative()`.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated = translate_batch(segments, 'es', chunk_size=4000, source='en')
    """

    segments_copy = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    # Get text
    text_lines = []
    for line in range(len(segments_copy)):
        text = segments_copy[line]["text"].strip()
        text_lines.append(text)

    # chunk limit
    text_merge = []
    actual_chunk = ""
    global_text_list = []
    actual_text_list = []
    for one_line in text_lines:
        one_line = " " if not one_line else one_line
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk:
                actual_chunk += " ||||| "
            actual_chunk += one_line
            actual_text_list.append(one_line)
        else:
            text_merge.append(actual_chunk)
            actual_chunk = one_line
            global_text_list.append(actual_text_list)
            actual_text_list = [one_line]
    if actual_chunk:
        text_merge.append(actual_chunk)
        global_text_list.append(actual_text_list)

    # translate chunks
    progress_bar = tqdm(total=len(segments), desc="Translating")
    translator = GoogleTranslator(source=source, target=target)
    split_list = []
    try:
        for text, text_iterable in zip(text_merge, global_text_list):
            translated_line = translator.translate(text.strip())
            split_text = translated_line.split("|||||")
            if len(split_text) == len(text_iterable):
                progress_bar.update(len(split_text))
            else:
                logger.debug(
                    "Chunk fixing iteratively. Len chunk: "
                    f"{len(split_text)}, expected: {len(text_iterable)}"
                )
                split_text = []
                for txt_iter in text_iterable:
                    translated_txt = translator.translate(txt_iter.strip())
                    split_text.append(translated_txt)
                    progress_bar.update(1)
            split_list.append(split_text)
        progress_bar.close()
    except Exception as error:
        progress_bar.close()
        logger.error(str(error))
        logger.warning(
            "The translation in chunks failed, switching to iterative."
            " Related: too many request"
        )  # use proxy or less chunk size
        return translate_iterative(segments, target, source)

    # un chunk
    translated_lines = list(chain.from_iterable(split_list))

    return verify_translate(
        segments, segments_copy, translated_lines, target, source
    )


def call_gpt_translate(
    client,
    model,
    system_prompt,
    user_prompt,
    original_text=None,
    batch_lines=None,
):

    # https://platform.openai.com/docs/guides/text-generation/json-mode
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content
    logger.debug(f"Result: {str(result)}")

    try:
        translation = json.loads(result)
    except Exception as error:
        match_result = re.search(r'\{.*?\}', result)
        if match_result:
            logger.error(str(error))
            json_str = match_result.group(0)
            translation = json.loads(json_str)
        else:
            raise error

    # Get valid data
    if batch_lines:
        for conversation in translation.values():
            if isinstance(conversation, dict):
                conversation = list(conversation.values())[0]
            if (
                list(
                    original_text["conversation"][0].values()
                )[0].strip() ==
                list(conversation[0].values())[0].strip()
            ):
                continue
            if len(conversation) == batch_lines:
                break

        fix_conversation_length = []
        for line in conversation:
            for speaker_code, text_tr in line.items():
                fix_conversation_length.append({speaker_code: text_tr})

        logger.debug(f"Data batch: {str(fix_conversation_length)}")
        logger.debug(
            f"Lines Received: {len(fix_conversation_length)},"
            f" expected: {batch_lines}"
        )

        return fix_conversation_length

    else:
        if isinstance(translation, dict):
            translation = list(translation.values())[0]
        if isinstance(translation, list):
            translation = translation[0]
        if isinstance(translation, set):
            translation = list(translation)[0]
        if not isinstance(translation, str):
            raise ValueError(f"No valid response received: {str(translation)}")

        return translation


def gpt_sequential(segments, model, target, source=None):
    from openai import OpenAI

    translated_segments = copy.deepcopy(segments)

    client = OpenAI()
    progress_bar = tqdm(total=len(segments), desc="Translating")

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    system_prompt = "Machine translation designed to output the translated_text JSON."

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        start = line["start"]
        user_prompt = f"Translate the following {lang_sc} text into {lang_tg}, write the fully translated text and nothing more:\n{text}"

        time.sleep(0.5)

        try:
            translated_text = call_gpt_translate(
                client,
                model,
                system_prompt,
                user_prompt,
            )

        except Exception as error:
            logger.error(
                f"{str(error)} >> The text of segment {start} "
                "is being corrected with Google Translate"
            )
            translator = GoogleTranslator(
                source=fixed_source, target=fixed_target
            )
            translated_text = translator.translate(text.strip())

        translated_segments[i]["text"] = translated_text.strip()
        progress_bar.update(1)

    progress_bar.close()

    return translated_segments


def gpt_batch(segments, model, target, token_batch_limit=900, source=None):
    from openai import OpenAI
    import tiktoken

    token_batch_limit = max(100, (token_batch_limit - 40) // 2)
    progress_bar = tqdm(total=len(segments), desc="Translating")
    segments_copy = copy.deepcopy(segments)
    encoding = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    name_speaker = "ABCDEFGHIJKL"

    translated_lines = []
    text_data_dict = []
    num_tokens = 0
    count_sk = {char: 0 for char in "ABCDEFGHIJKL"}

    for i, line in enumerate(segments_copy):
        text = line["text"]
        speaker = line["speaker"]
        last_start = line["start"]
        # text_data_dict.append({str(int(speaker[-1])+1): text})
        index_sk = int(speaker[-2:])
        character_sk = name_speaker[index_sk]
        count_sk[character_sk] += 1
        code_sk = character_sk+str(count_sk[character_sk])
        text_data_dict.append({code_sk: text})
        num_tokens += len(encoding.encode(text)) + 7
        if num_tokens >= token_batch_limit or i == len(segments_copy)-1:
            try:
                batch_lines = len(text_data_dict)
                batch_conversation = {"conversation": copy.deepcopy(text_data_dict)}
                # Reset vars
                num_tokens = 0
                text_data_dict = []
                count_sk = {char: 0 for char in "ABCDEFGHIJKL"}
                # Process translation
                # https://arxiv.org/pdf/2309.03409.pdf
                system_prompt = f"Machine translation designed to output the translated_conversation key JSON containing a list of {batch_lines} items."
                user_prompt = f"Translate each of the following text values in conversation{' from' if lang_sc else ''} {lang_sc} to {lang_tg}:\n{batch_conversation}"
                logger.debug(f"Prompt: {str(user_prompt)}")

                conversation = call_gpt_translate(
                    client,
                    model,
                    system_prompt,
                    user_prompt,
                    original_text=batch_conversation,
                    batch_lines=batch_lines,
                )

                if len(conversation) < batch_lines:
                    raise ValueError(
                        "Incomplete result received. Batch lines: "
                        f"{len(conversation)}, expected: {batch_lines}"
                    )

                for i, translated_text in enumerate(conversation):
                    if i+1 > batch_lines:
                        break
                    translated_lines.append(list(translated_text.values())[0])

                progress_bar.update(batch_lines)

            except Exception as error:
                logger.error(str(error))

                first_start = segments_copy[max(0, i-(batch_lines-1))]["start"]
                logger.warning(
                    f"The batch from {first_start} to {last_start} "
                    "failed, is being corrected with Google Translate"
                )

                translator = GoogleTranslator(
                    source=fixed_source,
                    target=fixed_target
                )

                for txt_source in batch_conversation["conversation"]:
                    translated_txt = translator.translate(
                        list(txt_source.values())[0].strip()
                    )
                    translated_lines.append(translated_txt.strip())
                    progress_bar.update(1)

    progress_bar.close()

    return verify_translate(
        segments, segments_copy, translated_lines, fixed_target, fixed_source
    )


# =====================================
# OpenRouter Translation
# =====================================

# Preference order by keyword — matched against live free model list
_OPENROUTER_PREFS = ["gemma", "qwen", "llama", "hermes", "nemotron", "gpt-oss"]

_validated_openrouter_model = None


def _get_best_openrouter_model(exclude=None):
    """
    Fetch live free models from OpenRouter, pick best by keyword preference.
    """
    global _validated_openrouter_model
    if _validated_openrouter_model and not exclude:
        return _validated_openrouter_model

    try:
        import requests
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        r.raise_for_status()
        free = [m["id"] for m in r.json()["data"] if m["id"].endswith(":free")]
        logger.info(f"OpenRouter free models: {len(free)}")

        for pref in _OPENROUTER_PREFS:
            matches = [m for m in free if pref in m.lower() and m not in (exclude or [])]
            if matches:
                chosen = sorted(matches)[-1]
                if not exclude:
                    _validated_openrouter_model = chosen
                logger.info(f"Selected: {chosen}")
                return chosen

        fallback = [m for m in free if m not in (exclude or [])]
        chosen = fallback[0] if fallback else None
        if chosen and not exclude:
            _validated_openrouter_model = chosen
        return chosen

    except Exception as e:
        logger.warning(f"Failed to get OpenRouter models: {e}")
        return None


# =====================================
# OpenRouter API Key Pool
# =====================================
# Keys are loaded from env vars: OPENROUTER_API_KEY, OPENROUTER_API_KEY_2, ...
# Or set dynamically via set_openrouter_keys()
_openrouter_keys = []
_openrouter_key_idx = 0


def set_openrouter_keys(keys_list):
    """Set API keys programmatically. Pass list of key strings."""
    global _openrouter_keys, _openrouter_key_idx
    _openrouter_keys = [k.strip() for k in keys_list if k and k.strip()]
    _openrouter_key_idx = 0
    if _openrouter_keys:
        logger.info(f"Loaded {len(_openrouter_keys)} OpenRouter API key(s)")


def _load_keys_from_env():
    """Load keys from environment variables."""
    import os
    keys = []
    # Primary
    k1 = os.environ.get("OPENROUTER_API_KEY", "")
    if k1:
        keys.append(k1)
    # Numbered: OPENROUTER_API_KEY_2, _3, ... _9
    for i in range(2, 10):
        ki = os.environ.get(f"OPENROUTER_API_KEY_{i}", "")
        if ki:
            keys.append(ki)
    return keys


def _get_all_keys():
    """Get all available keys, loading from env if pool is empty."""
    global _openrouter_keys
    if not _openrouter_keys:
        _openrouter_keys = _load_keys_from_env()
    return _openrouter_keys


def _get_openrouter_client():
    import os
    from openai import OpenAI

    keys = _get_all_keys()
    if not keys:
        raise ValueError(
            "No OPENROUTER_API_KEY set. "
            "Set OPENROUTER_API_KEY (and optionally OPENROUTER_API_KEY_2, _3, ... for failover)."
        )

    global _openrouter_key_idx
    api_key = keys[_openrouter_key_idx % len(keys)]

    base_url = "https://openrouter.ai/api/v1"

    try:
        import httpx
        http_client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/R3gm/SoniTranslate",
                "X-Title": "SoniTranslate",
            },
            timeout=120.0,
        )
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
    except (TypeError, ImportError):
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as e:
        logger.warning(f"OpenAI client creation failed: {e}, trying minimal setup")
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )


def _switch_to_next_key():
    """Switch to next API key. Returns True if switched, False if only 1 key."""
    global _openrouter_key_idx
    keys = _get_all_keys()
    if len(keys) <= 1:
        return False
    old = _openrouter_key_idx
    _openrouter_key_idx = (_openrouter_key_idx + 1) % len(keys)
    logger.info(f"Switched to API key #{_openrouter_key_idx + 1}/{len(keys)}")
    return True

def _get_target_lang_name(target_code):
    lang_name = INVERTED_LANGUAGES.get(target_code, target_code)
    return re.sub(r'\([^)]*\)', '', lang_name).strip()


def _openrouter_translate_sequential(
    segments, target, source=None, model=None,
    context_lines=3,
):
    if model is None:
        model = _get_best_openrouter_model()
    client = _get_openrouter_client()
    lang_tg = _get_target_lang_name(target)

    system_prompt = (
        f"You are a professional {lang_tg} dubbing dialogue writer. "
        f"Translate subtitle lines for dramatic audio dubbing. "
        f"Use natural, colloquial {lang_tg} — not formal or literal. "
        f"Preserve emotion and character voice."
    )

    # Try to resume from checkpoint
    checkpoint_lines, checkpoint_total = _load_checkpoint(target, source)
    translated_segments = copy.deepcopy(segments)
    total_segs = len(segments)

    if checkpoint_lines and checkpoint_total == total_segs:
        # Restore translated text from checkpoint
        for i, line in enumerate(translated_segments):
            if i < len(checkpoint_lines):
                line["text"] = checkpoint_lines[i]
        resume_from = sum(1 for cl in checkpoint_lines if cl.strip())
        logger.info(
            f"Resuming sequential translation from checkpoint: "
            f"{resume_from}/{total_segs} segments done"
        )
    else:
        resume_from = 0
        if checkpoint_lines:
            logger.warning("Checkpoint mismatch. Starting fresh.")

    progress_bar = tqdm(
        total=total_segs, desc="OpenRouter translating",
        initial=resume_from,
    )

    for i, line in enumerate(segments):
        if i < resume_from:
            progress_bar.update(1)
            continue
        _check_translation_cancelled()
        text = line["text"].strip()

        # Build context from surrounding lines
        ctx_start = max(0, i - context_lines)
        ctx_end = min(len(segments), i + 1 + context_lines)
        context_parts = []
        for j in range(ctx_start, ctx_end):
            prefix = "[TARGET LINE]" if j == i else "[CONTEXT]"
            context_parts.append(
                f"{prefix} {segments[j]['text'].strip()}"
            )
        context_block = "\n".join(context_parts)

        user_prompt = (
            f"Translate the line marked [TARGET LINE] into {lang_tg}.\n"
            f"Lines marked [CONTEXT] are for reference only.\n"
            f"Use natural, spoken {lang_tg} for voice dubbing.\n"
            f"Output ONLY the translated text, nothing else.\n\n"
            f"{context_block}"
        )

        time.sleep(0.5)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            translated_text = response.choices[0].message.content.strip()
            if not translated_text:
                raise ValueError("Empty response from OpenRouter")
        except Exception as error:
            if isinstance(error, InterruptedError):
                raise
            err_str = str(error).lower()
            is_429 = "429" in err_str or "rate limit" in err_str
            is_403 = "403" in err_str or "moderation" in err_str or "forbidden" in err_str

            if is_429 and _switch_to_next_key():
                client = _get_openrouter_client()
                logger.info("Rate limited, switched to next key, retrying...")
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                    )
                    translated_text = response.choices[0].message.content.strip()
                    if translated_text:
                        translated_segments[i]["text"] = translated_text
                        progress_bar.update(1)
                        continue
                except Exception:
                    pass

            elif is_403:
                old_model = model
                model = _get_best_openrouter_model(exclude=[old_model])
                if model and model != old_model:
                    logger.warning(f"403 moderation on {old_model}, switched to {model}")
                    client = _get_openrouter_client()
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0,
                        )
                        translated_text = response.choices[0].message.content.strip()
                        if translated_text:
                            translated_segments[i]["text"] = translated_text
                            progress_bar.update(1)
                            continue
                    except Exception:
                        pass

            logger.error(
                f"OpenRouter error at segment {line['start']}: {error} "
                ">> Falling back to Google Translate"
            )
            fixed_target = fix_code_language(target)
            fixed_source = fix_code_language(source) if source else "auto"
            try:
                translator = GoogleTranslator(source=fixed_source, target=fixed_target)
                translated_text = translator.translate(text)
                if not translated_text:
                    translated_text = text
            except Exception:
                logger.warning(f"Google fallback also failed for: {text[:50]}...")
                translated_text = text

        translated_segments[i]["text"] = translated_text
        progress_bar.update(1)

        # Save checkpoint every 10 segments
        if (i + 1) % 10 == 0:
            _save_checkpoint(
                [s["text"] for s in translated_segments],
                total_segs, target, source,
            )

    progress_bar.close()

    # Sequential translation complete — clear checkpoint
    _save_checkpoint(
        [s["text"] for s in translated_segments],
        total_segs, target, source,
    )
    _clear_checkpoint(target, source)

    return translated_segments


def openrouter_batch(
    segments, target, source=None,
    model=None,
    batch_size=20, max_retries=3,
    context_lines=3,
):
    """
    Batch translate using OpenRouter with surrounding context for coherence.
    Uses large batches + sleep between requests to stay within rate limits.
    Falls back to Google Translate if all retries fail.

    Args:
        segments: list of dicts with 'text' key
        target: target language code
        source: source language code
        model: OpenRouter model ID (auto-detected if None)
        batch_size: lines per batch (default 20 to minimize API calls)
        max_retries: retry count per batch
        context_lines: number of surrounding lines to include as context
    """
    if model is None:
        model = _get_best_openrouter_model()
    if not model:
        logger.warning("No OpenRouter model available, using Google Translate")
        return translate_batch(
            segments, fix_code_language(target), 4500,
            fix_code_language(source) if source else "auto",
        )

    client = _get_openrouter_client()
    lang_tg = _get_target_lang_name(target)
    segments_copy = copy.deepcopy(segments)
    total_segments = len(segments_copy)

    # Try to resume from checkpoint
    checkpoint_lines, checkpoint_total = _load_checkpoint(target, source)
    if checkpoint_lines and checkpoint_total == total_segments:
        translated_lines = checkpoint_lines
        resume_from = len(translated_lines)
        logger.info(
            f"Resuming translation from checkpoint: "
            f"{resume_from}/{total_segments} segments done"
        )
    else:
        translated_lines = []
        resume_from = 0
        if checkpoint_lines:
            logger.warning(
                "Checkpoint mismatch (segment count changed). Starting fresh."
            )

    progress_bar = tqdm(
        total=total_segments, desc="OpenRouter context-translating",
        initial=resume_from,
    )

    for batch_start in range(resume_from, total_segments, batch_size):
        _check_translation_cancelled()
        batch_end = min(batch_start + batch_size, total_segments)
        batch = segments_copy[batch_start:batch_end]
        texts = [s["text"].strip() for s in batch]

        # Build context: previous lines before batch
        ctx_before_start = max(0, batch_start - context_lines)
        ctx_before = segments_copy[ctx_before_start:batch_start]

        # Context: next lines after batch
        ctx_after_end = min(total_segments, batch_end + context_lines)
        ctx_after = segments_copy[batch_end:ctx_after_end]

        # Build the numbered batch lines (these MUST be translated)
        batch_numbered = []
        for i, t in enumerate(texts):
            batch_numbered.append(f"[TRANSLATE] {batch_start + i + 1}. {t}")

        # Build context lines (just for reference, NOT translated)
        context_before_lines = []
        for i, seg in enumerate(ctx_before):
            context_before_lines.append(
                f"[CONTEXT] {ctx_before_start + i + 1}. {seg['text'].strip()}"
            )

        context_after_lines = []
        for i, seg in enumerate(ctx_after):
            context_after_lines.append(
                f"[CONTEXT] {batch_end + i + 1}. {seg['text'].strip()}"
            )

        # Assemble prompt
        all_lines = context_before_lines + batch_numbered + context_after_lines
        numbered_block = "\n".join(all_lines)

        user_prompt = (
            f"You are translating subtitles for a {lang_tg} dubbing project.\n"
            f"Lines marked [CONTEXT] are for reference only — DO NOT translate them.\n"
            f"Lines marked [TRANSLATE] MUST be translated into {lang_tg}.\n\n"
            f"IMPORTANT RULES:\n"
            f"- Translate ONLY the [TRANSLATE] lines.\n"
            f"- Keep the same numbering as input.\n"
            f"- Use natural, spoken {lang_tg} suitable for voice dubbing.\n"
            f"- Maintain the tone and emotion from context.\n"
            f"- Keep proper nouns/names as-is or transliterate naturally.\n"
            f"- Output ONLY translated lines with [TRANSLATE] marker and number.\n\n"
            f"{numbered_block}"
        )

        batch_ok = False
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": (
                            f"You are a professional {lang_tg} dubbing dialogue "
                            f"writer. You translate subtitle lines for dramatic "
                            f"audio dubbing. You receive lines with context "
                            f"markers. Only translate lines marked [TRANSLATE]. "
                            f"Use natural, colloquial {lang_tg} — not formal or "
                            f"literal translation. Preserve emotion, timing, and "
                            f"character voice from the context lines."
                        )},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                result_text = response.choices[0].message.content.strip()
                result_lines = [
                    l.strip() for l in result_text.split("\n")
                    if l.strip()
                ]

                # Extract only [TRANSLATE] lines from response
                translated_batch = []
                for line in result_lines:
                    # Remove [TRANSLATE] marker if present
                    clean = re.sub(r'^\[TRANSLATE\]\s*', '', line)
                    # Strip numbering like "1. " or "1) "
                    clean = re.sub(r'^\d+[\.\)]\s*', '', clean).strip()
                    if clean:
                        translated_batch.append(clean)

                expected = len(texts)
                got = len(translated_batch)

                if got == expected:
                    pass
                elif got > expected:
                    logger.warning(
                        f"Got {got} translations, expected {expected}. "
                        "Trimming extras."
                    )
                    translated_batch = translated_batch[:expected]
                elif got >= expected - 2:
                    logger.warning(
                        f"Got {got} translations, expected {expected}. "
                        "Padding missing with originals."
                    )
                    while len(translated_batch) < expected:
                        idx = len(translated_batch)
                        translated_batch.append(texts[idx])
                else:
                    raise ValueError(
                        f"Expected {expected} translations, "
                        f"only got {got} from response"
                    )

                translated_lines.extend(translated_batch)
                batch_ok = True
                break

            except Exception as e:
                if isinstance(e, InterruptedError):
                    raise
                err_str = str(e).lower()
                is_429 = "429" in err_str or "rate limit" in err_str
                is_403 = "403" in err_str or "moderation" in err_str or "forbidden" in err_str

                if is_429:
                    if _switch_to_next_key():
                        client = _get_openrouter_client()
                        logger.info("Rate limited — switched to next API key, retrying...")
                        continue

                    wait = 60
                    try:
                        match = re.search(r'"X-RateLimit-Reset":\s*"?(\d+)"?', str(e))
                        if match:
                            reset_ms = int(match.group(1))
                            wait = max(1, (reset_ms / 1000) - time.time()) + 2
                            wait = min(wait, 120)
                    except Exception:
                        wait = 30

                    logger.warning(
                        f"Rate limited, all keys exhausted (attempt {attempt+1}/{max_retries}). "
                        f"Waiting {wait:.0f}s..."
                    )
                    time.sleep(wait)

                elif is_403:
                    old_model = model
                    model = _get_best_openrouter_model(exclude=[old_model])
                    if model and model != old_model:
                        logger.warning(f"403 moderation on {old_model}, switched to {model}")
                        continue
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        f"403 error, no alternate model (attempt {attempt+1}/{max_retries}). "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)

                else:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        f"OpenRouter batch attempt {attempt+1} failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)

        # Guarantee: every batch must produce translations
        if not batch_ok:
            logger.error(
                f"All OpenRouter retries failed for batch starting at {batch_start}. "
                "Falling back to Google Translate for this batch."
            )
            fixed_target = fix_code_language(target)
            fixed_source = fix_code_language(source) if source else "auto"
            for txt in texts:
                try:
                    translator = GoogleTranslator(
                        source=fixed_source, target=fixed_target
                    )
                    result = translator.translate(txt)
                    translated_lines.append(result if result else txt)
                except Exception:
                    logger.warning(f"Google Translate also failed for: {txt[:50]}...")
                    translated_lines.append(txt)

        # Sleep between batches to avoid per-minute rate limit
        progress_bar.update(len(batch))

        # Save checkpoint after each batch
        _save_checkpoint(translated_lines, total_segments, target, source)

        if batch_start + batch_size < total_segments:
            time.sleep(1.5)

    progress_bar.close()

    # Translation complete — clear checkpoint
    _clear_checkpoint(target, source)

    for i, line in enumerate(segments_copy):
        if i < len(translated_lines):
            line["text"] = translated_lines[i].strip()

    return verify_translate(
        segments, segments_copy, translated_lines,
        fix_code_language(target),
        fix_code_language(source) if source else "auto",
    )


def translate_text(
    segments,
    target,
    translation_process="google_translator_batch",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
    openrouter_batch_size=20,
):
    """Translates text segments using a specified process."""
    match translation_process:
        case "google_translator_batch":
            return translate_batch(
                segments,
                fix_code_language(target),
                chunk_size,
                fix_code_language(source)
            )
        case "google_translator":
            return translate_iterative(
                segments,
                fix_code_language(target),
                fix_code_language(source)
            )
        case model if model in ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]:
            return gpt_sequential(segments, model, target, source)
        case model if model in ["gpt-3.5-turbo-0125_batch", "gpt-4-turbo-preview_batch",]:
            return gpt_batch(
                segments,
                translation_process.replace("_batch", ""),
                target,
                token_batch_limit,
                source
            )
        case "openrouter_batch":
            return openrouter_batch(
                segments, target, source,
                batch_size=openrouter_batch_size,
            )
        case "openrouter_sequential":
            return _openrouter_translate_sequential(segments, target, source)
        case "disable_translation":
            return segments
        case _:
            raise ValueError("No valid translation process")
