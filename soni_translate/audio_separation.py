"""
Demucs-based audio source separation.
Separates vocals from background music/SFX for clean dubbing.
"""

import subprocess
import os
import sys
import shutil
import glob
import urllib.request
from .logging_setup import logger


# Use standard htdemucs to avoid OOM crashes on Colab/T4 GPUs
DEMUCS_MODEL = "htdemucs"

# Known Demucs model checkpoints (htdemucs uses hybrid_transformer)
DEMUCS_CHECKPOINTS = {
    "htdemucs_ft": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
    "htdemucs": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92c7343b.th",
}


def _find_demucs_python():
    """Find the Python executable that has demucs installed."""
    candidates = [
        sys.executable,
        shutil.which("python3") or "",
        shutil.which("python") or "",
    ]
    for py in candidates:
        if not py:
            continue
        try:
            result = subprocess.run(
                [py, "-c", "import demucs; print(demucs.__version__)"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                logger.info(f"Found demucs in: {py} (version: {result.stdout.strip()})")
                return py
        except Exception:
            continue

    # Last resort: try uv run
    uv = shutil.which("uv")
    if uv:
        try:
            result = subprocess.run(
                [uv, "run", "python", "-c", "import demucs; print(demucs.__version__)"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info(f"Found demucs via uv: {result.stdout.strip()}")
                return uv  # caller will use uv run python -m demucs ...
        except Exception:
            pass

    return None


def _ensure_demucs_model(model_name):
    """
    Pre-download Demucs checkpoint if not cached.
    NOTE: Demucs 4.0.1+ downloads models automatically on first run.
    This function is a no-op — demucs handles its own downloads.
    """
    pass


def separate_audio_sources(video_path, output_dir="audio_separated"):
    """
    Extracts audio from video, runs Demucs to separate vocals from
    background (music + effects).

    Returns:
        tuple: (vocals_path, no_vocals_path)
        - vocals_path: isolated vocals (used for ASR/diarization)
        - no_vocals_path: music + SFX (kept for final remix)
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_audio = os.path.join(output_dir, "raw_audio.wav")

    # Step 1: Extract raw audio from video
    if not os.path.exists(raw_audio):
        logger.info("Extracting audio from video for source separation...")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "2", "-ar", "44100",
            "-vn", raw_audio,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Raw audio extracted: {raw_audio}")

    # Step 2: Run Demucs
    model_output = os.path.join(output_dir, DEMUCS_MODEL, "raw_audio")
    vocals_path = os.path.join(model_output, "vocals.wav")
    no_vocals_path = os.path.join(model_output, "no_vocals.wav")

    if os.path.exists(vocals_path) and os.path.exists(no_vocals_path):
        logger.info("Demucs separation already done, using cached results")
        return vocals_path, no_vocals_path

    logger.info(f"Running Demucs ({DEMUCS_MODEL}) for source separation...")

    # Find which python has demucs
    demucs_python = _find_demucs_python()
    if demucs_python is None:
        raise RuntimeError(
            "Demucs not found in any Python environment. "
            "Install with: pip install demucs  OR  uv pip install demucs"
        )

    # Detect GPU availability
    import shutil
    torch_available = False
    try:
        import torch
        torch_available = True
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB) — Demucs will use CUDA")
        else:
            device = "cpu"
            logger.warning("No GPU detected — Demucs will use CPU (slower)")
    except ImportError:
        device = "cpu"
        logger.warning("PyTorch not found — Demucs will use CPU")

    is_uv = os.path.basename(demucs_python) == "uv"

    if is_uv:
        cmd = [
            demucs_python, "run", "python", "-m", "demucs",
            "--two-stems=vocals",
            "-n", DEMUCS_MODEL,
            "-o", output_dir,
            "--jobs", "2",
            "--device", device,
            raw_audio,
        ]
    else:
        cmd = [
            demucs_python, "-m", "demucs",
            "--two-stems=vocals",
            "-n", DEMUCS_MODEL,
            "-o", output_dir,
            "--jobs", "2",
            "--device", device,
            raw_audio,
        ]

    logger.info(f"Demucs command: {' '.join(cmd)}")

    # Pass full environment so subprocess inherits CUDA libs
    env = os.environ.copy()

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env,
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-5:]:
                logger.info(f"  demucs: {line}")
        logger.info("Demucs separation complete")
    except subprocess.CalledProcessError as e:
        stdout_msg = e.stdout.strip()[-500:] if e.stdout else "(empty)"
        stderr_msg = e.stderr.strip()[-500:] if e.stderr else "(empty)"
        logger.error(f"Demucs failed (rc={e.returncode})")
        logger.error(f"  stdout: {stdout_msg}")
        logger.error(f"  stderr: {stderr_msg}")
        raise RuntimeError(
            f"Demucs separation failed (rc={e.returncode}).\n"
            f"stdout: {stdout_msg}\n"
            f"stderr: {stderr_msg}\n"
            f"Install with: pip install demucs"
        )

    # Step 3: Verify output files exist
    if not os.path.exists(vocals_path):
        # Try to find the files in alternate locations
        alt_vocals = glob.glob(
            os.path.join(output_dir, "**", "vocals.wav"), recursive=True
        )
        if alt_vocals:
            vocals_path = alt_vocals[0]
            no_vocals_path = vocals_path.replace("vocals.wav", "no_vocals.wav")
        else:
            raise FileNotFoundError(
                f"Demucs output not found at {vocals_path}"
            )

    if not os.path.exists(no_vocals_path):
        # Create silent no_vocals as fallback
        logger.warning(
            "no_vocals track not found, creating silent fallback"
        )
        cmd_silent = [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"anullsrc=r=44100:cl=stereo",
            "-i", vocals_path,
            "-t", "0", "-map", "0:a",
            no_vocals_path,
        ]
        subprocess.run(cmd_silent, capture_output=True)

    logger.info(f"Vocals: {vocals_path}")
    logger.info(f"No vocals (BGM): {no_vocals_path}")
    return vocals_path, no_vocals_path


def remix_dubbed_audio(dubbed_vocals_path, no_vocals_path, output_path):
    """
    Merges new dubbed vocals with original background music/SFX.
    Uses amix with weights to keep music slightly lower than dubbing.
    """
    if not os.path.exists(no_vocals_path):
        logger.warning(
            "No background track found, using dubbed vocals only"
        )
        shutil.copy2(dubbed_vocals_path, output_path)
        return output_path

    logger.info("Remixing dubbed vocals with original background...")

    # Get duration of both files to use longest
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
    ]

    try:
        dur1 = float(
            subprocess.check_output(
                probe_cmd + [dubbed_vocals_path], text=True
            ).strip()
        )
        dur2 = float(
            subprocess.check_output(
                probe_cmd + [no_vocals_path], text=True
            ).strip()
        )
    except Exception:
        dur1, dur2 = 0, 0

    # Use the longer duration
    longest = max(dur1, dur2)

    cmd = [
        "ffmpeg", "-y",
        "-i", dubbed_vocals_path,
        "-i", no_vocals_path,
        "-filter_complex",
        (
            f"[0:a]volume=1.0[a1];"
            f"[1:a]volume=0.85[a2];"
            f"[a1][a2]amix=inputs=2:duration=longest"
        ),
        "-t", str(longest),
        "-c:a", "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Remixed audio saved: {output_path}")
    return output_path
