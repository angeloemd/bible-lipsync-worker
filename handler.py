"""
RunPod serverless handler para LatentSync.

LatentSync sincroniza labios sobre un VIDEO existente + audio nuevo — no
genera movimiento desde una foto fija por sí solo. Como los avatars del
proyecto (Sofía/Elías/Samuel) son fotos fijas, si el input es una imagen
la convertimos primero en un "video" de frame estático (misma duración que
el audio) con ffmpeg, y LatentSync le sincroniza la boca sobre eso.

Nota de calidad: un frame estático da lip-sync preciso pero sin parpadeo ni
micro-movimiento de cabeza. Si más adelante se genera un loop corto de
"idle motion" por avatar (una vez, no por video diario) con Higgsfield,
pasar ese loop como video_url en vez de image_url sube la calidad notablemente
sin cambiar nada de este handler.

Input esperado (event["input"]) — acepta base64 directo (sin hosting externo)
o URL, lo que sea más práctico según el caso:
  - image_base64 / image_url   (o) video_base64 / video_url : avatar base
  - audio_base64 / audio_url                                : audio narrado (mp3/wav)
  - fps                                                      : opcional, default 25

Output:
  - video_base64                : el mp4 resultante, base64
"""

import base64
import os
import subprocess
import tempfile
import uuid

import requests
import runpod

LATENTSYNC_DIR = "/workspace/latentsync"


def download(url, dest_path):
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    return dest_path


def save_base64(b64_str, dest_path):
    with open(dest_path, "wb") as f:
        f.write(base64.b64decode(b64_str))
    return dest_path


def resolve_input(job_input, base64_key, url_key, dest_path):
    if job_input.get(base64_key):
        return save_base64(job_input[base64_key], dest_path)
    if job_input.get(url_key):
        return download(job_input[url_key], dest_path)
    return None


def probe_duration_seconds(path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def image_to_static_video(image_path, audio_path, out_path, fps=25):
    duration = probe_duration_seconds(audio_path)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264", "-t", str(duration),
            "-pix_fmt", "yuv420p", "-vf", f"fps={fps}",
            "-shortest", out_path,
        ],
        check=True, capture_output=True,
    )
    return out_path


def run_latentsync(video_path, audio_path, out_path):
    subprocess.run(
        [
            "python", "-m", "scripts.inference",
            "--unet_config_path", "configs/unet/stage2_512.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--inference_steps", "20",
            "--guidance_scale", "1.5",
            "--enable_deepcache",
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", out_path,
        ],
        cwd=LATENTSYNC_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return out_path


def handler(event):
    job_input = event["input"]
    work_dir = tempfile.mkdtemp(prefix="latentsync_job_")
    job_id = uuid.uuid4().hex[:8]

    try:
        audio_path = resolve_input(job_input, "audio_base64", "audio_url", os.path.join(work_dir, "audio.wav"))
        if not audio_path:
            return {"error": "Falta audio_base64 o audio_url en el input"}

        video_path = resolve_input(job_input, "video_base64", "video_url", os.path.join(work_dir, "base.mp4"))
        if video_path:
            base_video_path = video_path
        else:
            image_path = resolve_input(job_input, "image_base64", "image_url", os.path.join(work_dir, "image.png"))
            if not image_path:
                return {"error": "Falta image_base64/image_url o video_base64/video_url en el input"}
            base_video_path = image_to_static_video(
                image_path, audio_path, os.path.join(work_dir, "base.mp4"),
                fps=job_input.get("fps", 25),
            )

        out_path = os.path.join(work_dir, f"out_{job_id}.mp4")
        run_latentsync(base_video_path, audio_path, out_path)

        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"video_base64": video_b64}

    except subprocess.CalledProcessError as e:
        return {"error": f"Fallo de comando: {e.cmd}\nstdout: {e.stdout}\nstderr: {e.stderr}"}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
# rebuild trigger 1785847394
