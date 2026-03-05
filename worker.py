import os
import time
import uuid
import math
import json
import tempfile
import subprocess

import boto3
import redis
import sqlalchemy as sa
from dotenv import load_dotenv


# --- Load env ---
load_dotenv()

def env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


DATABASE_URL = env_required("DATABASE_URL")
REDIS_URL = env_required("REDIS_URL")

AWS_REGION = env_required("AWS_REGION")
AWS_ACCESS_KEY_ID = env_required("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_required("AWS_SECRET_ACCESS_KEY")

S3_UPLOADS_BUCKET = env_required("S3_UPLOADS_BUCKET")
S3_CLIPS_BUCKET = env_required("S3_CLIPS_BUCKET")


# --- Tuning (safe defaults for Render) ---
CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "5"))        # 5-second segments
MAX_SEGMENTS = int(os.getenv("MAX_SEGMENTS", "60"))         # safety cap: 60 * 5s = first 5 minutes
SCALE_HEIGHT = int(os.getenv("SCALE_HEIGHT", "720"))        # output height for clips
FFMPEG_THREADS = os.getenv("FFMPEG_THREADS", "1")           # keep low to reduce memory
QUEUE_NAME = os.getenv("QUEUE_NAME", "clipflow:jobs")


# --- Clients ---
engine = sa.create_engine(DATABASE_URL, pool_pre_ping=True)
r = redis.Redis.from_url(REDIS_URL)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def db_get_upload(upload_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT id, bucket, s3_key, content_type FROM uploads WHERE id = :id"),
            {"id": upload_id},
        ).mappings().first()
    return row


def db_set_upload_status(upload_id: str, status: str):
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE uploads SET status = :s WHERE id = :id"),
            {"s": status, "id": upload_id},
        )


def db_insert_clip(upload_id: str, bucket: str, s3_key: str, start_sec: float, end_sec: float, label: str):
    clip_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                INSERT INTO clips (id, upload_id, bucket, s3_key, start_sec, end_sec, label)
                VALUES (:id, :upload_id, :bucket, :s3_key, :start_sec, :end_sec, :label)
                """
            ),
            {
                "id": clip_id,
                "upload_id": upload_id,
                "bucket": bucket,
                "s3_key": s3_key,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": label,
            },
        )
    return clip_id


def run_ffprobe_duration_seconds(source_path: str) -> float:
    """
    Returns duration in seconds (float). Uses ffprobe JSON output.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        source_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr or p.stdout}")

    data = json.loads(p.stdout or "{}")

    # Prefer format.duration
    fmt = data.get("format") or {}
    dur = fmt.get("duration")
    if dur:
        return float(dur)

    # Fallback: find video stream duration
    for s in data.get("streams") or []:
        if s.get("codec_type") == "video" and s.get("duration"):
            return float(s["duration"])

    raise RuntimeError("Could not determine duration from ffprobe output")


def build_segments(duration_sec: float, clip_seconds: float) -> list[tuple[float, float, str]]:
    """
    Returns list of (start_sec, end_sec, label).
    """
    duration_sec = float(duration_sec or 0)
    if duration_sec <= 0:
        return []

    clip_seconds = max(1.0, float(clip_seconds))
    count = int(math.ceil(duration_sec / clip_seconds))

    segments: list[tuple[float, float, str]] = []
    for i in range(count):
        start = round(i * clip_seconds, 3)
        end = round(min((i + 1) * clip_seconds, duration_sec), 3)
        if end <= start:
            continue
        label = f"segment_{i+1:03d}"
        segments.append((start, end, label))

    return segments


def run_ffmpeg_extract(source_path: str, out_path: str, start_sec: float, duration_sec: float):
    """
    Extract clip to out_path. Single-threaded + scaled output to reduce RAM/CPU.
    """
    # -ss before -i is faster seek; good for chunking many segments
    cmd = [
        "ffmpeg",
        "-y",
        "-threads", str(FFMPEG_THREADS),
        "-ss", str(start_sec),
        "-i", source_path,
        "-t", str(duration_sec),
        "-vf", f"scale=-2:{SCALE_HEIGHT}",
        "-preset", "veryfast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "96k",
        out_path,
    ]
    print("FFMPEG CMD:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)

    # ffmpeg logs to stderr a lot; keep it for debugging
    if p.stderr:
        print("FFMPEG STDERR:\n", p.stderr, flush=True)

    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {p.returncode}")


def is_highlight_candidate(clip_path: str) -> bool:
    """
    FUTURE: replace with Anthropic vision classification ("hit" vs "not a hit").
    For now, keep everything.
    """
    return True


def process_upload(upload_id: str):
    print(f"Processing upload_id={upload_id}", flush=True)

    upload = db_get_upload(upload_id)
    if not upload:
        print(f"Upload not found in DB: {upload_id}", flush=True)
        return

    src_bucket = upload["bucket"]
    src_key = upload["s3_key"]

    print("Source:", src_bucket, src_key, flush=True)
    db_set_upload_status(upload_id, "processing")
    print("Set status=processing", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source")
        print("Downloading source from S3...", flush=True)
        s3.download_file(src_bucket, src_key, source_path)
        print("Downloaded to:", source_path, flush=True)

        # Determine duration
        duration = run_ffprobe_duration_seconds(source_path)
        print(f"Detected duration: {duration:.3f} seconds", flush=True)

        # Build segments (5 seconds each)
        segments = build_segments(duration, CLIP_SECONDS)
        print(f"Built {len(segments)} segments of ~{CLIP_SECONDS}s", flush=True)

        # Safety cap (prevents Render OOM)
        if MAX_SEGMENTS > 0 and len(segments) > MAX_SEGMENTS:
            print(f"MAX_SEGMENTS cap: trimming {len(segments)} -> {MAX_SEGMENTS}", flush=True)
            segments = segments[:MAX_SEGMENTS]

        created = 0

        for (start_sec, end_sec, label) in segments:
            seg_dur = round(end_sec - start_sec, 3)
            if seg_dur <= 0:
                continue

            clip_path = os.path.join(tmpdir, f"{label}.mp4")

            print(f"Segment {label}: start={start_sec} dur={seg_dur}", flush=True)
            run_ffmpeg_extract(source_path, clip_path, start_sec=start_sec, duration_sec=seg_dur)

            # FUTURE: classify as "hit" or "not"
            if not is_highlight_candidate(clip_path):
                print(f"Skipping (not a highlight): {label}", flush=True)
                continue

            # Upload clip
            clip_s3_key = f"clips/{upload_id}/{label}_{uuid.uuid4()}.mp4"
            print("Uploading clip to S3:", S3_CLIPS_BUCKET, clip_s3_key, flush=True)
            s3.upload_file(
                clip_path,
                S3_CLIPS_BUCKET,
                clip_s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )

            # Insert DB row
            clip_id = db_insert_clip(
                upload_id=upload_id,
                bucket=S3_CLIPS_BUCKET,
                s3_key=clip_s3_key,
                start_sec=start_sec,
                end_sec=end_sec,
                label=label,
            )
            created += 1
            print(f"Inserted clip row OK. clip_id={clip_id}", flush=True)

        print(f"Created {created} clips for upload_id={upload_id}", flush=True)

    db_set_upload_status(upload_id, "complete")
    print("Set status=complete", flush=True)


def main():
    print("ClipFlow worker started. Waiting on Redis queue:", QUEUE_NAME, flush=True)
    print(
        "ENV CHECK:",
        "DB?", bool(os.getenv("DATABASE_URL")),
        "REDIS?", bool(os.getenv("REDIS_URL")),
        "UPLOADS_BUCKET=", S3_UPLOADS_BUCKET,
        "CLIPS_BUCKET=", S3_CLIPS_BUCKET,
        "REGION=", AWS_REGION,
        "CLIP_SECONDS=", CLIP_SECONDS,
        "MAX_SEGMENTS=", MAX_SEGMENTS,
        flush=True,
    )

    while True:
        try:
            item = r.brpop(QUEUE_NAME, timeout=10)
            if not item:
                continue

            _, upload_id_bytes = item
            upload_id = upload_id_bytes.decode("utf-8")
            print("Got job upload_id=" + upload_id, flush=True)

            try:
                process_upload(upload_id)
            except Exception as e:
                print(f"Worker error for {upload_id}: {e}", flush=True)
                try:
                    db_set_upload_status(upload_id, "error")
                except Exception as e2:
                    print("Failed to set error status:", str(e2), flush=True)

        except Exception as outer:
            print("Worker loop error:", str(outer), flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()