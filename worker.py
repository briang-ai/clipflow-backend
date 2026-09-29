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
from anthropic import Anthropic
from dotenv import load_dotenv

from clipflow_ai.pipeline import analyze


# --- Load env ---
load_dotenv()
print("WORKER VERSION: hitfinder_v1", flush=True)


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

ANTHROPIC_API_KEY = env_required("ANTHROPIC_API_KEY")
# Model that judges each detected swing. Sonnet found 13/13 hits on the test set;
# Haiku is about half the cost. Set HIT_MODEL=claude-haiku-4-5-20251001 to switch.
HIT_MODEL = os.getenv("HIT_MODEL", "claude-sonnet-5")

# --- Tuning ---
HIT_PRE_S = float(os.getenv("HIT_PRE_S", "2.5"))        # seconds kept before the swing
HIT_POST_S = float(os.getenv("HIT_POST_S", "4.0"))      # seconds kept after (ball flight, run)
FULL_CLIP_MAX_S = float(os.getenv("FULL_CLIP_MAX_S", "90"))  # also keep the whole upload if shorter
LONG_CHUNK_S = float(os.getenv("LONG_CHUNK_S", "30"))   # longer uploads: plain chunks to browse
MAX_SEGMENTS = int(os.getenv("MAX_SEGMENTS", "30"))
SCALE_HEIGHT = int(os.getenv("SCALE_HEIGHT", "1080"))
FFMPEG_THREADS = os.getenv("FFMPEG_THREADS", "1")
QUEUE_NAME = os.getenv("QUEUE_NAME", "clipflow:jobs")

# Logo watermark
LOGO_PATH = os.path.join(os.path.dirname(__file__), "app", "logo.png")


# --- Clients ---
engine = sa.create_engine(DATABASE_URL, pool_pre_ping=True)
r = redis.Redis.from_url(REDIS_URL)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------

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


def db_insert_clip(
    upload_id: str,
    bucket: str,
    s3_key: str,
    thumbnail_s3_key: str | None,
    start_sec: float,
    end_sec: float,
    label: str,
    is_hit: bool | None,
    is_swing: bool | None,
    ai_confidence: float | None,
    ai_reason: str | None,
):
    clip_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                INSERT INTO clips (
                    id, upload_id, bucket, s3_key, thumbnail_s3_key,
                    start_sec, end_sec, label,
                    is_hit, is_swing, ai_confidence, ai_reason
                )
                VALUES (
                    :id, :upload_id, :bucket, :s3_key, :thumbnail_s3_key,
                    :start_sec, :end_sec, :label,
                    :is_hit, :is_swing, :ai_confidence, :ai_reason
                )
                """
            ),
            {
                "id": clip_id,
                "upload_id": upload_id,
                "bucket": bucket,
                "s3_key": s3_key,
                "thumbnail_s3_key": thumbnail_s3_key,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": label,
                "is_hit": is_hit,
                "is_swing": is_swing,
                "ai_confidence": ai_confidence,
                "ai_reason": ai_reason,
            },
        )
    return clip_id


def db_get_clips_by_ids(clip_ids: list[str]) -> list[dict]:
    if not clip_ids:
        return []
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                """
                SELECT id, upload_id, bucket, s3_key, start_sec
                FROM clips
                WHERE id = ANY(CAST(:ids AS uuid[]))
                ORDER BY start_sec ASC
                """
            ),
            {"ids": clip_ids},
        ).mappings().all()
    return [dict(r) for r in rows]


def db_set_reel_status(reel_id: str, status: str):
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE reels SET status = :s WHERE id = :id"),
            {"s": status, "id": reel_id},
        )


# ---------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------

def run_ffprobe_duration_seconds(source_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        source_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr or p.stdout}")
    data = json.loads(p.stdout or "{}")
    fmt = data.get("format") or {}
    dur = fmt.get("duration")
    if dur:
        return float(dur)
    for s in data.get("streams") or []:
        if s.get("codec_type") == "video" and s.get("duration"):
            return float(s["duration"])
    raise RuntimeError("Could not determine duration from ffprobe output")


def run_ffmpeg_extract(source_path: str, out_path: str, start_sec: float, duration_sec: float):
    cmd = [
        "ffmpeg", "-y",
        "-threads", str(FFMPEG_THREADS),
        "-ss", str(start_sec),
        "-i", source_path,
        "-t", str(duration_sec),
        "-map_metadata", "-1",          # drop phone metadata (incl. GPS location)
        "-vf", f"scale=-2:{SCALE_HEIGHT}",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # only the tail: the head of ffmpeg's log lists the phone's metadata, GPS included
        print("FFMPEG ERROR:", (p.stderr or "")[-400:], flush=True)
        raise RuntimeError(f"ffmpeg failed with code {p.returncode}")


def extract_jpeg_frame(video_path: str, out_path: str, offset_sec: float):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(offset_sec),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        out_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"frame extraction failed: {p.stderr or p.stdout}")
    if not os.path.exists(out_path):
        raise RuntimeError(f"frame extraction did not create file: {out_path}")


def extract_thumbnail(clip_path: str, tmpdir: str, at_sec: float, clip_duration: float) -> str:
    """Extract a thumbnail JPEG at at_sec into the clip. Returns local path."""
    offset = min(max(0.1, at_sec), max(0.1, clip_duration - 0.1))
    thumb_path = os.path.join(tmpdir, f"thumb_{uuid.uuid4().hex[:8]}.jpg")
    extract_jpeg_frame(clip_path, thumb_path, offset)
    return thumb_path


def call_claude(request: dict) -> dict:
    """One Messages API call; returns the reply as a plain dict."""
    resp = anthropic.messages.create(**request)
    return resp.model_dump()


# ---------------------------------------------------------------
# Upload processor
# ---------------------------------------------------------------

def save_clip(upload_id: str, source_path: str, tmpdir: str, start: float, end: float,
              label: str, thumb_at: float, is_hit, is_swing, confidence, reason):
    """Cut [start, end] from the source, upload clip + thumbnail, insert the row."""
    dur = round(end - start, 3)
    if dur <= 0.2:
        return None
    clip_path = os.path.join(tmpdir, f"{label}.mp4")
    run_ffmpeg_extract(source_path, clip_path, start_sec=round(start, 3), duration_sec=dur)

    thumbnail_s3_key = None
    try:
        thumb_path = extract_thumbnail(clip_path, tmpdir, thumb_at, dur)
        thumbnail_s3_key = f"thumbs/{upload_id}/{label}_{uuid.uuid4()}.jpg"
        s3.upload_file(thumb_path, S3_CLIPS_BUCKET, thumbnail_s3_key,
                       ExtraArgs={"ContentType": "image/jpeg"})
    except Exception as e:
        print(f"Thumbnail failed for {label}: {e}", flush=True)
        thumbnail_s3_key = None

    clip_s3_key = f"clips/{upload_id}/{label}_{uuid.uuid4()}.mp4"
    s3.upload_file(clip_path, S3_CLIPS_BUCKET, clip_s3_key, ExtraArgs={"ContentType": "video/mp4"})
    clip_id = db_insert_clip(
        upload_id=upload_id, bucket=S3_CLIPS_BUCKET, s3_key=clip_s3_key,
        thumbnail_s3_key=thumbnail_s3_key, start_sec=round(start, 3), end_sec=round(end, 3),
        label=label, is_hit=is_hit, is_swing=is_swing, ai_confidence=confidence,
        ai_reason=(reason or "")[:500],
    )
    try:
        os.remove(clip_path)
    except OSError:
        pass
    print(f"Saved clip {label} {start:.2f}-{end:.2f}s hit={is_hit} id={clip_id}", flush=True)
    return clip_id


def process_upload(upload_id: str):
    print(f"Processing upload_id={upload_id}", flush=True)

    upload = db_get_upload(upload_id)
    if not upload:
        print(f"Upload not found in DB: {upload_id}", flush=True)
        return

    db_set_upload_status(upload_id, "processing")

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source")
        s3.download_file(upload["bucket"], upload["s3_key"], source_path)
        duration = run_ffprobe_duration_seconds(source_path)
        print(f"Detected duration: {duration:.3f} seconds", flush=True)

        # 1) find and judge the swings
        result = analyze(source_path, call_claude, HIT_MODEL,
                         log=lambda s: print(s, flush=True))
        ai_down = result.ai_calls == 0 and bool(result.ai_errors)
        if result.ai_errors:
            print(f"AI problems for {upload_id}: {result.ai_errors}", flush=True)

        created = 0
        # 2) one trimmed clip per swing, hits marked
        swings = sorted(result.swings, key=lambda m: m.t)
        if ai_down:
            # AI unavailable (e.g. no API credit): keep the loudest moments, unscored,
            # so the family can still pick their hits by hand.
            swings = sorted(sorted(result.moments, key=lambda m: -m.audio_z)[:4], key=lambda m: m.t)
        for n, m in enumerate(swings, 1):
            start = max(0.0, m.t - HIT_PRE_S)
            end = min(duration, m.t + HIT_POST_S)
            if ai_down:
                label, is_hit, is_swing, conf = f"moment_{n:02d}", None, None, None
                reason = "AI check unavailable - not scored"
            elif m.possible_hit:
                # blocked view (netting / other cages): let the family confirm
                label, is_hit, is_swing, conf = f"maybe_{n:02d}", None, True, m.confidence
                reason = ("Possible hit - netting or other hitters kept the AI from seeing "
                          f"contact clearly. Mark it if it was a hit. ({m.reason})")
            else:
                label = f"{'hit' if m.final_hit else 'swing'}_{n:02d}"
                is_hit, is_swing, conf = bool(m.final_hit), True, m.confidence
                reason = f"{m.label.replace('_', ' ')}: {m.reason}"
                if m.label == "hit" and not m.final_hit:
                    reason = "likely foul or duplicate of a louder hit - " + reason
            if save_clip(upload_id, source_path, tmpdir, start, end, label, m.t - start,
                         is_hit, is_swing, conf, reason):
                created += 1

        # 3) the rest of the footage, so nothing the AI missed is lost
        if duration <= FULL_CLIP_MAX_S:
            if save_clip(upload_id, source_path, tmpdir, 0.0, duration, "full_clip",
                         min(1.0, duration / 2), None, False, None, "Whole upload"):
                created += 1
        else:
            n_chunks = min(MAX_SEGMENTS, int(math.ceil(duration / LONG_CHUNK_S)))
            for i in range(n_chunks):
                a, b = i * LONG_CHUNK_S, min(duration, (i + 1) * LONG_CHUNK_S)
                if save_clip(upload_id, source_path, tmpdir, a, b, f"part_{i + 1:03d}",
                             min(1.0, (b - a) / 2), None, False, None, "Part of the full upload"):
                    created += 1

        print(f"Created {created} clips for upload_id={upload_id}; "
              f"AI cost ${result.cost_usd:.4f} ({result.input_tokens} in / {result.output_tokens} out)",
              flush=True)

    db_set_upload_status(upload_id, "complete")
    print("Set status=complete", flush=True)


# ---------------------------------------------------------------
# Reel compiler
# ---------------------------------------------------------------

def process_compile_reel(job: dict):
    reel_id       = job["reel_id"]
    user_id       = job["user_id"]
    player_name   = job.get("player_name", "unknown")
    jersey_number = job.get("jersey_number", "")
    game_date     = job.get("game_date", "unknown_date")
    clip_ids      = job.get("clip_ids", [])
    watermark     = job.get("watermark", True)

    print(f"compile_reel reel_id={reel_id} player={player_name} clips={len(clip_ids)} watermark={watermark}", flush=True)

    if not clip_ids:
        print("No clip_ids provided — aborting.", flush=True)
        db_set_reel_status(reel_id, "error")
        return

    db_set_reel_status(reel_id, "processing")
    clips = db_get_clips_by_ids(clip_ids)
    if not clips:
        print("No clips found in DB.", flush=True)
        db_set_reel_status(reel_id, "error")
        return

    print(f"Found {len(clips)} clips to stitch.", flush=True)

    logo_available = watermark and os.path.exists(LOGO_PATH)
    if logo_available:
        print(f"Using logo watermark from: {LOGO_PATH}", flush=True)
    elif watermark:
        print(f"WARNING: logo.png not found at {LOGO_PATH}, falling back to text watermark", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_paths = []

        for i, clip in enumerate(clips):
            raw_path  = os.path.join(tmpdir, f"clip_{i:03d}_raw.mp4")
            norm_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
            print(f"Downloading clip {clip['id']} from s3://{clip['bucket']}/{clip['s3_key']}", flush=True)
            s3.download_file(clip["bucket"], clip["s3_key"], raw_path)

            # Normalize every clip to identical specs and bake in the
            # watermark per-clip. This means the final concat is a pure
            # stream copy with nothing that can break mid-video.
            # scale=-2:720 → portrait-first 720px tall, width auto even.
            # 30fps CFR, 44100Hz stereo AAC ensures uniform timebases.
            # Logo scaled to 86px wide (~20% larger than previous 72px).
            if logo_available:
                filter_complex = (
                    f"[1:v]scale=86:-1,format=rgba,colorchannelmixer=aa=0.8[wm];"
                    f"[0:v]scale=-2:720,setsar=1[base];"
                    f"[base][wm]overlay=W-w-20:H-h-20[out]"
                )
                norm_cmd = [
                    "ffmpeg", "-y",
                    "-i", raw_path,
                    "-i", LOGO_PATH,
                    "-filter_complex", filter_complex,
                    "-map", "[out]", "-map", "0:a?",
                    "-r", "30", "-vsync", "cfr",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                    norm_path,
                ]
            elif watermark:
                norm_cmd = [
                    "ffmpeg", "-y", "-i", raw_path,
                    "-vf", (
                        "scale=-2:720,setsar=1,"
                        "drawtext=text='clipflow.pro':fontsize=28:fontcolor=white@0.6:"
                        "shadowcolor=black@0.5:shadowx=1:shadowy=1:x=w-tw-20:y=h-th-20"
                    ),
                    "-r", "30", "-vsync", "cfr",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                    norm_path,
                ]
            else:
                norm_cmd = [
                    "ffmpeg", "-y", "-i", raw_path,
                    "-vf", "scale=-2:720,setsar=1",
                    "-r", "30", "-vsync", "cfr",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                    norm_path,
                ]

            np = subprocess.run(norm_cmd, capture_output=True, text=True)
            if np.returncode != 0:
                print(f"Normalize clip {i} error: {(np.stderr or '')[-400:]}", flush=True)
                print(f"Normalize failed for clip {i} — aborting reel.", flush=True)
                db_set_reel_status(reel_id, "error")
                return
            local_paths.append(norm_path)

        concat_list_path = os.path.join(tmpdir, "concat.txt")
        with open(concat_list_path, "w") as f:
            for p in local_paths:
                escaped = p.replace("\\", "/").replace("'", "\\'")
                f.write(f"file '{escaped}'\n")

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in player_name)
        output_filename = f"highlights_{safe_name}_{game_date}.mp4"
        output_path = os.path.join(tmpdir, output_filename)

        # Watermark already baked per-clip — final concat is a pure stream copy
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-map_metadata", "-1",
            "-c", "copy",
            output_path,
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print("FFMPEG CONCAT ERROR:", (p.stderr or "")[-400:], flush=True)
            raise RuntimeError(f"ffmpeg concat failed with code {p.returncode}")

        reel_s3_key = f"reels/{user_id}/{reel_id}/{output_filename}"
        s3.upload_file(output_path, S3_CLIPS_BUCKET, reel_s3_key, ExtraArgs={"ContentType": "video/mp4"})

        with engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE reels SET s3_key = :key, status = 'complete' WHERE id = :id"),
                {"key": reel_s3_key, "id": reel_id},
            )

        print(f"Reel complete. reel_id={reel_id} s3_key={reel_s3_key}", flush=True)


# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------

def main():
    print("ClipFlow worker started. Waiting on Redis queue:", QUEUE_NAME, flush=True)
    print(
        "ENV CHECK:",
        "DB?", bool(os.getenv("DATABASE_URL")),
        "REDIS?", bool(os.getenv("REDIS_URL")),
        "UPLOADS_BUCKET=", S3_UPLOADS_BUCKET,
        "CLIPS_BUCKET=", S3_CLIPS_BUCKET,
        "REGION=", AWS_REGION,
        "HIT_MODEL=", HIT_MODEL,
        "MAX_SEGMENTS=", MAX_SEGMENTS,
        "SCALE_HEIGHT=", SCALE_HEIGHT,
        "LOGO_PATH=", LOGO_PATH,
        "LOGO_EXISTS=", os.path.exists(LOGO_PATH),
        flush=True,
    )

    while True:
        try:
            item = r.brpop(QUEUE_NAME, timeout=10)
            if not item:
                continue

            _, job_bytes = item
            raw = job_bytes.decode("utf-8")

            try:
                job = json.loads(raw)
                job_type = job.get("type")
            except json.JSONDecodeError:
                job = None
                job_type = None

            print(f"Got job type={job_type or 'upload'} raw={raw[:80]}", flush=True)

            try:
                if job_type == "compile_reel":
                    process_compile_reel(job)
                else:
                    process_upload(raw)
            except Exception as e:
                print(f"Worker error: {e}", flush=True)
                if job_type == "compile_reel" and job:
                    try:
                        db_set_reel_status(job["reel_id"], "error")
                    except Exception as e2:
                        print("Failed to set reel error status:", str(e2), flush=True)
                else:
                    try:
                        db_set_upload_status(raw, "error")
                    except Exception as e2:
                        print("Failed to set upload error status:", str(e2), flush=True)

        except Exception as outer:
            print("Worker loop error:", str(outer), flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()