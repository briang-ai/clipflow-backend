import os
import time
import uuid
import math
import json
import base64
import tempfile
import subprocess

import boto3
import redis
import sqlalchemy as sa
from anthropic import Anthropic
from dotenv import load_dotenv


# --- Load env ---
load_dotenv()
print("WORKER VERSION: thumbnails_v3", flush=True)


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
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
AI_MIN_CONFIDENCE = float(os.getenv("AI_MIN_CONFIDENCE", "0.70"))

# --- Tuning ---
CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "8"))
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


def build_segments(duration_sec: float, clip_seconds: float) -> list[tuple[float, float, str]]:
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
        segments.append((start, end, f"segment_{i+1:03d}"))
    return segments


def run_ffmpeg_extract(source_path: str, out_path: str, start_sec: float, duration_sec: float):
    cmd = [
        "ffmpeg", "-y",
        "-threads", str(FFMPEG_THREADS),
        "-ss", str(start_sec),
        "-i", source_path,
        "-t", str(duration_sec),
        "-vf", f"scale=-2:{SCALE_HEIGHT}",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]
    print("FFMPEG CMD:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stderr:
        print("FFMPEG STDERR:\n", p.stderr, flush=True)
    if p.returncode != 0:
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


def extract_thumbnail(clip_path: str, tmpdir: str, clip_duration: float) -> str:
    """Extract a thumbnail JPEG from ~10% into the clip. Returns local path."""
    offset = min(max(0.5, clip_duration * 0.1), clip_duration - 0.1)
    thumb_path = os.path.join(tmpdir, "thumb.jpg")
    extract_jpeg_frame(clip_path, thumb_path, offset)
    return thumb_path


def image_block_from_file(path: str) -> dict:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
    }


# ---------------------------------------------------------------
# AI classification
# ---------------------------------------------------------------

def extract_frames_for_ai(clip_path: str, tdir: str, clip_duration: float, count: int = 10) -> list[str]:
    frame_times = [
        min(max(0.1, round(clip_duration * (i / (count - 1)), 3)), clip_duration - 0.1)
        for i in range(count)
    ]
    frame_paths = []
    for i, t in enumerate(frame_times):
        fp = os.path.join(tdir, f"f{i+1:02d}.jpg")
        extract_jpeg_frame(clip_path, fp, t)
        frame_paths.append(fp)
    print(f"Extracted {count} frames at times: {frame_times}", flush=True)
    return frame_paths


def get_audio_features(video_path: str) -> str:
    with tempfile.TemporaryDirectory() as tdir:
        stats_path = os.path.join(tdir, "astats.txt")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-af", "astats=metadata=1:reset=1,ametadata=print:file=" + stats_path,
            "-vn", "-f", "null", "-",
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(stats_path):
            return "Audio stats unavailable."
        with open(stats_path) as f:
            lines = f.readlines()
        peaks = []
        for line in lines:
            if "lavfi.astats.Overall.Peak_level" in line:
                try:
                    peaks.append(float(line.strip().split("=")[1]))
                except ValueError:
                    pass
        if not peaks:
            return "No audio peak data available."
        max_peak = max(peaks)
        min_peak = min(peaks)
        peak_range = max_peak - min_peak
        summary = (f"Peak audio level: {max_peak:.1f} dB, Min level: {min_peak:.1f} dB, "
                   f"Dynamic range: {peak_range:.1f} dB. ")
        summary += ("A sharp audio transient was detected — possible bat-ball contact."
                    if peak_range > 20 else
                    "No sharp audio transient — less likely to contain bat-ball contact.")
        return summary


def classify_clip_with_ai(clip_path: str) -> tuple[bool | None, bool | None, float | None, str | None]:
    with tempfile.TemporaryDirectory() as tdir:
        clip_duration = run_ffprobe_duration_seconds(clip_path)
        audio_summary = "Audio analysis unavailable."
        try:
            audio_summary = get_audio_features(clip_path)
            print(f"Audio features: {audio_summary}", flush=True)
        except Exception as e:
            print(f"Audio feature extraction skipped: {e}", flush=True)

        frame_paths = extract_frames_for_ai(clip_path, tdir, clip_duration, count=10)

        system_prompt = (
            "You are an expert baseball video analyst specializing in youth Little League games. "
            "Your job is to classify batting moments for a player highlight reel.\n\n"
            "You must return TWO boolean values:\n\n"
            "1. is_hit — true ONLY if the batter makes confirmed contact and the ball leaves the bat:\n"
            "   TRUE: bat-ball contact visible, ball in flight, batter drops bat and runs, "
            "sharp audio crack, fielders reacting, batter already on bases.\n"
            "   FALSE: swing and miss, batter standing still, dead time, fielding only.\n\n"
            "2. is_swing — true if the batter attempts ANY full swing, hit OR miss:\n"
            "   TRUE: full swing arc visible regardless of contact. Swing and miss = is_swing true, is_hit false.\n"
            "   FALSE: no swing initiated, checked swing, dead time, fielding plays.\n\n"
            "Rule: if is_hit=true then is_swing must also be true.\n"
            "Return strict JSON only."
        )

        user_text = (
            f"Here are 10 sequential frames from a {clip_duration:.1f}-second youth baseball clip. "
            "Treat them as a flip-book.\n\n"
            "Classify: did the batter swing? Did they make contact?\n\n"
            f"Audio analysis: {audio_summary}\n\n"
            "Return ONLY this JSON:\n"
            "{\n"
            "  \"is_hit\": true or false,\n"
            "  \"is_swing\": true or false,\n"
            "  \"confidence\": 0.0 to 1.0,\n"
            "  \"reason\": \"1-2 sentences citing specific visual evidence\"\n"
            "}"
        )

        content: list[dict] = [{"type": "text", "text": user_text}]
        for fp in frame_paths:
            content.append(image_block_from_file(fp))

        resp = anthropic.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_parts).strip()
        print("AI RAW RESPONSE:", raw, flush=True)

        try:
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            is_hit   = bool(data.get("is_hit"))
            is_swing = bool(data.get("is_swing"))
            if is_hit:
                is_swing = True
            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", ""))[:500]
            return is_hit, is_swing, confidence, reason
        except Exception:
            return None, None, None, f"Unparseable AI response: {raw[:500]}"


# ---------------------------------------------------------------
# Upload processor
# ---------------------------------------------------------------

def process_upload(upload_id: str):
    print(f"Processing upload_id={upload_id}", flush=True)

    upload = db_get_upload(upload_id)
    if not upload:
        print(f"Upload not found in DB: {upload_id}", flush=True)
        return

    src_bucket = upload["bucket"]
    src_key = upload["s3_key"]

    db_set_upload_status(upload_id, "processing")
    print("Set status=processing", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source")
        print("Downloading source from S3...", flush=True)
        s3.download_file(src_bucket, src_key, source_path)

        duration = run_ffprobe_duration_seconds(source_path)
        print(f"Detected duration: {duration:.3f} seconds", flush=True)

        segments = build_segments(duration, CLIP_SECONDS)
        print(f"Built {len(segments)} segments of ~{CLIP_SECONDS}s", flush=True)

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

            # ── Generate thumbnail ──────────────────────────────────────────
            thumbnail_s3_key = None
            try:
                thumb_path = extract_thumbnail(clip_path, tmpdir, seg_dur)
                thumbnail_s3_key = f"thumbs/{upload_id}/{label}_{uuid.uuid4()}.jpg"
                s3.upload_file(
                    thumb_path, S3_CLIPS_BUCKET, thumbnail_s3_key,
                    ExtraArgs={"ContentType": "image/jpeg"},
                )
                print(f"Thumbnail uploaded: {thumbnail_s3_key}", flush=True)
            except Exception as e:
                print(f"Thumbnail failed for {label}: {e}", flush=True)
                thumbnail_s3_key = None

            # ── AI classification ───────────────────────────────────────────
            try:
                print(f"ABOUT TO CALL AI FOR {label}", flush=True)
                is_hit, is_swing, ai_confidence, ai_reason = classify_clip_with_ai(clip_path)
                print(
                    f"AI RESULT {label}: is_hit={is_hit} is_swing={is_swing} "
                    f"confidence={ai_confidence} reason={ai_reason}",
                    flush=True,
                )
            except Exception as e:
                print(f"AI FAILED FOR {label}: {e}", flush=True)
                is_hit = None
                is_swing = None
                ai_confidence = None
                ai_reason = f"AI failed: {e}"

            # ── Upload clip ─────────────────────────────────────────────────
            clip_s3_key = f"clips/{upload_id}/{label}_{uuid.uuid4()}.mp4"
            s3.upload_file(
                clip_path, S3_CLIPS_BUCKET, clip_s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )

            clip_id = db_insert_clip(
                upload_id=upload_id,
                bucket=S3_CLIPS_BUCKET,
                s3_key=clip_s3_key,
                thumbnail_s3_key=thumbnail_s3_key,
                start_sec=start_sec,
                end_sec=end_sec,
                label=label,
                is_hit=is_hit,
                is_swing=is_swing,
                ai_confidence=ai_confidence,
                ai_reason=ai_reason,
            )
            created += 1
            print(f"Inserted clip row OK. clip_id={clip_id}", flush=True)

        print(f"Created {created} clips for upload_id={upload_id}", flush=True)

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

    with tempfile.TemporaryDirectory() as tmpdir:
        local_paths = []
        for i, clip in enumerate(clips):
            raw_path = os.path.join(tmpdir, f"clip_{i:03d}_raw.mp4")
            norm_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
            print(f"Downloading clip {clip['id']} from s3://{clip['bucket']}/{clip['s3_key']}", flush=True)
            s3.download_file(clip["bucket"], clip["s3_key"], raw_path)

            # Normalize every clip to the same resolution, pixel format,
            # timebase, and audio sample rate before concatenating.
            # This eliminates mismatched stream properties that cause
            # the overlay filter to drop, audio stutter, and DTS errors.
            norm_cmd = [
                "ffmpeg", "-y", "-i", raw_path,
                "-vf", "scale=720:-2,setsar=1",
                "-r", "30",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                "-vsync", "cfr",
                norm_path,
            ]
            np = subprocess.run(norm_cmd, capture_output=True, text=True)
            if np.returncode != 0:
                print(f"Normalize failed for clip {i}, using raw: {np.stderr[-300:]}", flush=True)
                local_paths.append(raw_path)
            else:
                local_paths.append(norm_path)

        concat_list_path = os.path.join(tmpdir, "concat.txt")
        with open(concat_list_path, "w") as f:
            for p in local_paths:
                escaped = p.replace("\\", "/").replace("'", "\\'")
                f.write(f"file '{escaped}'\n")

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in player_name)
        output_filename = f"highlights_{safe_name}_{game_date}.mp4"
        output_path = os.path.join(tmpdir, output_filename)

        logo_available = watermark and os.path.exists(LOGO_PATH)

        if logo_available:
            print(f"Using logo watermark from: {LOGO_PATH}", flush=True)
            # Scale logo to 72px wide (visible on portrait 406px wide video,
            # unobtrusive on landscape). Simple explicit scale avoids
            # scale2ref ordering bugs. aresample=async=1 fixes non-monotonous
            # DTS audio stuttering from concatenating clips recorded at
            # different times.
            filter_complex = (
                "[1:v]scale=72:-1,format=rgba,colorchannelmixer=aa=0.8[wm];"
                "[0:v][wm]overlay=W-w-20:H-h-20[out]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-i", LOGO_PATH,
                "-filter_complex", filter_complex,
                "-map", "[out]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                output_path,
            ]
        elif watermark:
            print(f"WARNING: logo.png not found at {LOGO_PATH}, falling back to text watermark", flush=True)
            drawtext = (
                "drawtext=text='clipflow.pro':fontsize=28:fontcolor=white@0.6:"
                "shadowcolor=black@0.5:shadowx=1:shadowy=1:x=w-tw-20:y=h-th-20"
            )
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-vf", drawtext,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                output_path,
            ]

        print("FFMPEG CONCAT CMD:", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.stderr:
            print("FFMPEG CONCAT STDERR:\n", p.stderr, flush=True)
        if p.returncode != 0:
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
        "MODEL=", ANTHROPIC_MODEL,
        "CLIP_SECONDS=", CLIP_SECONDS,
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