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
print("WORKER VERSION: ai_video_v2", flush=True)

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
SCALE_HEIGHT = int(os.getenv("SCALE_HEIGHT", "720"))
FFMPEG_THREADS = os.getenv("FFMPEG_THREADS", "1")
QUEUE_NAME = os.getenv("QUEUE_NAME", "clipflow:jobs")
MAX_VIDEO_MB = float(os.getenv("MAX_VIDEO_MB", "4.5"))  # stay under 5MB API limit


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
    start_sec: float,
    end_sec: float,
    label: str,
    is_hit: bool | None,
    ai_confidence: float | None,
    ai_reason: str | None,
):
    clip_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                INSERT INTO clips (
                    id, upload_id, bucket, s3_key, start_sec, end_sec, label,
                    is_hit, ai_confidence, ai_reason
                )
                VALUES (
                    :id, :upload_id, :bucket, :s3_key, :start_sec, :end_sec, :label,
                    :is_hit, :ai_confidence, :ai_reason
                )
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
                "is_hit": is_hit,
                "ai_confidence": ai_confidence,
                "ai_reason": ai_reason,
            },
        )
    return clip_id


def run_ffprobe_duration_seconds(source_path: str) -> float:
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
        label = f"segment_{i+1:03d}"
        segments.append((start, end, label))

    return segments


def run_ffmpeg_extract(source_path: str, out_path: str, start_sec: float, duration_sec: float):
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
    if p.stderr:
        print("FFMPEG STDERR:\n", p.stderr, flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {p.returncode}")


def extract_jpeg_frame(video_path: str, out_path: str, offset_sec: float):
    cmd = [
        "ffmpeg",
        "-y",
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


def image_block_from_file(path: str) -> dict:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": b64,
        },
    }


def shrink_clip_for_ai(source_path: str, out_path: str, max_mb: float = 4.5):
    """
    Re-encode the clip to a small size suitable for base64 sending to Claude.
    Targets ~360p, high CRF, low fps — enough for visual analysis.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-vf", "scale=-2:360",
        "-r", "10",           # 10fps — plenty for motion analysis
        "-crf", "35",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-b:a", "32k",
        "-ac", "1",
        out_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"shrink_clip_for_ai failed: {p.stderr or p.stdout}")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Shrunk clip size: {size_mb:.2f} MB", flush=True)

    if size_mb > max_mb:
        raise RuntimeError(
            f"Shrunk clip still too large ({size_mb:.2f} MB > {max_mb} MB). "
            "Consider reducing CLIP_SECONDS."
        )


def get_audio_features(video_path: str) -> str:
    """
    Use ffmpeg astats to detect sharp audio transients (bat crack).
    Returns a plain-text summary Claude can reason about.
    """
    with tempfile.TemporaryDirectory() as tdir:
        stats_path = os.path.join(tdir, "astats.txt")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
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
                    val = float(line.strip().split("=")[1])
                    peaks.append(val)
                except ValueError:
                    pass

        if not peaks:
            return "No audio peak data available."

        max_peak = max(peaks)
        min_peak = min(peaks)
        peak_range = max_peak - min_peak
        has_transient = peak_range > 20

        summary = (
            f"Peak audio level: {max_peak:.1f} dB, "
            f"Min level: {min_peak:.1f} dB, "
            f"Dynamic range: {peak_range:.1f} dB. "
        )
        summary += (
            "A sharp audio transient was detected — possible bat-ball contact."
            if has_transient else
            "No sharp audio transient — less likely to contain bat-ball contact."
        )
        return summary


def classify_clip_with_ai(clip_path: str) -> tuple[bool | None, float | None, str | None]:
    """
    Returns (is_hit, confidence, reason).

    Primary strategy: send the actual video to Claude as base64 so it can
    analyze true motion, swing arc, bat drop, and ball trajectory — not just
    static frames. Falls back to 8-frame analysis if the clip is too large.
    """
    with tempfile.TemporaryDirectory() as tdir:
        clip_duration = run_ffprobe_duration_seconds(clip_path)

        # --- Audio features (always collected regardless of video strategy) ---
        audio_summary = "Audio analysis unavailable."
        try:
            audio_summary = get_audio_features(clip_path)
            print(f"Audio features: {audio_summary}", flush=True)
        except Exception as e:
            print(f"Audio feature extraction skipped: {e}", flush=True)

        # --- Try video strategy first ---
        video_b64 = None
        small_clip_path = os.path.join(tdir, "small.mp4")
        try:
            shrink_clip_for_ai(clip_path, small_clip_path, max_mb=MAX_VIDEO_MB)
            with open(small_clip_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
            print("Using VIDEO strategy for AI classification.", flush=True)
        except Exception as e:
            print(f"Video strategy unavailable ({e}), falling back to frames.", flush=True)

        system_prompt = (
            "You are an expert baseball video analyst specializing in youth Little League games. "
            "Your job is to detect batting highlight moments for a player highlight reel.\n\n"

            "STRONG indicators a clip IS a hit (is_hit=true):\n"
            "- Batter completes a swing with visible bat-ball contact or ball leaving the bat\n"
            "- Ball in flight off the bat (line drive, fly ball, ground ball off bat)\n"
            "- Batter immediately drops bat and starts running toward first base\n"
            "- Sharp crack or impact sound at the moment of swing\n"
            "- Fielders reacting to a live ball — tracking a fly, charging a grounder\n"
            "- Batter running the bases (already past contact moment)\n\n"

            "STRONG indicators a clip is NOT a hit (is_hit=false):\n"
            "- Batter standing still waiting for a pitch (no swing)\n"
            "- Swing and miss — bat completes arc with no contact, ball reaches catcher\n"
            "- Pitcher in wind-up with no batter action visible\n"
            "- Dead time between pitches — batter stepping out, coach visit, timeout\n"
            "- Pure fielding play with no batter involvement\n"
            "- Batter remains in box holding bat after pitch (called strike or ball)\n\n"

            "Key signals to weight heavily:\n"
            "1. BAT DROP + IMMEDIATE RUN = almost certain hit. This is the strongest single signal.\n"
            "2. SHARP AUDIO TRANSIENT at swing moment = strong bat-crack indicator.\n"
            "3. BALL TRAJECTORY CHANGE = ball visibly redirected off bat.\n"
            "4. FOLLOW-THROUGH SWING POSTURE vs. checked/incomplete swing.\n\n"

            "When uncertain, lean toward is_hit=false to avoid false positives. "
            "But if bat drop + running is visible, override uncertainty and call it a hit. "
            "Return strict JSON only."
        )

        if video_b64:
            # ---- VIDEO PATH ----
            user_text = (
                f"Here is a {clip_duration:.1f}-second youth baseball video clip. "
                "Watch for the full motion sequence: pitch arrival, batter swing, "
                "bat-ball contact, ball trajectory, bat drop, and runner movement.\n\n"
                f"Audio analysis: {audio_summary}\n\n"
                "Return ONLY this JSON:\n"
                "{\n"
                "  \"is_hit\": true or false,\n"
                "  \"confidence\": 0.0 to 1.0,\n"
                "  \"reason\": \"1-2 sentences citing specific motion/audio evidence you observed\"\n"
                "}"
            )
            content: list[dict] = [
                {"type": "text", "text": user_text},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "video/mp4",
                        "data": video_b64,
                    },
                },
            ]
        else:
            # ---- FRAME FALLBACK: 8 frames for longer/larger clips ----
            frame_count = 8
            frame_times = [
                min(max(0.1, round(clip_duration * (i / (frame_count - 1)), 3)), clip_duration - 0.1)
                for i in range(frame_count)
            ]
            frame_paths = []
            for i, t in enumerate(frame_times):
                fp = os.path.join(tdir, f"f{i+1}.jpg")
                extract_jpeg_frame(clip_path, fp, t)
                frame_paths.append(fp)
            print(f"Frame fallback — times: {frame_times}", flush=True)

            user_text = (
                f"Here are {frame_count} sequential frames from a {clip_duration:.1f}-second "
                "youth baseball clip, evenly spaced from start to end.\n\n"
                "Analyze as a motion sequence — look for swing arc progression across frames, "
                "bat drop between frames, player starting to run, ball trajectory change, "
                "and fielder reactions.\n\n"
                f"Audio analysis: {audio_summary}\n\n"
                "Return ONLY this JSON:\n"
                "{\n"
                "  \"is_hit\": true or false,\n"
                "  \"confidence\": 0.0 to 1.0,\n"
                "  \"reason\": \"1-2 sentences citing specific visual/audio evidence\"\n"
                "}"
            )
            content = [{"type": "text", "text": user_text}]
            for fp in frame_paths:
                content.append(image_block_from_file(fp))

        resp = anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_parts).strip()
        print("AI RAW RESPONSE:", raw, flush=True)

        try:
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            is_hit = bool(data.get("is_hit"))
            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", ""))[:500]
            return is_hit, confidence, reason
        except Exception:
            return None, None, f"Unparseable AI response: {raw[:500]}"


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

            try:
                print(f"ABOUT TO CALL AI FOR {label}", flush=True)
                is_hit, ai_confidence, ai_reason = classify_clip_with_ai(clip_path)
                print(
                    f"AI RESULT {label}: is_hit={is_hit} confidence={ai_confidence} reason={ai_reason}",
                    flush=True,
                )
            except Exception as e:
                print(f"AI FAILED FOR {label}: {e}", flush=True)
                is_hit = None
                ai_confidence = None
                ai_reason = f"AI failed: {e}"

            # Phase 1: KEEP ALL CLIPS, just store the AI result
            clip_s3_key = f"clips/{upload_id}/{label}_{uuid.uuid4()}.mp4"
            print("Uploading clip to S3:", S3_CLIPS_BUCKET, clip_s3_key, flush=True)
            s3.upload_file(
                clip_path,
                S3_CLIPS_BUCKET,
                clip_s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )

            clip_id = db_insert_clip(
                upload_id=upload_id,
                bucket=S3_CLIPS_BUCKET,
                s3_key=clip_s3_key,
                start_sec=start_sec,
                end_sec=end_sec,
                label=label,
                is_hit=is_hit,
                ai_confidence=ai_confidence,
                ai_reason=ai_reason,
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
        "MODEL=", ANTHROPIC_MODEL,
        "CLIP_SECONDS=", CLIP_SECONDS,
        "MAX_SEGMENTS=", MAX_SEGMENTS,
        "MAX_VIDEO_MB=", MAX_VIDEO_MB,
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