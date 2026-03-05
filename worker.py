import os
import time
import uuid
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
S3_CLIPS_BUCKET = env_required("S3_CLIPS_BUCKET")  # must exist on Render worker


# --- Clients ---
engine = sa.create_engine(DATABASE_URL, pool_pre_ping=True)
r = redis.Redis.from_url(REDIS_URL)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

QUEUE_NAME = "clipflow:jobs"


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


def run_ffmpeg_extract(source_path: str, out_path: str, start_sec: float, duration_sec: float):
    # Single-threaded to reduce RAM use on small instances.
    cmd = [
        "ffmpeg",
        "-y",
        "-threads", "1",
        "-ss", str(start_sec),
        "-i", source_path,
        "-t", str(duration_sec),
        "-vf", "scale=-2:720",
        "-preset", "veryfast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "96k",
        out_path,
    ]
    print("FFMPEG CMD:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    # Always print stderr for debugging (ffmpeg logs to stderr a lot)
    if p.stdout:
        print("FFMPEG STDOUT:\n", p.stdout, flush=True)
    if p.stderr:
        print("FFMPEG STDERR:\n", p.stderr, flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {p.returncode}")


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
        clip_path = os.path.join(tmpdir, "clip.mp4")

        # Download source file to disk (streaming; avoids RAM spikes)
        print("Downloading source from S3...", flush=True)
        s3.download_file(src_bucket, src_key, source_path)
        print("Downloaded to:", source_path, flush=True)

        # For MVP: extract a single 2-second clip starting at 0 sec
        start_sec = 0.0
        duration_sec = 2.0
        end_sec = start_sec + duration_sec

        print("Running ffmpeg...", flush=True)
        run_ffmpeg_extract(source_path, clip_path, start_sec=start_sec, duration_sec=duration_sec)
        print("FFmpeg wrote:", clip_path, flush=True)

        # Upload clip to clips bucket
        clip_s3_key = f"clips/{upload_id}/{uuid.uuid4()}.mp4"
        print("Uploading clip to S3:", S3_CLIPS_BUCKET, clip_s3_key, flush=True)
        s3.upload_file(
            clip_path,
            S3_CLIPS_BUCKET,
            clip_s3_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        print("Uploaded clip OK.", flush=True)

        # Insert clip row in DB
        print("Inserting clip row in DB...", flush=True)
        clip_id = db_insert_clip(
            upload_id=upload_id,
            bucket=S3_CLIPS_BUCKET,
            s3_key=clip_s3_key,
            start_sec=start_sec,
            end_sec=end_sec,
            label="MVP clip",
        )
        print("Inserted clip row OK. clip_id=", clip_id, flush=True)

    db_set_upload_status(upload_id, "complete")
    print("Set status=complete", flush=True)


def main():
    print("ClipFlow worker started. Waiting on Redis queue:", QUEUE_NAME, flush=True)
    print("ENV CHECK:",
          "DB?", bool(os.getenv("DATABASE_URL")),
          "REDIS?", bool(os.getenv("REDIS_URL")),
          "UPLOADS_BUCKET=", S3_UPLOADS_BUCKET,
          "CLIPS_BUCKET=", S3_CLIPS_BUCKET,
          "REGION=", AWS_REGION,
          flush=True)

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
                # mark as error so UI can show it
                try:
                    db_set_upload_status(upload_id, "error")
                except Exception as e2:
                    print("Failed to set error status:", str(e2), flush=True)

        except Exception as outer:
            print("Worker loop error:", str(outer), flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()