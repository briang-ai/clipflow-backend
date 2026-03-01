import os
import time
import uuid
import tempfile
import subprocess

from dotenv import load_dotenv
import redis
import sqlalchemy as sa
import boto3

load_dotenv()

def env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

# Environment
DATABASE_URL = env_required("DATABASE_URL")
REDIS_URL = env_required("REDIS_URL")
AWS_REGION = env_required("AWS_REGION")
AWS_ACCESS_KEY_ID = env_required("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_required("AWS_SECRET_ACCESS_KEY")
S3_CLIPS_BUCKET = env_required("S3_CLIPS_BUCKET")

# Services
engine = sa.create_engine(DATABASE_URL, pool_pre_ping=True)
r = redis.Redis.from_url(REDIS_URL)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

QUEUE = "clipflow:jobs"

def set_status(upload_id: str, status: str):
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE uploads SET status = :status WHERE id = :id"),
            {"status": status, "id": upload_id},
        )

def insert_clip(upload_id: str, bucket: str, s3_key: str):
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
                "start_sec": 0,
                "end_sec": 5,
                "label": "first_5_seconds",
            },
        )
    return clip_id

def process_upload(upload_id: str):
    print(f"Processing upload_id={upload_id}")

    with engine.connect() as conn:
        upload_row = conn.execute(
            sa.text("SELECT bucket, s3_key FROM uploads WHERE id = :id"),
            {"id": upload_id},
        ).mappings().first()

    if not upload_row:
        raise RuntimeError("Upload not found in DB")

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source")
        clip_path = os.path.join(tmpdir, "clip.mp4")

        # Download original video
        s3.download_file(upload_row["bucket"], upload_row["s3_key"], source_path)
        print("Downloaded source video")

        # Get video duration using ffprobe
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                source_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        duration = float(result.stdout.strip())
        print(f"Video duration: {duration} seconds")



        # Generate 5-second clip
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                source_path,
                "-t",
                "5",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                clip_path,
            ],
            check=True,
        )
        print("Generated clip via ffmpeg")

        segment_length = 5
        max_segments = 3  # keep it small for now

        for i in range(max_segments):
            start = i * segment_length
            end = min(start + segment_length, duration)

            if start >= duration:
                break

            clip_filename = f"clip_{start}_{end}.mp4"
            clip_path = os.path.join(tmpdir, clip_filename)

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    source_path,
                    "-ss",
                    str(start),
                    "-t",
                    str(end - start),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-c:a",
                    "aac",
                    clip_path,
                ],
                check=True,
            )

            clip_key = f"clips/{upload_id}/{clip_filename}"

            s3.upload_file(
                clip_path,
                S3_CLIPS_BUCKET,
                clip_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )

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
                        "bucket": S3_CLIPS_BUCKET,
                        "s3_key": clip_key,
                        "start_sec": start,
                        "end_sec": end,
                        "label": f"segment_{start}_{end}",
                    },
                )

                print(f"Created segment clip {start}-{end}")
        

def main():
    print(f"ClipFlow worker started. Waiting on Redis queue: {QUEUE}")

    while True:
        item = r.brpop(QUEUE, timeout=0)
        if not item:
            continue

        _, upload_id_bytes = item
        upload_id = upload_id_bytes.decode("utf-8")

        try:
            set_status(upload_id, "processing")
            print("Set status=processing")

            process_upload(upload_id)

            set_status(upload_id, "ready")
            print("Set status=ready")

        except Exception as e:
            print(f"Worker error for {upload_id}: {e}")
            try:
                set_status(upload_id, "error")
            except Exception:
                pass

if __name__ == "__main__":
    main()