import os
import uuid
from typing import Optional

import boto3
import redis
import sqlalchemy as sa
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

"ANTHROPIC_KEY?", bool(os.getenv("ANTHROPIC_API_KEY")),
"MODEL=", ANTHROPIC_MODEL,

# Load .env for local dev only; Render uses its own Environment settings
load_dotenv()

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="ClipFlow API")

# -----------------------------
# CORS
# -----------------------------
# Allow your Vercel domains + local dev.
# Keep this list tight (not "*") since allow_credentials=True.
ALLOWED_ORIGINS = [
    "https://clipflow.pro",
    "https://www.clipflow.pro",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure all OPTIONS (preflight) requests succeed
@app.options("/{path:path}")
def preflight_handler(path: str):
    return Response(status_code=204)


# -----------------------------
# Env helpers
# -----------------------------
def env_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


DATABASE_URL = env_required("DATABASE_URL")
REDIS_URL = env_required("REDIS_URL")

AWS_REGION = env_required("AWS_REGION")
AWS_ACCESS_KEY_ID = env_required("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_required("AWS_SECRET_ACCESS_KEY")

S3_UPLOADS_BUCKET = env_required("S3_UPLOADS_BUCKET")
# Keep this required so prod matches your intended setup (even if not used here yet)
S3_CLIPS_BUCKET = env_required("S3_CLIPS_BUCKET")


# -----------------------------
# Clients
# -----------------------------
engine = sa.create_engine(DATABASE_URL, pool_pre_ping=True)
r = redis.Redis.from_url(REDIS_URL)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


# -----------------------------
# Models
# -----------------------------
class CreateUploadRequest(BaseModel):
    user_id: str
    original_filename: str
    content_type: Optional[str] = None


class CompleteUploadRequest(BaseModel):
    upload_id: str

class UpdateClipRequest(BaseModel):
    player_name: Optional[str] = None
    jersey_number: Optional[str] = None

# -----------------------------
# Routes
# -----------------------------
@app.get("/api/health")
def health():
    # DB check
    db_ok = False
    db_error = None
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_error = str(e)

    # Redis check
    redis_ok = False
    redis_error = None
    try:
        r.ping()
        redis_ok = True
    except Exception as e:
        redis_error = str(e)

    return {
        "status": "ok",
        "db_ok": db_ok,
        "db_error": db_error,
        "redis_ok": redis_ok,
        "redis_error": redis_error,
    }


@app.post("/api/uploads/create")
def create_upload(req: CreateUploadRequest):
    upload_id = uuid.uuid4()

    original_filename = (req.original_filename or "").strip() or "upload.bin"
    safe_name = original_filename.replace("\\", "_").replace("/", "_")

    content_type = (req.content_type or "").strip() or "application/octet-stream"

    s3_key = f"uploads/{req.user_id}/{upload_id}/{safe_name}"

    # Insert DB row
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                INSERT INTO uploads (id, user_id, original_filename, content_type, s3_key, bucket, status)
                VALUES (:id, :user_id, :original_filename, :content_type, :s3_key, :bucket, 'created')
                """
            ),
            {
                "id": str(upload_id),
                "user_id": req.user_id,
                "original_filename": original_filename,
                "content_type": content_type,
                "s3_key": s3_key,
                "bucket": S3_UPLOADS_BUCKET,
            },
        )

    # Generate presigned PUT URL (15 minutes)
    presigned_url = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": S3_UPLOADS_BUCKET,
            "Key": s3_key,
        },
        ExpiresIn=900,
    )

    return {
        "upload_id": str(upload_id),
        "bucket": S3_UPLOADS_BUCKET,
        "s3_key": s3_key,
        "content_type": content_type,
        "presigned_url": presigned_url,
        "status": "created",
        "queued": True,
    }


@app.post("/api/uploads/complete")
def complete_upload(req: CompleteUploadRequest):
    # Mark uploaded
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE uploads SET status='uploaded' WHERE id = :id"),
            {"id": req.upload_id},
        )

    # Queue processing only AFTER upload is done
    r.lpush("clipflow:jobs", req.upload_id)

    return {"status": "ok", "upload_id": req.upload_id}


@app.get("/api/uploads/recent")
def recent_uploads(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                """
                SELECT id, user_id, original_filename, content_type, s3_key, bucket, status, created_at
                FROM uploads
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()

    return {"uploads": [dict(row) for row in rows]}

@app.get("/api/debug/uploads/{upload_id}/counts")
def debug_counts(upload_id: str):
    with engine.connect() as conn:
        upload = conn.execute(
            sa.text("SELECT id, status FROM uploads WHERE id = :id"),
            {"id": upload_id},
        ).mappings().first()

        clip_count = conn.execute(
            sa.text("SELECT COUNT(*) AS n FROM clips WHERE upload_id = :id"),
            {"id": upload_id},
        ).mappings().first()

    return {"upload": upload, "clip_count": int(clip_count["n"])}

@app.get("/api/uploads/{upload_id}/clips")
def clips_for_upload(upload_id: str):
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                """
                SELECT id, upload_id, bucket, s3_key, start_sec, end_sec, label,
                       player_name, jersey_number, is_hit, ai_confidence, ai_reason, created_at
                FROM clips
                WHERE upload_id = :upload_id
                ORDER BY created_at DESC
                """
            ),
            {"upload_id": upload_id},
        ).mappings().all()

    return {"clips": [dict(row) for row in rows]}


@app.get("/api/clips/{clip_id}/download")
def clip_download(clip_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT bucket, s3_key FROM clips WHERE id = :id"),
            {"id": clip_id},
        ).mappings().first()

    if not row:
        # Returning dict is fine for MVP; later we can use HTTPException(404)
        return {"error": "not_found"}

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": row["bucket"], "Key": row["s3_key"]},
        ExpiresIn=900,
    )
    return {"download_url": url}

@app.patch("/api/clips/{clip_id}")
def update_clip(clip_id: str, req: UpdateClipRequest):
    player_name = (req.player_name or "").strip() or None
    jersey_number = (req.jersey_number or "").strip() or None

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                UPDATE clips
                SET player_name = :player_name,
                    jersey_number = :jersey_number
                WHERE id = :id
                """
            ),
            {"id": clip_id, "player_name": player_name, "jersey_number": jersey_number},
        )

        row = conn.execute(
            sa.text(
                """
                SELECT id, upload_id, bucket, s3_key, start_sec, end_sec, label,
                       player_name, jersey_number, created_at
                FROM clips
                WHERE id = :id
                """
            ),
            {"id": clip_id},
        ).mappings().first()

    if not row:
        return {"error": "not_found"}

    return {"clip": dict(row)}