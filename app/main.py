import json
import os
import uuid
from typing import List, Optional

import boto3
import redis
import sqlalchemy as sa
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

"ANTHROPIC_KEY?", bool(os.getenv("ANTHROPIC_API_KEY")),

# Load .env for local dev only; Render uses its own Environment settings
load_dotenv()

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="ClipFlow API")

# -----------------------------
# CORS
# -----------------------------
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
REDIS_URL    = env_required("REDIS_URL")

AWS_REGION            = env_required("AWS_REGION")
AWS_ACCESS_KEY_ID     = env_required("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_required("AWS_SECRET_ACCESS_KEY")

S3_UPLOADS_BUCKET = env_required("S3_UPLOADS_BUCKET")
S3_CLIPS_BUCKET   = env_required("S3_CLIPS_BUCKET")


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

class CompileReelRequest(BaseModel):
    upload_id: str
    clip_ids: List[str]

class BulkDeleteRequest(BaseModel):
    upload_ids: List[str]


# -----------------------------
# Routes — health
# -----------------------------
@app.get("/api/health")
def health():
    db_ok = False
    db_error = None
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_error = str(e)

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


# -----------------------------
# Routes — uploads
# -----------------------------
@app.post("/api/uploads/create")
def create_upload(req: CreateUploadRequest):
    upload_id = uuid.uuid4()

    original_filename = (req.original_filename or "").strip() or "upload.bin"
    safe_name = original_filename.replace("\\", "_").replace("/", "_")
    content_type = (req.content_type or "").strip() or "application/octet-stream"
    s3_key = f"uploads/{req.user_id}/{upload_id}/{safe_name}"

    with engine.connect() as conn:
        row = conn.execute(
            sa.text("""
                SELECT COUNT(*) AS n
                FROM uploads
                WHERE user_id = :user_id
                  AND created_at >= NOW() - INTERVAL '1 day'
            """),
            {"user_id": req.user_id},
        ).mappings().first()

    if int(row["n"] or 0) >= 20:
        return {
            "error": "upload_limit_reached",
            "message": "Daily upload limit reached. Please try again tomorrow.",
        }

    with engine.begin() as conn:
        conn.execute(
            sa.text("""
                INSERT INTO uploads (id, user_id, original_filename, content_type, s3_key, bucket, status)
                VALUES (:id, :user_id, :original_filename, :content_type, :s3_key, :bucket, 'created')
            """),
            {
                "id": str(upload_id),
                "user_id": req.user_id,
                "original_filename": original_filename,
                "content_type": content_type,
                "s3_key": s3_key,
                "bucket": S3_UPLOADS_BUCKET,
            },
        )

    presigned_url = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": S3_UPLOADS_BUCKET, "Key": s3_key},
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
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE uploads SET status='uploaded' WHERE id = :id"),
            {"id": req.upload_id},
        )
    r.lpush("clipflow:jobs", req.upload_id)
    return {"status": "ok", "upload_id": req.upload_id}


@app.get("/api/uploads/recent")
def recent_uploads(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("""
                SELECT id, user_id, original_filename, content_type, s3_key, bucket, status, created_at
                FROM uploads
                ORDER BY created_at DESC
                LIMIT :limit
            """),
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
            sa.text("""
                SELECT id, upload_id, bucket, s3_key, start_sec, end_sec, label,
                       player_name, jersey_number, is_hit, ai_confidence, ai_reason, created_at
                FROM clips
                WHERE upload_id = :upload_id
                ORDER BY created_at DESC
            """),
            {"upload_id": upload_id},
        ).mappings().all()
    return {"clips": [dict(row) for row in rows]}


@app.get("/api/uploads/{upload_id}/reels")
def reels_for_upload(upload_id: str):
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("""
                SELECT r.id, r.user_id, r.player_name, r.jersey_number,
                       r.game_date, r.clip_count, r.duration_sec,
                       r.status, r.error_message, r.s3_key, r.created_at
                FROM   reels r
                WHERE  r.upload_id = :upload_id
                ORDER  BY r.created_at DESC
            """),
            {"upload_id": upload_id},
        ).mappings().all()
    return {"reels": [dict(row) for row in rows]}


# -----------------------------------------------------------------------
# DELETE /api/uploads/bulk
# Must be registered BEFORE /api/uploads/{upload_id} so FastAPI doesn't
# match "bulk" as an upload_id path param.
# Body: { "upload_ids": ["id1", "id2", ...] }
# -----------------------------------------------------------------------
@app.delete("/api/uploads/bulk")
def bulk_delete_uploads(req: BulkDeleteRequest):
    if not req.upload_ids:
        return {"deleted": [], "errors": []}

    deleted = []
    errors = []

    for upload_id in req.upload_ids:
        try:
            _delete_upload(upload_id)
            deleted.append(upload_id)
        except HTTPException as e:
            errors.append({"upload_id": upload_id, "error": e.detail})
        except Exception as e:
            errors.append({"upload_id": upload_id, "error": str(e)})

    return {"deleted": deleted, "errors": errors}


# -----------------------------------------------------------------------
# DELETE /api/uploads/{upload_id}
# Deletes the upload record, all associated clips, any associated reels,
# and the corresponding S3 objects.
# -----------------------------------------------------------------------
def _delete_upload(upload_id: str):
    """Core delete logic, callable internally and from the HTTP handler."""
    with engine.connect() as conn:
        upload = conn.execute(
            sa.text("SELECT s3_key, bucket FROM uploads WHERE id = :id"),
            {"id": upload_id},
        ).mappings().first()

        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")

        clip_rows = conn.execute(
            sa.text("SELECT s3_key, bucket FROM clips WHERE upload_id = :id"),
            {"id": upload_id},
        ).mappings().all()

        reel_rows = conn.execute(
            sa.text("SELECT s3_key FROM reels WHERE upload_id = :id"),
            {"id": upload_id},
        ).mappings().all()

    # ── Delete S3 objects (failures are logged but don't abort DB cleanup) ──
    def delete_s3_key(bucket: str, key: str):
        try:
            if bucket and key:
                s3.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            print(f"[delete_upload] S3 delete failed for {key}: {e}")

    delete_s3_key(upload["bucket"], upload["s3_key"])
    for clip in clip_rows:
        delete_s3_key(clip["bucket"], clip["s3_key"])
    for reel in reel_rows:
        if reel["s3_key"]:
            delete_s3_key(S3_CLIPS_BUCKET, reel["s3_key"])

    # ── Delete DB rows ───────────────────────────────────────────────────────
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM clips WHERE upload_id = :id"), {"id": upload_id})
        conn.execute(sa.text("DELETE FROM reels WHERE upload_id = :id"), {"id": upload_id})
        conn.execute(sa.text("DELETE FROM uploads WHERE id = :id"),      {"id": upload_id})


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str):
    _delete_upload(upload_id)
    return {"deleted": True, "upload_id": upload_id}


# -----------------------------
# Routes — clips
# -----------------------------
@app.get("/api/clips/{clip_id}/download")
def clip_download(clip_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT bucket, s3_key FROM clips WHERE id = :id"),
            {"id": clip_id},
        ).mappings().first()

    if not row:
        return {"error": "not_found"}

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": row["bucket"], "Key": row["s3_key"]},
        ExpiresIn=900,
    )
    return {"download_url": url}


@app.patch("/api/clips/{clip_id}")
def update_clip(clip_id: str, req: UpdateClipRequest):
    player_name   = (req.player_name   or "").strip() or None
    jersey_number = (req.jersey_number or "").strip() or None

    with engine.begin() as conn:
        conn.execute(
            sa.text("""
                UPDATE clips
                SET player_name   = :player_name,
                    jersey_number = :jersey_number
                WHERE id = :id
            """),
            {"id": clip_id, "player_name": player_name, "jersey_number": jersey_number},
        )
        row = conn.execute(
            sa.text("""
                SELECT id, upload_id, bucket, s3_key, start_sec, end_sec, label,
                       player_name, jersey_number, created_at
                FROM clips WHERE id = :id
            """),
            {"id": clip_id},
        ).mappings().first()

    if not row:
        return {"error": "not_found"}
    return {"clip": dict(row)}


# -----------------------------
# Routes — reels
# -----------------------------
@app.post("/api/reels/compile")
def compile_reel(req: CompileReelRequest):
    if not req.clip_ids:
        return {"error": "no_clips", "message": "clip_ids must not be empty"}

    with engine.connect() as conn:
        player_row = conn.execute(
            sa.text("""
                SELECT player_name, jersey_number
                FROM   clips
                WHERE  id = ANY(CAST(:ids AS uuid[]))
                  AND  player_name IS NOT NULL
                GROUP  BY player_name, jersey_number
                ORDER  BY COUNT(*) DESC
                LIMIT  1
            """),
            {"ids": req.clip_ids},
        ).mappings().first()

        upload_row = conn.execute(
            sa.text("SELECT user_id, created_at FROM uploads WHERE id = :id"),
            {"id": req.upload_id},
        ).mappings().first()

    if not upload_row:
        return {"error": "upload_not_found"}

    player_name   = player_row["player_name"]   if player_row else "Unknown"
    jersey_number = player_row["jersey_number"] if player_row else None
    game_date     = upload_row["created_at"].date()
    user_id       = upload_row["user_id"]
    reel_id       = str(uuid.uuid4())

    with engine.begin() as conn:
        conn.execute(
            sa.text("""
                INSERT INTO reels
                    (id, user_id, upload_id, player_name, jersey_number,
                     game_date, status, clip_count)
                VALUES
                    (:id, :user_id, :upload_id, :player_name, :jersey_number,
                     :game_date, 'pending', :clip_count)
            """),
            {
                "id":            reel_id,
                "user_id":       user_id,
                "upload_id":     req.upload_id,
                "player_name":   player_name,
                "jersey_number": jersey_number,
                "game_date":     game_date,
                "clip_count":    len(req.clip_ids),
            },
        )

    job_payload = json.dumps({
        "type":     "compile_reel",
        "reel_id":  reel_id,
        "clip_ids": req.clip_ids,
    })
    r.lpush("clipflow:jobs", job_payload)

    return {"status": "queued", "reel_id": reel_id}


@app.get("/api/reels/{reel_id}/download")
def reel_download(reel_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT status, s3_key FROM reels WHERE id = :id"),
            {"id": reel_id},
        ).mappings().first()

    if not row:
        return {"error": "not_found"}
    if row["status"] != "complete":
        return {"error": "not_ready", "status": row["status"]}

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_CLIPS_BUCKET, "Key": row["s3_key"]},
        ExpiresIn=900,
    )
    return {"download_url": url}