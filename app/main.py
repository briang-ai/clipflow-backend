import json
import os
import uuid
from typing import List, Literal, Optional

import boto3
import httpx
import redis
import sqlalchemy as sa
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="ClipFlow API")

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


def env_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


DATABASE_URL      = env_required("DATABASE_URL")
REDIS_URL         = env_required("REDIS_URL")
AWS_REGION        = env_required("AWS_REGION")
AWS_ACCESS_KEY_ID = env_required("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_required("AWS_SECRET_ACCESS_KEY")
S3_UPLOADS_BUCKET = env_required("S3_UPLOADS_BUCKET")
S3_CLIPS_BUCKET   = env_required("S3_CLIPS_BUCKET")
CLERK_SECRET_KEY  = env_required("CLERK_SECRET_KEY")
ADMIN_SECRET      = env_required("ADMIN_SECRET")
ANTHROPIC_API_KEY = env_required("ANTHROPIC_API_KEY")

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
    watermark: bool = True
    mode: Literal["hits_only", "all_swings"] = "hits_only"

class BulkDeleteRequest(BaseModel):
    upload_ids: List[str]


# -----------------------------
# Admin helpers
# -----------------------------
async def _assert_clerk_admin(user_id: str | None):
    if not user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"https://api.clerk.com/v1/users/{user_id}",
            headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
        )
    if res.status_code != 200:
        raise HTTPException(status_code=403, detail="Forbidden")
    if res.json().get("private_metadata", {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")


async def _fetch_clerk_users(user_ids: list[str]) -> dict[str, dict]:
    results = {}
    async with httpx.AsyncClient() as client:
        for uid in user_ids:
            try:
                res = await client.get(
                    f"https://api.clerk.com/v1/users/{uid}",
                    headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
                )
                if res.status_code != 200:
                    continue
                data = res.json()
                primary_email = next(
                    (e["email_address"] for e in data.get("email_addresses", [])
                     if e["id"] == data.get("primary_email_address_id")), "",
                )
                first = data.get("first_name") or ""
                last  = data.get("last_name")  or ""
                results[uid] = {
                    "email": primary_email,
                    "name":  f"{first} {last}".strip() or primary_email,
                }
            except Exception:
                continue
    return results


# -----------------------------
# Routes — health
# -----------------------------
@app.get("/api/health")
def health():
    db_ok = False; db_error = None
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_error = str(e)

    redis_ok = False; redis_error = None
    try:
        r.ping(); redis_ok = True
    except Exception as e:
        redis_error = str(e)

    return {"status": "ok", "db_ok": db_ok, "db_error": db_error,
            "redis_ok": redis_ok, "redis_error": redis_error}


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
        row = conn.execute(sa.text("""
            SELECT COUNT(*) AS n FROM uploads
            WHERE user_id = :user_id AND created_at >= NOW() - INTERVAL '1 day'
        """), {"user_id": req.user_id}).mappings().first()

    if int(row["n"] or 0) >= 20:
        return {"error": "upload_limit_reached", "message": "Daily upload limit reached."}

    with engine.begin() as conn:
        conn.execute(sa.text("""
            INSERT INTO uploads (id, user_id, original_filename, content_type, s3_key, bucket, status)
            VALUES (:id, :user_id, :original_filename, :content_type, :s3_key, :bucket, 'created')
        """), {"id": str(upload_id), "user_id": req.user_id,
               "original_filename": original_filename, "content_type": content_type,
               "s3_key": s3_key, "bucket": S3_UPLOADS_BUCKET})

    presigned_url = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": S3_UPLOADS_BUCKET, "Key": s3_key},
        ExpiresIn=900,
    )
    return {"upload_id": str(upload_id), "bucket": S3_UPLOADS_BUCKET,
            "s3_key": s3_key, "content_type": content_type,
            "presigned_url": presigned_url, "status": "created", "queued": True}


@app.post("/api/uploads/complete")
def complete_upload(req: CompleteUploadRequest):
    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE uploads SET status='uploaded' WHERE id = :id"),
                     {"id": req.upload_id})
    r.lpush("clipflow:jobs", req.upload_id)
    return {"status": "ok", "upload_id": req.upload_id}


@app.get("/api/uploads/recent")
def recent_uploads(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT id, user_id, original_filename, content_type, s3_key, bucket, status, created_at
            FROM uploads ORDER BY created_at DESC LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"uploads": [dict(row) for row in rows]}


@app.get("/api/debug/uploads/{upload_id}/counts")
def debug_counts(upload_id: str):
    with engine.connect() as conn:
        upload = conn.execute(
            sa.text("SELECT id, status FROM uploads WHERE id = :id"),
            {"id": upload_id}).mappings().first()
        clip_count = conn.execute(
            sa.text("SELECT COUNT(*) AS n FROM clips WHERE upload_id = :id"),
            {"id": upload_id}).mappings().first()
    return {"upload": upload, "clip_count": int(clip_count["n"])}


@app.get("/api/uploads/{upload_id}/clips")
def clips_for_upload(upload_id: str):
    with engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT id, upload_id, bucket, s3_key, start_sec, end_sec, label,
                   player_name, jersey_number, is_hit, is_swing,
                   ai_confidence, ai_reason, created_at
            FROM clips
            WHERE upload_id = :upload_id
            ORDER BY start_sec ASC
        """), {"upload_id": upload_id}).mappings().all()
    return {"clips": [dict(row) for row in rows]}


@app.get("/api/uploads/{upload_id}/summary")
def upload_summary(upload_id: str):
    """Returns hit + swing counts for an upload — used by the uploads page."""
    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT
                COUNT(*) FILTER (WHERE is_hit = true)   AS hit_count,
                COUNT(*) FILTER (WHERE is_swing = true) AS swing_count,
                COUNT(*)                                AS total_clips
            FROM clips WHERE upload_id = :id
        """), {"id": upload_id}).mappings().first()
    return {
        "hit_count":   int(row["hit_count"]   or 0),
        "swing_count": int(row["swing_count"] or 0),
        "total_clips": int(row["total_clips"] or 0),
    }


@app.get("/api/uploads/{upload_id}/reels")
def reels_for_upload(upload_id: str):
    with engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT r.id, r.user_id, r.player_name, r.jersey_number,
                   r.game_date, r.clip_count, r.duration_sec,
                   r.status, r.error_message, r.s3_key, r.created_at
            FROM reels r WHERE r.upload_id = :upload_id
            ORDER BY r.created_at DESC
        """), {"upload_id": upload_id}).mappings().all()
    return {"reels": [dict(row) for row in rows]}


# -----------------------------------------------------------------------
# DELETE /api/uploads/bulk  — must be before /{upload_id}
# -----------------------------------------------------------------------
@app.delete("/api/uploads/bulk")
def bulk_delete_uploads(req: BulkDeleteRequest):
    if not req.upload_ids:
        return {"deleted": [], "errors": []}
    deleted, errors = [], []
    for upload_id in req.upload_ids:
        try:
            _delete_upload(upload_id); deleted.append(upload_id)
        except HTTPException as e:
            errors.append({"upload_id": upload_id, "error": e.detail})
        except Exception as e:
            errors.append({"upload_id": upload_id, "error": str(e)})
    return {"deleted": deleted, "errors": errors}


def _delete_upload(upload_id: str):
    with engine.connect() as conn:
        upload = conn.execute(
            sa.text("SELECT s3_key, bucket FROM uploads WHERE id = :id"),
            {"id": upload_id}).mappings().first()
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        clip_rows = conn.execute(
            sa.text("SELECT s3_key, bucket FROM clips WHERE upload_id = :id"),
            {"id": upload_id}).mappings().all()
        reel_rows = conn.execute(
            sa.text("SELECT s3_key FROM reels WHERE upload_id = :id"),
            {"id": upload_id}).mappings().all()

    def del_s3(bucket, key):
        try:
            if bucket and key: s3.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            print(f"[delete_upload] S3 fail {key}: {e}")

    del_s3(upload["bucket"], upload["s3_key"])
    for c in clip_rows: del_s3(c["bucket"], c["s3_key"])
    for re in reel_rows:
        if re["s3_key"]: del_s3(S3_CLIPS_BUCKET, re["s3_key"])

    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM clips  WHERE upload_id = :id"), {"id": upload_id})
        conn.execute(sa.text("DELETE FROM reels  WHERE upload_id = :id"), {"id": upload_id})
        conn.execute(sa.text("DELETE FROM uploads WHERE id = :id"),       {"id": upload_id})


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
            {"id": clip_id}).mappings().first()
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
        conn.execute(sa.text("""
            UPDATE clips SET player_name=:player_name, jersey_number=:jersey_number
            WHERE id=:id
        """), {"id": clip_id, "player_name": player_name, "jersey_number": jersey_number})
        row = conn.execute(sa.text("""
            SELECT id, upload_id, bucket, s3_key, start_sec, end_sec, label,
                   player_name, jersey_number, created_at
            FROM clips WHERE id=:id
        """), {"id": clip_id}).mappings().first()
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
        # For all_swings mode, expand clip_ids to include swing clips from same uploads
        if req.mode == "all_swings":
            # Get upload_ids from the provided clip_ids
            upload_ids_row = conn.execute(sa.text("""
                SELECT DISTINCT upload_id FROM clips
                WHERE id = ANY(CAST(:ids AS uuid[]))
            """), {"ids": req.clip_ids}).mappings().all()
            upload_ids = [r["upload_id"] for r in upload_ids_row]

            # Fetch all swing clips (hits + misses) from those uploads
            swing_rows = conn.execute(sa.text("""
                SELECT id FROM clips
                WHERE upload_id = ANY(CAST(:uids AS uuid[]))
                  AND (is_hit = true OR is_swing = true)
                ORDER BY start_sec ASC
            """), {"uids": upload_ids}).mappings().all()
            clip_ids_to_use = [r["id"] for r in swing_rows] or req.clip_ids
        else:
            clip_ids_to_use = req.clip_ids

        player_row = conn.execute(sa.text("""
            SELECT player_name, jersey_number FROM clips
            WHERE id = ANY(CAST(:ids AS uuid[]))
              AND player_name IS NOT NULL
            GROUP BY player_name, jersey_number
            ORDER BY COUNT(*) DESC LIMIT 1
        """), {"ids": clip_ids_to_use}).mappings().first()

        upload_row = conn.execute(
            sa.text("SELECT user_id, created_at FROM uploads WHERE id = :id"),
            {"id": req.upload_id}).mappings().first()

    if not upload_row:
        return {"error": "upload_not_found"}

    player_name   = player_row["player_name"]   if player_row else "Unknown"
    jersey_number = player_row["jersey_number"] if player_row else None
    game_date     = upload_row["created_at"].date()
    user_id       = upload_row["user_id"]
    reel_id       = str(uuid.uuid4())

    with engine.begin() as conn:
        conn.execute(sa.text("""
            INSERT INTO reels (id, user_id, upload_id, player_name, jersey_number,
                               game_date, status, clip_count)
            VALUES (:id, :user_id, :upload_id, :player_name, :jersey_number,
                    :game_date, 'pending', :clip_count)
        """), {"id": reel_id, "user_id": user_id, "upload_id": req.upload_id,
               "player_name": player_name, "jersey_number": jersey_number,
               "game_date": game_date, "clip_count": len(clip_ids_to_use)})

    job_payload = json.dumps({
        "type":          "compile_reel",
        "reel_id":       reel_id,
        "user_id":       user_id,
        "player_name":   player_name,
        "jersey_number": jersey_number or "",
        "game_date":     game_date.isoformat(),
        "clip_ids":      clip_ids_to_use,
        "watermark":     req.watermark,
    })
    r.lpush("clipflow:jobs", job_payload)
    return {"status": "queued", "reel_id": reel_id}


@app.delete("/api/reels/{reel_id}")
def delete_reel(reel_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT s3_key FROM reels WHERE id = :id"),
            {"id": reel_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Reel not found")
    if row["s3_key"]:
        try:
            s3.delete_object(Bucket=S3_CLIPS_BUCKET, Key=row["s3_key"])
        except Exception as e:
            print(f"[delete_reel] S3 fail: {e}")
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM reels WHERE id = :id"), {"id": reel_id})
    return {"deleted": True, "reel_id": reel_id}


@app.get("/api/reels/{reel_id}/public")
def reel_public(reel_id: str):
    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT id, player_name, jersey_number, game_date,
                   clip_count, duration_sec, status, s3_key
            FROM reels WHERE id = :id
        """), {"id": reel_id}).mappings().first()
    if not row or row["status"] != "complete":
        return {"error": "not_found"}
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_CLIPS_BUCKET, "Key": row["s3_key"]},
        ExpiresIn=3600,
    )
    return {"player_name": row["player_name"], "jersey_number": row["jersey_number"],
            "game_date": str(row["game_date"]), "clip_count": row["clip_count"],
            "duration_sec": row["duration_sec"], "video_url": url}


@app.get("/api/reels/{reel_id}/download")
def reel_download(reel_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT status, s3_key FROM reels WHERE id = :id"),
            {"id": reel_id}).mappings().first()
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


# -----------------------------
# Routes — admin
# -----------------------------
@app.get("/api/admin/stats")
async def admin_stats(
    x_admin_secret: str = Header(default=None),
    x_clerk_user_id: str = Header(default=None),
):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    await _assert_clerk_admin(x_clerk_user_id)

    with engine.connect() as conn:
        dau = conn.execute(sa.text("""
            SELECT COUNT(DISTINCT user_id) AS n FROM uploads
            WHERE created_at >= NOW() - INTERVAL '1 day'
        """)).mappings().first()
        uploads_today = conn.execute(sa.text("""
            SELECT COUNT(*) AS n FROM uploads
            WHERE created_at >= NOW() - INTERVAL '1 day'
        """)).mappings().first()
        clips_today = conn.execute(sa.text("""
            SELECT COUNT(*) AS n FROM clips
            WHERE created_at >= NOW() - INTERVAL '1 day'
        """)).mappings().first()
        reels_today = conn.execute(sa.text("""
            SELECT COUNT(*) AS n FROM reels
            WHERE created_at >= NOW() - INTERVAL '1 day'
        """)).mappings().first()
        hit_stats = conn.execute(sa.text("""
            SELECT COUNT(*) FILTER (WHERE is_hit=true) AS hits, COUNT(*) AS total
            FROM clips WHERE is_hit IS NOT NULL
        """)).mappings().first()
        totals = conn.execute(sa.text("""
            SELECT (SELECT COUNT(*) FROM uploads) AS total_uploads,
                   (SELECT COUNT(*) FROM clips)   AS total_clips,
                   (SELECT COUNT(*) FROM reels)   AS total_reels,
                   (SELECT COUNT(DISTINCT user_id) FROM uploads) AS total_users
        """)).mappings().first()

    hit_rate = round(hit_stats["hits"] / hit_stats["total"] * 100, 1) if hit_stats["total"] else 0
    return {
        "dau": int(dau["n"]), "uploads_today": int(uploads_today["n"]),
        "clips_today": int(clips_today["n"]), "reels_today": int(reels_today["n"]),
        "hit_rate_pct": hit_rate, "total_uploads": int(totals["total_uploads"]),
        "total_clips": int(totals["total_clips"]), "total_reels": int(totals["total_reels"]),
        "total_users": int(totals["total_users"]),
    }


@app.get("/api/admin/users")
async def admin_users(
    x_admin_secret: str = Header(default=None),
    x_clerk_user_id: str = Header(default=None),
):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    await _assert_clerk_admin(x_clerk_user_id)

    with engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT user_id, COUNT(*) AS upload_count, MAX(created_at) AS last_upload_at
            FROM uploads GROUP BY user_id ORDER BY last_upload_at DESC
        """)).mappings().all()

    db_users = {r["user_id"]: dict(r) for r in rows}
    clerk_users = await _fetch_clerk_users(list(db_users.keys()))
    result = []
    for user_id, db in db_users.items():
        clerk = clerk_users.get(user_id, {})
        result.append({
            "user_id": user_id, "email": clerk.get("email", ""),
            "name": clerk.get("name", ""), "upload_count": int(db.get("upload_count", 0)),
            "last_upload_at": str(db.get("last_upload_at", "")),
        })
    return {"users": result}


@app.get("/api/admin/anthropic-balance")
async def admin_anthropic_balance(
    x_admin_secret: str = Header(default=None),
    x_clerk_user_id: str = Header(default=None),
):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    await _assert_clerk_admin(x_clerk_user_id)
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://api.anthropic.com/v1/organizations/balance",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
                timeout=10,
            )
        if res.status_code == 200:
            data = res.json()
            available = data.get("balance", {}).get("available", None)
            return {"ok": True, "raw": data,
                    "available_usd": round(available / 100, 2) if available is not None else None}
        return {"ok": False, "status_code": res.status_code, "error": res.text[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)}