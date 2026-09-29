import json
import os
import uuid
from typing import List, Literal, Optional

import boto3
import httpx
import jwt
import redis
import sqlalchemy as sa
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Response
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
# Auth — verify the signed-in Clerk user on every request
# -----------------------------
# The browser sends "Authorization: Bearer <Clerk session token>".
# We check the token's signature against Clerk's public keys, so the
# user ID we get back can't be faked by editing a request.
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL", "https://api.clerk.com/v1/jwks")
_jwks_client = jwt.PyJWKClient(
    CLERK_JWKS_URL,
    headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
    cache_keys=True,
    lifespan=3600,
    timeout=10,
)


def verify_clerk_token(token: str) -> str:
    """Return the Clerk user ID for a valid session token, else raise 401."""
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"require": ["exp", "sub"], "verify_aud": False},
            leeway=5,
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Not signed in")
    # Clerk puts the page's origin in "azp"; only accept tokens minted for our site.
    azp = claims.get("azp")
    if azp and azp not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=401, detail="Not signed in")
    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user_id


def current_user(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not signed in")
    return verify_clerk_token(authorization.split(" ", 1)[1].strip())


def _valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


def _require_upload_owner(upload_id: str, user_id: str) -> dict:
    """404 unless the upload exists and belongs to this user."""
    if not _valid_uuid(upload_id):
        raise HTTPException(status_code=404, detail="Upload not found")
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT id, user_id, s3_key, bucket, status FROM uploads WHERE id = :id"),
            {"id": upload_id}).mappings().first()
    if not row or row["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Upload not found")
    return dict(row)


def _require_clip_owner(clip_id: str, user_id: str) -> dict:
    """404 unless the clip exists and its upload belongs to this user."""
    if not _valid_uuid(clip_id):
        raise HTTPException(status_code=404, detail="Clip not found")
    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT c.id, c.bucket, c.s3_key, c.thumbnail_s3_key, u.user_id
            FROM clips c JOIN uploads u ON u.id = c.upload_id
            WHERE c.id = :id
        """), {"id": clip_id}).mappings().first()
    if not row or row["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Clip not found")
    return dict(row)


def _require_reel_owner(reel_id: str, user_id: str) -> dict:
    if not _valid_uuid(reel_id):
        raise HTTPException(status_code=404, detail="Reel not found")
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT id, user_id, status, s3_key FROM reels WHERE id = :id"),
            {"id": reel_id}).mappings().first()
    if not row or row["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Reel not found")
    return dict(row)


# -----------------------------
# Models
# -----------------------------
class CreateUploadRequest(BaseModel):
    # user_id is ignored (kept so older app versions don't error);
    # the owner always comes from the verified login token.
    user_id: Optional[str] = None
    original_filename: str
    content_type: Optional[str] = None

class CompleteUploadRequest(BaseModel):
    upload_id: str

class UpdateClipRequest(BaseModel):
    player_name: Optional[str] = None
    jersey_number: Optional[str] = None
    is_hit: Optional[bool] = None   # family's own call; overrides the AI

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
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    redis_ok = False
    try:
        r.ping(); redis_ok = True
    except Exception:
        pass

    # Error details go to the server log only, not to the public.
    return {"status": "ok", "db_ok": db_ok, "redis_ok": redis_ok}


# -----------------------------
# Routes — uploads (signed-in owner only)
# -----------------------------
@app.post("/api/uploads/create")
def create_upload(req: CreateUploadRequest, user_id: str = Depends(current_user)):
    upload_id = uuid.uuid4()
    original_filename = (req.original_filename or "").strip() or "upload.bin"
    safe_name = original_filename.replace("\\", "_").replace("/", "_")
    content_type = (req.content_type or "").strip() or "application/octet-stream"
    s3_key = f"uploads/{user_id}/{upload_id}/{safe_name}"

    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT COUNT(*) AS n FROM uploads
            WHERE user_id = :user_id AND created_at >= NOW() - INTERVAL '1 day'
        """), {"user_id": user_id}).mappings().first()

    if int(row["n"] or 0) >= 20:
        return {"error": "upload_limit_reached", "message": "Daily upload limit reached."}

    with engine.begin() as conn:
        conn.execute(sa.text("""
            INSERT INTO uploads (id, user_id, original_filename, content_type, s3_key, bucket, status)
            VALUES (:id, :user_id, :original_filename, :content_type, :s3_key, :bucket, 'created')
        """), {"id": str(upload_id), "user_id": user_id,
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
def complete_upload(req: CompleteUploadRequest, user_id: str = Depends(current_user)):
    upload = _require_upload_owner(req.upload_id, user_id)
    # Only queue each upload once, so repeated calls can't run up AI costs.
    if upload["status"] != "created":
        return {"status": "ok", "upload_id": req.upload_id, "already_queued": True}
    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE uploads SET status='uploaded' WHERE id = :id"),
                     {"id": req.upload_id})
    r.lpush("clipflow:jobs", req.upload_id)
    return {"status": "ok", "upload_id": req.upload_id}


@app.get("/api/uploads/recent")
def recent_uploads(limit: int = 20, user_id: str = Depends(current_user)):
    limit = max(1, min(int(limit), 200))
    with engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT id, user_id, original_filename, content_type, s3_key, bucket, status, created_at
            FROM uploads WHERE user_id = :user_id
            ORDER BY created_at DESC LIMIT :limit
        """), {"user_id": user_id, "limit": limit}).mappings().all()
    return {"uploads": [dict(row) for row in rows]}


@app.get("/api/debug/uploads/{upload_id}/counts")
def debug_counts(upload_id: str, user_id: str = Depends(current_user)):
    upload = _require_upload_owner(upload_id, user_id)
    with engine.connect() as conn:
        clip_count = conn.execute(
            sa.text("SELECT COUNT(*) AS n FROM clips WHERE upload_id = :id"),
            {"id": upload_id}).mappings().first()
    return {"upload": {"id": upload["id"], "status": upload["status"]},
            "clip_count": int(clip_count["n"])}


@app.get("/api/uploads/{upload_id}/clips")
def clips_for_upload(upload_id: str, user_id: str = Depends(current_user)):
    _require_upload_owner(upload_id, user_id)
    with engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT id, upload_id, bucket, s3_key, thumbnail_s3_key,
                   start_sec, end_sec, label,
                   player_name, jersey_number, is_hit, is_swing,
                   ai_confidence, ai_reason, created_at
            FROM clips
            WHERE upload_id = :upload_id
            ORDER BY start_sec ASC
        """), {"upload_id": upload_id}).mappings().all()
    return {"clips": [dict(row) for row in rows]}


@app.get("/api/uploads/{upload_id}/summary")
def upload_summary(upload_id: str, user_id: str = Depends(current_user)):
    """Returns hit + swing counts for an upload — used by the uploads page."""
    _require_upload_owner(upload_id, user_id)
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


@app.get("/api/uploads/{upload_id}/thumbnail")
def upload_thumbnail(upload_id: str, user_id: str = Depends(current_user)):
    """Returns a short-lived presigned URL for the first clip thumbnail of an upload."""
    _require_upload_owner(upload_id, user_id)
    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT thumbnail_s3_key
            FROM clips
            WHERE upload_id = :upload_id
              AND thumbnail_s3_key IS NOT NULL
            ORDER BY start_sec ASC
            LIMIT 1
        """), {"upload_id": upload_id}).mappings().first()

    if not row or not row["thumbnail_s3_key"]:
        raise HTTPException(status_code=404, detail="No thumbnail available")

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_CLIPS_BUCKET, "Key": row["thumbnail_s3_key"]},
        ExpiresIn=3600,
    )
    return {"thumbnail_url": url}


@app.get("/api/uploads/{upload_id}/reels")
def reels_for_upload(upload_id: str, user_id: str = Depends(current_user)):
    _require_upload_owner(upload_id, user_id)
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
def bulk_delete_uploads(req: BulkDeleteRequest, user_id: str = Depends(current_user)):
    if not req.upload_ids:
        return {"deleted": [], "errors": []}
    deleted, errors = [], []
    for upload_id in req.upload_ids:
        try:
            _delete_upload(upload_id, user_id); deleted.append(upload_id)
        except HTTPException as e:
            errors.append({"upload_id": upload_id, "error": e.detail})
        except Exception as e:
            print(f"[bulk_delete] {upload_id}: {e}")
            errors.append({"upload_id": upload_id, "error": "delete_failed"})
    return {"deleted": deleted, "errors": errors}


def _delete_upload(upload_id: str, user_id: str):
    upload = _require_upload_owner(upload_id, user_id)
    with engine.connect() as conn:
        clip_rows = conn.execute(
            sa.text("SELECT s3_key, thumbnail_s3_key, bucket FROM clips WHERE upload_id = :id"),
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
    for c in clip_rows:
        del_s3(c["bucket"], c["s3_key"])
        del_s3(S3_CLIPS_BUCKET, c["thumbnail_s3_key"])
    for re in reel_rows:
        if re["s3_key"]: del_s3(S3_CLIPS_BUCKET, re["s3_key"])

    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM clips  WHERE upload_id = :id"), {"id": upload_id})
        conn.execute(sa.text("DELETE FROM reels  WHERE upload_id = :id"), {"id": upload_id})
        conn.execute(sa.text("DELETE FROM uploads WHERE id = :id"),       {"id": upload_id})


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str, user_id: str = Depends(current_user)):
    _delete_upload(upload_id, user_id)
    return {"deleted": True, "upload_id": upload_id}


# -----------------------------
# Routes — clips (signed-in owner only)
# -----------------------------
@app.get("/api/clips/{clip_id}/download")
def clip_download(clip_id: str, user_id: str = Depends(current_user)):
    clip = _require_clip_owner(clip_id, user_id)
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": clip["bucket"], "Key": clip["s3_key"]},
        ExpiresIn=900,
    )
    return {"download_url": url}


@app.get("/api/clips/{clip_id}/thumbnail")
def clip_thumbnail(clip_id: str, user_id: str = Depends(current_user)):
    """Returns a short-lived presigned URL for a single clip's thumbnail."""
    clip = _require_clip_owner(clip_id, user_id)
    if not clip["thumbnail_s3_key"]:
        raise HTTPException(status_code=404, detail="No thumbnail available")
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_CLIPS_BUCKET, "Key": clip["thumbnail_s3_key"]},
        ExpiresIn=3600,
    )
    return {"thumbnail_url": url}


@app.patch("/api/clips/{clip_id}")
def update_clip(clip_id: str, req: UpdateClipRequest, user_id: str = Depends(current_user)):
    _require_clip_owner(clip_id, user_id)
    sent = req.model_dump(exclude_unset=True)   # only change the fields the page sent
    sets, params = [], {"id": clip_id}
    for field in ("player_name", "jersey_number"):
        if field in sent:
            sets.append(f"{field} = :{field}")
            params[field] = (sent[field] or "").strip() or None
    if "is_hit" in sent:
        sets.append("is_hit = :is_hit")
        params["is_hit"] = sent["is_hit"]
        if sent["is_hit"]:
            sets.append("is_swing = true")
        sets.append("ai_reason = :why")
        params["why"] = "Marked by you as a hit" if sent["is_hit"] else "Marked by you as not a hit"
    with engine.begin() as conn:
        if sets:
            conn.execute(sa.text(f"UPDATE clips SET {', '.join(sets)} WHERE id = :id"), params)
        row = conn.execute(sa.text("""
            SELECT id, upload_id, bucket, s3_key, start_sec, end_sec, label,
                   player_name, jersey_number, is_hit, is_swing, ai_confidence, ai_reason, created_at
            FROM clips WHERE id=:id
        """), {"id": clip_id}).mappings().first()
    return {"clip": dict(row)}


# -----------------------------
# Routes — reels
# -----------------------------
@app.post("/api/reels/compile")
def compile_reel(req: CompileReelRequest, user_id: str = Depends(current_user)):
    if not req.clip_ids:
        return {"error": "no_clips", "message": "clip_ids must not be empty"}
    if not all(_valid_uuid(c) for c in req.clip_ids):
        raise HTTPException(status_code=404, detail="Clip not found")

    upload_row = _require_upload_owner(req.upload_id, user_id)

    with engine.connect() as conn:
        # Every requested clip must come from one of this user's uploads.
        owned_rows = conn.execute(sa.text("""
            SELECT c.id, c.upload_id FROM clips c
            JOIN uploads u ON u.id = c.upload_id
            WHERE c.id = ANY(CAST(:ids AS uuid[])) AND u.user_id = :user_id
        """), {"ids": req.clip_ids, "user_id": user_id}).mappings().all()
        if len(owned_rows) != len(set(req.clip_ids)):
            raise HTTPException(status_code=404, detail="Clip not found")

        # For all_swings mode, expand clip_ids to include swing clips from same uploads
        if req.mode == "all_swings":
            upload_ids = list({str(r["upload_id"]) for r in owned_rows})
            swing_rows = conn.execute(sa.text("""
                SELECT id FROM clips
                WHERE upload_id = ANY(CAST(:uids AS uuid[]))
                  AND (is_hit = true OR is_swing = true)
                ORDER BY start_sec ASC
            """), {"uids": upload_ids}).mappings().all()
            clip_ids_to_use = [str(r["id"]) for r in swing_rows] or req.clip_ids
        else:
            clip_ids_to_use = req.clip_ids

        player_row = conn.execute(sa.text("""
            SELECT player_name, jersey_number FROM clips
            WHERE id = ANY(CAST(:ids AS uuid[]))
              AND player_name IS NOT NULL
            GROUP BY player_name, jersey_number
            ORDER BY COUNT(*) DESC LIMIT 1
        """), {"ids": clip_ids_to_use}).mappings().first()

        created_row = conn.execute(
            sa.text("SELECT created_at FROM uploads WHERE id = :id"),
            {"id": req.upload_id}).mappings().first()

    player_name   = player_row["player_name"]   if player_row else "Unknown"
    jersey_number = player_row["jersey_number"] if player_row else None
    game_date     = created_row["created_at"].date()
    reel_id       = str(uuid.uuid4())

    with engine.begin() as conn:
        conn.execute(sa.text("""
            INSERT INTO reels (id, user_id, upload_id, player_name, jersey_number,
                               game_date, status, clip_count)
            VALUES (:id, :user_id, :upload_id, :player_name, :jersey_number,
                    :game_date, 'pending', :clip_count)
        """), {"id": reel_id, "user_id": user_id, "upload_id": upload_row["id"],
               "player_name": player_name, "jersey_number": jersey_number,
               "game_date": game_date, "clip_count": len(clip_ids_to_use)})

    job_payload = json.dumps({
        "type":          "compile_reel",
        "reel_id":       reel_id,
        "user_id":       user_id,
        "player_name":   player_name,
        "jersey_number": jersey_number or "",
        "game_date":     game_date.isoformat(),
        "clip_ids":      [str(c) for c in clip_ids_to_use],
        "watermark":     req.watermark,
    })
    r.lpush("clipflow:jobs", job_payload)
    return {"status": "queued", "reel_id": reel_id}


@app.delete("/api/reels/{reel_id}")
def delete_reel(reel_id: str, user_id: str = Depends(current_user)):
    row = _require_reel_owner(reel_id, user_id)
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
    """Public on purpose: this powers the share link a family sends out."""
    if not _valid_uuid(reel_id):
        return {"error": "not_found"}
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
def reel_download(reel_id: str, user_id: str = Depends(current_user)):
    row = _require_reel_owner(reel_id, user_id)
    if row["status"] != "complete":
        return {"error": "not_ready", "status": row["status"]}
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_CLIPS_BUCKET, "Key": row["s3_key"]},
        ExpiresIn=900,
    )
    return {"download_url": url}


# -----------------------------
# Routes — admin (verified login + Clerk "admin" role)
# -----------------------------
async def admin_user(user_id: str = Depends(current_user)) -> str:
    await _assert_clerk_admin(user_id)
    return user_id


@app.get("/api/admin/stats")
async def admin_stats(_admin: str = Depends(admin_user)):
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
async def admin_users(_admin: str = Depends(admin_user)):
    with engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT user_id, COUNT(*) AS upload_count, MAX(created_at) AS last_upload_at
            FROM uploads GROUP BY user_id ORDER BY last_upload_at DESC
        """)).mappings().all()

    db_users = {r["user_id"]: dict(r) for r in rows}
    clerk_users = await _fetch_clerk_users(list(db_users.keys()))
    result = []
    for uid, db in db_users.items():
        clerk = clerk_users.get(uid, {})
        result.append({
            "user_id": uid, "email": clerk.get("email", ""),
            "name": clerk.get("name", ""), "upload_count": int(db.get("upload_count", 0)),
            "last_upload_at": str(db.get("last_upload_at", "")),
        })
    return {"users": result}


@app.get("/api/admin/anthropic-balance")
async def admin_anthropic_balance(_admin: str = Depends(admin_user)):
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
