"""Security tests for the ClipFlow API: login required + owner-only access.

Runs the real app/main.py against SQLite with fake Clerk keys.
"""
import json, os, sys, time, types, uuid

import jwt
import sqlalchemy as sa
from cryptography.hazmat.primitives.asymmetric import rsa

# ---- fake env + stub AWS/Redis (no network) ------------------------------
for k in ["DATABASE_URL", "REDIS_URL", "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
          "S3_UPLOADS_BUCKET", "S3_CLIPS_BUCKET", "CLERK_SECRET_KEY", "ANTHROPIC_API_KEY"]:
    os.environ[k] = "test"
os.environ["DATABASE_URL"] = "sqlite://"

class FakeS3:
    deleted = []
    def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):
        return f"https://s3.example/{Params['Key']}"
    def delete_object(self, Bucket, Key):
        FakeS3.deleted.append(Key)
sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: FakeS3())

class FakeRedis:
    pushed = []
    @classmethod
    def from_url(cls, url): return cls()
    def lpush(self, q, v): FakeRedis.pushed.append(v)
    def ping(self): return True
sys.modules["redis"] = types.SimpleNamespace(Redis=FakeRedis)

sys.path.insert(0, os.environ["BACKEND_DIR"])
from sqlalchemy.pool import StaticPool
_real_create_engine = sa.create_engine
sa.create_engine = lambda url, **kw: _real_create_engine(
    "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
import app.main as main
from fastapi.testclient import TestClient

# ---- fake Clerk signing keys ----------------------------------------------
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key()))
jwk.update({"kid": "k1", "use": "sig", "alg": "RS256"})
main._jwks_client.fetch_data = lambda: {"keys": [jwk]}

def token(sub, key=KEY, exp_in=60, azp="https://clipflow.pro", kid="k1"):
    now = int(time.time())
    claims = {"sub": sub, "iat": now, "nbf": now, "exp": now + exp_in, "sid": "s"}
    if azp: claims["azp"] = azp
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})

def H(t): return {"Authorization": f"Bearer {t}"}

# ---- data: two families --------------------------------------------------
with main.engine.begin() as c:
    c.execute(sa.text("CREATE TABLE uploads (id TEXT PRIMARY KEY, user_id TEXT, original_filename TEXT, content_type TEXT, s3_key TEXT, bucket TEXT, status TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"))
    c.execute(sa.text("CREATE TABLE clips (id TEXT PRIMARY KEY, upload_id TEXT, bucket TEXT, s3_key TEXT, thumbnail_s3_key TEXT, start_sec REAL, end_sec REAL, label TEXT, player_name TEXT, jersey_number TEXT, is_hit BOOLEAN, is_swing BOOLEAN, ai_confidence REAL, ai_reason TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"))
    c.execute(sa.text("CREATE TABLE reels (id TEXT PRIMARY KEY, user_id TEXT, upload_id TEXT, player_name TEXT, jersey_number TEXT, game_date TEXT, clip_count INT, duration_sec REAL, status TEXT, error_message TEXT, s3_key TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"))
    ids = {}
    for fam in ["alice", "bob"]:
        u, cl, rl = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        ids[fam] = dict(upload=u, clip=cl, reel=rl)
        c.execute(sa.text("INSERT INTO uploads (id,user_id,original_filename,s3_key,bucket,status) VALUES (:u,:f,'g.mp4',:k,'up','created')"), dict(u=u, f=f"user_{fam}", k=f"uploads/{fam}.mp4"))
        c.execute(sa.text("INSERT INTO clips (id,upload_id,bucket,s3_key,thumbnail_s3_key,start_sec,end_sec,label) VALUES (:c,:u,'clips',:k,:t,0,8,'s1')"), dict(c=cl, u=u, k=f"clips/{fam}.mp4", t=f"thumbs/{fam}.jpg"))
        c.execute(sa.text("INSERT INTO reels (id,user_id,upload_id,status,s3_key) VALUES (:r,:f,:u,'complete',:k)"), dict(r=rl, f=f"user_{fam}", u=u, k=f"reels/{fam}.mp4"))

client = TestClient(main.app)
A, B = ids["alice"], ids["bob"]
ALICE = H(token("user_alice"))
results = []

def check(name, cond):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)

# ---- 1. no / bad tokens are rejected --------------------------------------
protected = [
    ("get", "/api/uploads/recent"), ("get", f"/api/uploads/{A['upload']}/clips"),
    ("get", f"/api/uploads/{A['upload']}/summary"), ("get", f"/api/uploads/{A['upload']}/reels"),
    ("get", f"/api/uploads/{A['upload']}/thumbnail"), ("get", f"/api/debug/uploads/{A['upload']}/counts"),
    ("delete", f"/api/uploads/{A['upload']}"), ("get", f"/api/clips/{A['clip']}/download"),
    ("get", f"/api/clips/{A['clip']}/thumbnail"), ("get", f"/api/reels/{A['reel']}/download"),
    ("delete", f"/api/reels/{A['reel']}"), ("get", "/api/admin/stats"), ("get", "/api/admin/users"),
]
check("every private endpoint refuses a request with no login",
      all(getattr(client, m)(p).status_code == 401 for m, p in protected))
check("create upload refuses no login", client.post("/api/uploads/create", json={"original_filename": "x.mp4"}).status_code == 401)
check("forged token (wrong signing key) refused", client.get("/api/uploads/recent", headers=H(token("user_alice", key=OTHER_KEY))).status_code == 401)
check("expired token refused", client.get("/api/uploads/recent", headers=H(token("user_alice", exp_in=-120))).status_code == 401)
check("token minted for another website refused", client.get("/api/uploads/recent", headers=H(token("user_alice", azp="https://evil.example"))).status_code == 401)
check("unsigned 'none' token refused", client.get("/api/uploads/recent", headers=H(jwt.encode({"sub": "user_alice", "exp": int(time.time())+60}, None, algorithm="none"))).status_code == 401)
check("garbage token refused", client.get("/api/uploads/recent", headers=H("not-a-token")).status_code == 401)
check("old fake x-clerk-user-id header no longer grants anything", client.get("/api/uploads/recent", headers={"x-clerk-user-id": "user_alice"}).status_code == 401)

# ---- 2. owners see only their own things -----------------------------------
r = client.get("/api/uploads/recent", headers=ALICE)
check("Alice's upload list contains only Alice's uploads", r.status_code == 200 and [u["id"] for u in r.json()["uploads"]] == [A["upload"]])
check("Alice can read her own clips", client.get(f"/api/uploads/{A['upload']}/clips", headers=ALICE).status_code == 200)
check("Alice can download her own clip", client.get(f"/api/clips/{A['clip']}/download", headers=ALICE).json().get("download_url", "").endswith("clips/alice.mp4"))
check("Alice can download her own reel", client.get(f"/api/reels/{A['reel']}/download", headers=ALICE).status_code == 200)
for m, p in [("get", f"/api/uploads/{B['upload']}/clips"), ("get", f"/api/uploads/{B['upload']}/summary"),
             ("get", f"/api/uploads/{B['upload']}/reels"), ("get", f"/api/uploads/{B['upload']}/thumbnail"),
             ("get", f"/api/debug/uploads/{B['upload']}/counts"), ("get", f"/api/clips/{B['clip']}/download"),
             ("get", f"/api/clips/{B['clip']}/thumbnail"), ("get", f"/api/reels/{B['reel']}/download")]:
    check(f"Alice blocked from Bob's {p.split('/api/')[1].split('/')[0]} ({p.rsplit('/',1)[1]})", getattr(client, m)(p, headers=ALICE).status_code == 404)
r = client.patch(f"/api/clips/{A['clip']}", headers=ALICE, json={"player_name": "Sam", "jersey_number": "7"})
check("Alice can name her own clip", r.status_code == 200 and r.json()["clip"]["player_name"] == "Sam")
r = client.patch(f"/api/clips/{A['clip']}", headers=ALICE, json={"is_hit": True})
check("marking a hit keeps the player name", r.json()["clip"]["player_name"] == "Sam" and r.json()["clip"]["is_hit"] in (True, 1))
check("Alice can't mark Bob's clip as a hit", client.patch(f"/api/clips/{B['clip']}", headers=ALICE, json={"is_hit": True}).status_code == 404)
check("Alice can't rename Bob's clip", client.patch(f"/api/clips/{B['clip']}", headers=ALICE, json={"player_name": "x"}).status_code == 404)
check("Alice can't queue Bob's upload for AI", client.post("/api/uploads/complete", headers=ALICE, json={"upload_id": B["upload"]}).status_code == 404)
check("Alice can't build a reel from Bob's upload", client.post("/api/reels/compile", headers=ALICE, json={"upload_id": B["upload"], "clip_ids": [B["clip"]]}).status_code == 404)
check("Alice can't delete Bob's reel", client.delete(f"/api/reels/{B['reel']}", headers=ALICE).status_code == 404)
r = client.request("DELETE", "/api/uploads/bulk", headers=ALICE, json={"upload_ids": [B["upload"]]})
check("bulk delete of Bob's upload refused", r.json()["deleted"] == [])
check("Alice can't delete Bob's upload", client.delete(f"/api/uploads/{B['upload']}", headers=ALICE).status_code == 404)
with main.engine.connect() as c:
    check("Bob's upload, clip and reel are all still there",
          c.execute(sa.text("SELECT COUNT(*) FROM uploads WHERE id=:u"), {"u": B["upload"]}).scalar() == 1
          and c.execute(sa.text("SELECT COUNT(*) FROM clips WHERE upload_id=:u"), {"u": B["upload"]}).scalar() == 1
          and c.execute(sa.text("SELECT COUNT(*) FROM reels WHERE id=:r"), {"r": B["reel"]}).scalar() == 1)
check("nothing of Bob's was deleted from S3", not any("bob" in k for k in FakeS3.deleted))
check("junk IDs give 404, not a server error", client.get("/api/uploads/not-a-uuid/clips", headers=ALICE).status_code == 404)

# ---- 3. queueing is once-only (protects AI spend) ---------------------------
FakeRedis.pushed.clear()
client.post("/api/uploads/complete", headers=ALICE, json={"upload_id": A["upload"]})
client.post("/api/uploads/complete", headers=ALICE, json={"upload_id": A["upload"]})
check("an upload is queued for AI only once even if 'complete' is sent twice", FakeRedis.pushed == [A["upload"]])

# ---- 4. public share link still works without login -------------------------
r = client.get(f"/api/reels/{B['reel']}/public")
check("public share link works with no login", r.status_code == 200 and r.json()["video_url"].endswith("reels/bob.mp4"))
check("health check still public", client.get("/api/health").status_code == 200)

# ---- 5. admin needs a real login AND the admin role -------------------------
async def not_admin(uid):
    from fastapi import HTTPException
    if uid != "user_admin": raise HTTPException(status_code=403, detail="Forbidden")
main._assert_clerk_admin = not_admin
check("signed-in non-admin blocked from admin pages", client.get("/api/admin/users", headers=ALICE).status_code == 403)

# ---- 6. deleting your own upload works ------------------------------------
r = client.delete(f"/api/uploads/{A['upload']}", headers=ALICE)
check("Alice can delete her own upload (files + thumbnails removed)", r.status_code == 200 and {"uploads/alice.mp4", "clips/alice.mp4", "thumbs/alice.jpg", "reels/alice.mp4"} <= set(FakeS3.deleted))

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
