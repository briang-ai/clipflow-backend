"""Stage 1 (free, no AI): find the few moments in a clip that might be a swing.

Two cheap signals, each normalised against the clip's own background level:
  * sound  - sharp high-frequency transients (bat crack, mitt pop, bat tapping plate)
  * motion - fast movement that is NOT explained by the phone shaking
             (camera shake is removed with phase correlation before differencing)

Only numpy + OpenCV + ffmpeg are needed.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict

import cv2
import numpy as np

AUDIO_SR = 22050
HOP_S = 0.005          # audio analysis step (5 ms)
MOTION_FPS = 15
MOTION_W = 192         # analysis width in pixels


# ----------------------------------------------------------------------------
# decoding helpers
# ----------------------------------------------------------------------------
def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def duration_s(path: str) -> float:
    d = probe(path)
    if d.get("format", {}).get("duration"):
        return float(d["format"]["duration"])
    for s in d.get("streams", []):
        if s.get("codec_type") == "video" and s.get("duration"):
            return float(s["duration"])
    raise RuntimeError("could not read duration")


def has_audio(path: str) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def load_audio(path: str, sr: int = AUDIO_SR) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def load_gray(path: str, fps: int = MOTION_FPS, width: int = MOTION_W):
    """Small grayscale frames at a fixed rate. ffmpeg applies phone rotation."""
    vf = f"fps={fps},scale={width}:-2,format=gray"
    head = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vf", vf,
                           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "pgm", "-"],
                          capture_output=True, check=True).stdout
    w, h = map(int, head.split(b"\n", 3)[1].split())
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vf", vf,
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
    times = np.arange(len(frames)) / fps
    return times, frames


# ----------------------------------------------------------------------------
# signals
# ----------------------------------------------------------------------------
def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def audio_onsets(x: np.ndarray, sr: int = AUDIO_SR):
    """Positive spectral flux in 1.5-8 kHz, as a robust z-score per 5 ms step."""
    if len(x) < sr // 4:
        return np.zeros(0), np.zeros(0)
    x = x / (np.abs(x).max() + 1e-9)
    n, hop = 512, int(sr * HOP_S)
    win = np.hanning(n).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) / win.sum()
    f = np.fft.rfftfreq(n, 1 / sr)
    band = (f >= 1500) & (f <= 8000)
    mag = np.log1p(40 * spec[:, band])
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    t = (np.arange(len(flux)) * hop + n / 2) / sr
    return t, _robust_z(flux)


def motion_signal(frames: np.ndarray):
    """Per-frame 'local fast motion' after removing camera shake.

    Returns (ratio, maps): ratio is the motion score divided by the clip's
    median (1.0 = typical fidgeting; swings measured 1.6-4x on test clips), and
    maps[i] is the stabilised difference image between frame i and i+1 (used
    later to find where the action is).
    """
    if len(frames) < 3:
        return np.zeros(max(0, len(frames) - 1)), np.zeros((0,) + frames.shape[1:], np.float32)
    f = frames.astype(np.float32)
    h, w = f.shape[1:]
    han = cv2.createHanningWindow((w, h), cv2.CV_32F)
    score = np.zeros(len(f) - 1, np.float32)
    maps = np.zeros((len(f) - 1, h, w), np.float32)
    for i in range(len(f) - 1):
        a, b = f[i], f[i + 1]
        (dx, dy), _ = cv2.phaseCorrelate(a, b, han)
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        b_al = cv2.warpAffine(b, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        d = cv2.absdiff(a, b_al)
        d = cv2.GaussianBlur(d, (5, 5), 0)          # suppress thin netting edges
        mg = 6
        d[:mg] = d[-mg:] = 0
        d[:, :mg] = d[:, -mg:] = 0                  # warp borders
        maps[i] = d
        # mean of the top 1% of pixels: big for a compact fast mover (swing),
        # small for residual shake spread thinly over the frame
        k = max(1, d.size // 100)
        score[i] = np.partition(d.ravel(), -k)[-k:].mean()
    return score / (np.median(score) + 1e-6), maps


def _peaks(t: np.ndarray, z: np.ndarray, min_z: float, min_gap: float):
    """Local maxima above min_z, greedily keeping the biggest and dropping
    anything within min_gap seconds of a bigger one."""
    if len(z) == 0:
        return []
    idx = np.where((z >= min_z) & (z >= np.roll(z, 1)) & (z >= np.roll(z, -1)))[0]
    idx = sorted(idx, key=lambda i: -z[i])
    kept = []
    for i in idx:
        if all(abs(t[i] - t[j]) >= min_gap for j in kept):
            kept.append(i)
    return [(float(t[i]), float(z[i])) for i in kept]


# ----------------------------------------------------------------------------
# candidate finder
# ----------------------------------------------------------------------------
@dataclass
class Candidate:
    t: float              # best estimate of the contact / swing moment (s)
    audio_z: float        # loudest crack within +-0.25 s (0 if none)
    motion_z: float       # strongest local motion within +-0.35 s (x clip median)
    audio_rank: int       # 1 = loudest crack in the clip
    score: float
    box: tuple            # (x0, y0, x1, y1) as fractions of the frame: where the action is


def _action_box(maps: np.ndarray, mt: np.ndarray, t: float, aspect: float):
    """Region covering the stabilised motion around time t, padded (fractions of frame).

    Covers the batter's body AND the bat/ball path, not just the motion's centre,
    so a fast-moving bat can't pull the crop off the batter.
    """
    h, w = maps.shape[1:]
    sel = (mt >= t - 0.4) & (mt <= t + 0.4)
    if not sel.any():
        return (0.0, 0.0, 1.0, 1.0)
    acc = cv2.GaussianBlur(maps[sel].sum(axis=0), (0, 0), sigmaX=w / 50)
    ys, xs = np.where(acc >= np.percentile(acc, 92))
    if len(xs) < 5:
        return (0.0, 0.0, 1.0, 1.0)
    x0, x1 = np.percentile(xs, 5) / w, np.percentile(xs, 95) / w
    y0, y1 = np.percentile(ys, 5) / h, np.percentile(ys, 95) / h
    # pad generously: incoming pitch and first metres of ball flight
    padx, pady = 0.22, 0.18
    min_w, min_h = (0.9, 0.5) if aspect < 1 else (0.62, 0.8)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw = min(1.0, max(min_w, (x1 - x0) + 2 * padx))
    bh = min(1.0, max(min_h, (y1 - y0) + 2 * pady))
    bx0 = min(max(cx - bw / 2, 0.0), 1 - bw)
    by0 = min(max(cy - bh / 2, 0.0), 1 - bh)
    return (round(bx0, 3), round(by0, 3), round(bx0 + bw, 3), round(by0 + bh, 3))


def find_candidates(path: str, max_candidates: int | None = None, debug: bool = False):
    dur = duration_s(path)
    if max_candidates is None:
        # a single at-bat has 1-4 swings; allow ~1 extra look per 5 s of footage
        max_candidates = int(min(80, 3 + dur // 5))
    mt, frames = load_gray(path)
    mz, maps = motion_signal(frames)
    mt = mt[1:]                                # diff i sits between frame i and i+1
    aspect = frames.shape[2] / frames.shape[1]

    if has_audio(path):
        at, az = audio_onsets(load_audio(path))
    else:
        at, az = np.zeros(0), np.zeros(0)

    audio_pk = _peaks(at, az, min_z=25, min_gap=0.35)
    motion_pk = _peaks(mt, mz, min_z=1.5, min_gap=0.6)
    loud_order = sorted(audio_pk, key=lambda p: -p[1])

    def near(peaks, t, win):
        vals = [z for (pt, z) in peaks if abs(pt - t) <= win]
        return max(vals) if vals else 0.0

    def motion_at(t, win=0.35):
        sel = (mt >= t - win) & (mt <= t + win)
        return float(mz[sel].max()) if sel.any() else 0.0

    # every motion peak is a possible swing; every loud sound is a possible contact
    seeds = [(t, "m") for t, _ in motion_pk] + [(t, "a") for t, _ in audio_pk]
    cands: list[Candidate] = []
    for t, kind in seeds:
        a = near(audio_pk, t, 0.25)
        m = motion_at(t)
        if kind == "m" and a > 0:
            # snap to the crack: sound gives the contact instant to within 5 ms
            t = max((p for p in audio_pk if abs(p[0] - t) <= 0.25), key=lambda p: p[1])[0]
        rank = next((i + 1 for i, p in enumerate(loud_order) if abs(p[0] - t) < 0.01), 0)
        # sound without motion is usually off-camera noise; motion without sound
        # can be a clean miss. Score favours both together.
        score = 4.0 * (min(m, 4.0) - 1.0) + 2.0 * np.log1p(a / 25.0)
        if m < 1.2:
            score -= 3.0
        cands.append(Candidate(t=round(t, 3), audio_z=round(a, 1), motion_z=round(m, 2),
                               audio_rank=rank, score=round(float(score), 2), box=(0, 0, 1, 1)))

    # merge seeds that describe the same moment
    cands.sort(key=lambda c: -c.score)
    merged: list[Candidate] = []
    for c in cands:
        if all(abs(c.t - k.t) > 0.7 for k in merged):
            merged.append(c)
    # people upload clips because something happened: always keep the best moment
    merged = [c for i, c in enumerate(merged) if c.score > 1.5 or i == 0][:max_candidates]
    for c in merged:
        c.t = float(min(max(c.t, 0.0), dur))
        c.box = _action_box(maps, mt, c.t, aspect)
    merged.sort(key=lambda c: c.t)
    if debug:
        return merged, dict(at=at, az=az, mt=mt, mz=mz, audio_pk=audio_pk, motion_pk=motion_pk)
    return merged


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        cs = find_candidates(p)
        print(p.rsplit("/", 1)[-1], [asdict(c) for c in cs])
