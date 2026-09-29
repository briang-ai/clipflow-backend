"""Build the single image the AI looks at for one candidate moment.

8 frames around the moment, cropped to where the action is, laid out 4 x 2
with a frame number in the corner. About 1,000 image tokens per candidate.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import cv2
import numpy as np

# seconds relative to the candidate moment. Dense around contact, then two
# later frames that show where the ball went / whether the batter ran.
OFFSETS = (-0.30, -0.15, -0.05, 0.03, 0.12, 0.25, 0.55, 1.10)
TILE_LONG = 320


def grab_frame(path: str, t: float, box, tile_long: int = TILE_LONG):
    """One frame at time t, cropped to box (fractions), scaled to tile_long."""
    x0, y0, x1, y1 = box
    crop = f"crop=iw*{x1 - x0:.4f}:ih*{y1 - y0:.4f}:iw*{x0:.4f}:ih*{y0:.4f}"
    scale = f"scale='if(gt(iw,ih),{tile_long},-2)':'if(gt(iw,ih),-2,{tile_long})'"
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "f.png")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", path,
                        "-frames:v", "1", "-vf", f"{crop},{scale}", out], check=True)
        return cv2.imread(out)


def candidate_grid(path: str, t: float, box, duration: float, offsets=OFFSETS,
                   tile_long: int = TILE_LONG, cols: int = 4) -> np.ndarray:
    tiles = []
    for i, off in enumerate(offsets):
        ft = min(max(t + off, 0.0), max(duration - 0.05, 0.0))
        im = grab_frame(path, ft, box, tile_long)
        if im is None:
            im = np.zeros((tile_long, tile_long, 3), np.uint8)
        cv2.rectangle(im, (0, 0), (26, 22), (0, 0, 0), -1)
        cv2.putText(im, str(i + 1), (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(im)
    h = min(x.shape[0] for x in tiles)
    w = min(x.shape[1] for x in tiles)
    tiles = [cv2.resize(x, (w, h)) for x in tiles]
    rows = [np.hstack([np.pad(x, ((2, 2), (2, 2), (0, 0))) for x in tiles[r:r + cols]])
            for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def encode_jpeg(img: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def image_tokens(img: np.ndarray) -> int:
    """Claude's image token estimate (standard tier)."""
    h, w = img.shape[:2]
    scale = min(1.0, 1568 / max(h, w), (1.15e6 / (h * w)) ** 0.5)
    w2, h2 = int(w * scale), int(h * scale)
    return int(np.ceil(w2 / 28) * np.ceil(h2 / 28))
