"""Stage 2: one AI call per clip that labels every candidate moment.

The request is plain JSON for the Anthropic Messages API, so it can be sent
with the official SDK (production) or plain HTTP (evaluation runner).
"""
from __future__ import annotations

import base64

LABELS = ("hit", "foul_or_tip", "miss", "no_swing")

SYSTEM_PROMPT = """You review youth baseball and softball video for a highlight-reel app.
A parent recorded a short clip (usually one at-bat or a few practice swings). A detector has
already found a few MOMENTS that might be a swing. For each moment you get one image: 8 frames
in time order (numbered 1-8, left to right, top row then bottom row). Frame 4 is closest to
the moment itself; frames 1-3 are just before, 5-6 just after, 7-8 are later
(about +0.5 s and +1 s) to show what happened next. The frames are cropped around the batter.

Label each moment with exactly one of:
- "hit": the batter swings and the bat clearly meets the ball, sending it away (line drive,
  grounder, fly ball, or a solid hit into a practice net). Signs: ball changes direction or
  leaves off the bat, ball not seen continuing past the batter, batter drops bat and runs.
- "foul_or_tip": the bat touches the ball but it goes straight back, straight down, or barely
  changes direction. Not highlight-worthy.
- "miss": a real swing at a pitch with no contact. Signs: ball continues past the bat into the
  catcher, net or dirt behind the plate; batter stays in the box.
- "no_swing": anything else - taking a pitch, practice or warm-up swings with no pitch, the
  bat tapping the plate or ground, adjusting stance, dead time, fielding or running.

How to decide:
- Trust what you SEE over the sound hint. Loud sharp sounds also come from a ball hitting a
  catcher's mitt, a bat tapping the plate, a pitching machine firing or a neighbouring cage.
  But a loud crack at the exact moment the bat crosses the hitting zone is strong support for
  contact, and a silent swing is usually a miss.
- Compare moments within the clip. A solid hit is usually the loudest crack in the clip. A
  swing whose crack is much quieter than another moment's is more often a foul tip, a ball
  into the dirt or a miss. Balls already lying on the ground near the plate are from earlier
  pitches: a ball that was there before the swing is not evidence of contact.
- One play is one moment: follow-through, dropping the bat and running after a hit belong to
  that hit, so label the later moments of the same play "no_swing".
- The ball is small and may be motion-blurred or partly hidden by netting. Look for it in
  frames 2-6 near the bat and again in 5-8 to see which way it went.
- Batting cages: most solid swings at machine pitches are hits; still check.
- If a moment is a swing but you cannot tell contact from miss, pick the more likely label
  and give a low confidence.

Also judge the SETTING of the whole clip:
- through_netting: the camera films through netting, a fence or a cage wall that hides the ball.
- other_hitters_nearby: other batters, cages or games are visible or likely audible (busy
  facility, neighbouring cages, balls from elsewhere).
In either case sounds are NOT evidence (they may come from another cage, a pitching machine or
the ball hitting the back net), so judge contact only from what you can see.

For every moment also set view_blocked: true if netting, a fence or anything else hides the
ball around the bat. And set contact_seen: true only if you can actually see the ball come off
the bat (the ball visibly moving away from the bat after contact). If netting, blur or the crop
hides that, set contact_seen false even when you think it was probably a hit.

Report through the report_moments tool, with moments as a JSON array of objects. Keep each
reason under 20 words and name the frames you relied on."""

TOOL = {
    "name": "report_moments",
    "description": "Report the label for every candidate moment in this clip.",
    "input_schema": {
        "type": "object",
        "properties": {
            "through_netting": {"type": "boolean"},
            "other_hitters_nearby": {"type": "boolean"},
            "moments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Moment id, e.g. M1"},
                        "label": {"type": "string", "enum": list(LABELS)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "contact_seen": {"type": "boolean"},
                        "view_blocked": {"type": "boolean",
                                         "description": "netting, a fence or an obstruction hides the ball"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "label", "confidence", "contact_seen", "reason"],
                },
            }
        },
        "required": ["through_netting", "other_hitters_nearby", "moments"],
    },
}


def sound_hint(audio_z: float, audio_rank: int, loudest_z: float = 0.0) -> str:
    if audio_z < 25:
        return "no sharp sound"
    level = "faint" if audio_z < 80 else "moderate" if audio_z < 250 else "very loud"
    if audio_rank == 1:
        rel = "the loudest sharp sound in the clip"
    elif loudest_z > 0:
        rel = f"about {round(100 * audio_z / loudest_z)}% as loud as the loudest crack in the clip"
    else:
        rel = ""
    return f"{level} sharp crack" + (f", {rel}" if rel else "")


def build_request(model: str, clip_seconds: float, moments: list[dict], loudest_z: float = 0.0,
                  max_tokens: int | None = None) -> dict:
    """moments: [{id, t, audio_z, audio_rank, jpeg: bytes}]"""
    if max_tokens is None:
        max_tokens = 400 + 160 * len(moments)
    content: list[dict] = [{
        "type": "text",
        "text": (f"Clip length {clip_seconds:.1f} s. {len(moments)} candidate moment(s) follow, "
                 "each as one 8-frame image.")}]
    for m in moments:
        content.append({"type": "text",
                        "text": f"{m['id']} at {m['t']:.2f} s - sound: {sound_hint(m['audio_z'], m['audio_rank'], loudest_z)}."})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(m["jpeg"]).decode()}})
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "report_moments"},
        "messages": [{"role": "user", "content": content}],
    }


def parse_response(resp: dict) -> dict[str, dict]:
    """-> {moment_id: {label, confidence, reason}} (missing ids simply absent)."""
    import json

    def as_obj(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except ValueError:
                return None
        return x

    out = {}
    for block in resp.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "report_moments":
            items = as_obj((block.get("input") or {}).get("moments", [])) or []
            for m in items if isinstance(items, list) else []:
                m = as_obj(m)
                if not isinstance(m, dict):
                    continue
                if m.get("label") in LABELS and m.get("id"):
                    out[m["id"]] = {"label": m["label"],
                                    "confidence": float(m.get("confidence", 0) or 0),
                                    "contact_seen": bool(m.get("contact_seen", False)),
                                    "view_blocked": bool(m.get("view_blocked", False)),
                                    "reason": str(m.get("reason", ""))[:300]}
    return out


def parse_setting(resp: dict) -> dict:
    """-> {"through_netting": bool, "other_hitters_nearby": bool} (False when not reported)."""
    for block in resp.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "report_moments":
            inp = block.get("input") or {}
            return {"through_netting": bool(inp.get("through_netting", False)),
                    "other_hitters_nearby": bool(inp.get("other_hitters_nearby", False))}
    return {"through_netting": False, "other_hitters_nearby": False}
