"""Stage 3 (free): turn per-moment AI labels into the clip's final hits.

Two physical rules, applied after the AI:
  1. One play, one hit. A hit is followed by more motion and noise (follow-
     through, bat drop, running), so "hits" less than 1.5 s apart are the
     same play - keep the one with the louder crack.
  2. The real hit is the loud one. If the clip already has a confirmed hit
     whose crack is at least 2.5x louder, a much quieter "hit" is almost always
     a foul tip, a ball into the dirt, or a bat tapping the plate.
  3. Blocked view, trust only eyes. Through netting, or with other hitters
     nearby, sound proves nothing (machines, other cages, the back net). There a
     hit counts only if the AI actually saw the ball leave the bat; otherwise it
     becomes a "possible hit" for the family to confirm. Rule 2 is off there,
     since the loudest crack may belong to someone else. A clip counts as
     blocked if the AI flags it OR its own per-moment answers say netting or
     an obstruction hid the ball (the clip-level flag alone proved unreliable).
  4. Unsure means ask. A "hit" the AI gives under 70% confidence without
     seeing contact becomes a possible hit too.

Measured on 19 test clips (32 real hits, labelled by hand): clear view 12/13
hits found automatically with 0 false hits (2 sent to check); through netting
0 false hits, every swing sent to check (18 of 19 real hits among them).
"""
from __future__ import annotations

import re

SAME_PLAY_S = 1.5
LOUDER_RATIO = 2.5
UNSURE_BELOW = 0.70

_BLOCKED_WORDS = re.compile(r"\bnet(s|ting)?\b|obscur|hidden by|hides? the ball|behind the net", re.I)


def view_blocked(setting: dict | None, moments: list[dict]) -> bool:
    """Blocked if the AI flagged the clip, flagged any moment, or said so in a reason."""
    setting = setting or {}
    if setting.get("through_netting") or setting.get("other_hitters_nearby"):
        return True
    return any(m.get("view_blocked") or _BLOCKED_WORDS.search(m.get("reason", "") or "")
               for m in moments)


def final_hits(moments: list[dict], setting: dict | None = None) -> list[dict]:
    """moments: [{t, audio_z, label, confidence, contact_seen?}] -> the moments kept as hits.

    Moments that were labelled "hit" but not kept get m["possible_hit"] = True when the
    only reason is a blocked view (rule 3).
    """
    blocked = view_blocked(setting, moments)
    hits = sorted((m for m in moments if m["label"] == "hit"), key=lambda m: m["t"])

    # rule 1: collapse same-play duplicates
    plays: list[dict] = []
    for m in hits:
        if plays and m["t"] - plays[-1]["t"] < SAME_PLAY_S:
            if (m["audio_z"], m["confidence"]) > (plays[-1]["audio_z"], plays[-1]["confidence"]):
                plays[-1] = m
        else:
            plays.append(m)

    if blocked:
        # rule 3: only hits the AI actually saw
        kept = []
        for m in plays:
            if m.get("contact_seen"):
                kept.append(m)
            else:
                m["possible_hit"] = True
        return kept

    # rule 2: drop quiet "hits" when a much louder hit exists in the same clip
    loudest = max((m["audio_z"] for m in plays), default=0.0)
    plays = [m for m in plays if not (loudest >= 50 and m["audio_z"] * LOUDER_RATIO <= loudest)]

    # rule 4: unsure means ask
    kept = []
    for m in plays:
        if m["confidence"] < UNSURE_BELOW and not m.get("contact_seen"):
            m["possible_hit"] = True
        else:
            kept.append(m)
    return kept
