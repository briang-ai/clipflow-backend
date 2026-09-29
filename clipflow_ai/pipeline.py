"""Find the hits in one uploaded clip.

    stage 1  detect.find_candidates   free: sound + motion -> a few moments
    stage 2  classify                 one AI call per 8 moments -> hit / foul / miss / no swing
    stage 3  decide.final_hits        free: one play = one hit; the real hit is the loud one

`call_api(request_dict) -> response_dict` is injected so the same code runs in
the worker (Anthropic SDK) and in the offline evaluation (plain HTTP).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .classify import build_request, parse_response, parse_setting
from .compose import candidate_grid, encode_jpeg
from .decide import final_hits
from .detect import duration_s, find_candidates

CHUNK = 8

# $ per million tokens (input, output) for cost logging
PRICES = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
}


@dataclass
class Moment:
    t: float
    audio_z: float
    audio_rank: int
    motion_z: float
    box: tuple
    label: str = "no_swing"          # hit | foul_or_tip | miss | no_swing
    confidence: float = 0.0
    reason: str = ""
    contact_seen: bool = False
    view_blocked: bool = False
    final_hit: bool = False
    possible_hit: bool = False          # AI thinks hit but the view was blocked


@dataclass
class Analysis:
    duration: float
    moments: list[Moment] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    ai_calls: int = 0
    ai_errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    through_netting: bool = False
    other_hitters_nearby: bool = False

    @property
    def hits(self) -> list[Moment]:
        return [m for m in self.moments if m.final_hit]

    @property
    def swings(self) -> list[Moment]:
        return [m for m in self.moments if m.label in ("hit", "foul_or_tip", "miss")]


def analyze(path: str, call_api, model: str, log=print) -> Analysis:
    t0 = time.time()
    dur = duration_s(path)
    cands = find_candidates(path)
    res = Analysis(duration=dur)
    res.moments = [Moment(t=c.t, audio_z=c.audio_z, audio_rank=c.audio_rank,
                          motion_z=c.motion_z, box=tuple(c.box)) for c in cands]
    log(f"[clipflow_ai] {len(res.moments)} candidate moment(s) in {dur:.1f}s clip")
    if not res.moments:
        res.seconds = time.time() - t0
        return res

    loudest = max(m.audio_z for m in res.moments)
    price_in, price_out = PRICES.get(model, (2.0, 10.0))
    for k in range(0, len(res.moments), CHUNK):
        part = res.moments[k:k + CHUNK]
        payload = []
        for i, m in enumerate(part):
            jpg = encode_jpeg(candidate_grid(path, m.t, m.box, dur))
            payload.append({"id": f"M{i + 1}", "t": m.t, "audio_z": m.audio_z,
                            "audio_rank": m.audio_rank, "jpeg": jpg})
        req = build_request(model, dur, payload, loudest_z=loudest)
        try:
            resp = call_api(req)
        except Exception as e:  # network, credit balance, overload...
            res.ai_errors.append(str(e)[:300])
            log(f"[clipflow_ai] AI call failed: {e}")
            continue
        res.ai_calls += 1
        usage = resp.get("usage") or {}
        res.input_tokens += usage.get("input_tokens", 0) or 0
        res.output_tokens += usage.get("output_tokens", 0) or 0
        labels = parse_response(resp)
        setting = parse_setting(resp)
        res.through_netting |= setting["through_netting"]
        res.other_hitters_nearby |= setting["other_hitters_nearby"]
        if not labels:
            res.ai_errors.append(f"unparseable reply (stop_reason={resp.get('stop_reason')})")
        for i, m in enumerate(part):
            got = labels.get(f"M{i + 1}")
            if got:
                m.label, m.confidence, m.reason = got["label"], got["confidence"], got["reason"]
                m.contact_seen = got.get("contact_seen", False)
                m.view_blocked = got.get("view_blocked", False)

    res.cost_usd = (res.input_tokens * price_in + res.output_tokens * price_out) / 1e6
    rows = [{"t": m.t, "audio_z": m.audio_z, "label": m.label, "confidence": m.confidence,
             "contact_seen": m.contact_seen, "view_blocked": m.view_blocked, "reason": m.reason,
             "_m": m} for m in res.moments]
    kept = final_hits(rows, {"through_netting": res.through_netting,
                             "other_hitters_nearby": res.other_hitters_nearby})
    for d in kept:
        d["_m"].final_hit = True
    for d in rows:
        d["_m"].possible_hit = bool(d.get("possible_hit"))
    res.seconds = time.time() - t0
    log(f"[clipflow_ai] setting: netting={res.through_netting} other_hitters={res.other_hitters_nearby}; "
        f"{len(res.hits)} hit(s), {sum(m.possible_hit for m in res.moments)} possible, {len(res.swings)} swing(s); "
        f"{res.input_tokens} in / {res.output_tokens} out tokens = ${res.cost_usd:.4f}; "
        f"{res.seconds:.1f}s")
    return res
