"""Unit tests for the hit finder's pure logic (no video, no network).

Run from the backend folder:  python tests/test_hitfinder_rules.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clipflow_ai.classify import parse_response, sound_hint  # noqa: E402
from clipflow_ai.decide import final_hits  # noqa: E402

results = []


def check(name, cond):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def m(t, a, label="hit", conf=0.8):
    return {"t": t, "audio_z": a, "label": label, "confidence": conf}


# rule 1: one play, one hit
kept = final_hits([m(6.39, 68), m(7.13, 50)])
check("follow-through 0.7 s after a hit is merged into it", [k["t"] for k in kept] == [6.39])
kept = final_hits([m(3.06, 1346), m(14.23, 1096)])
check("two hits 11 s apart are both kept", len(kept) == 2)

# rule 2: the real hit is the loud one
kept = final_hits([m(11.64, 672), m(20.95, 156), m(30.06, 445)])
check("quiet 'hit' is dropped when a 4x louder hit exists", [k["t"] for k in kept] == [11.64, 30.06])
kept = final_hits([m(1.5, 138), m(8.3, 152), m(13.5, 100)])
check("batting-cage hits of similar loudness are all kept", len(kept) == 3)
kept = final_hits([m(6.39, 42)])
check("a lone quiet game hit is kept", len(kept) == 1)
kept = final_hits([m(5.0, 0, "miss"), m(9.0, 30, "no_swing")])
check("misses and non-swings never become hits", kept == [])

# rule 3: blocked view -> only hits the AI saw; the rest become possible hits
net = {"through_netting": True, "other_hitters_nearby": False}
rows = [{**m(1.5, 138), "contact_seen": True}, {**m(8.3, 40), "contact_seen": False}]
kept = final_hits(rows, net)
check("through netting, a hit the AI saw is kept", [k["t"] for k in kept] == [1.5])
check("through netting, an unseen 'hit' becomes a possible hit", rows[1].get("possible_hit") is True)
rows = [{**m(2.0, 400), "contact_seen": True}, {**m(9.0, 60), "contact_seen": True}]
check("with other hitters nearby, the loudness rule is off", len(final_hits(rows, {"other_hitters_nearby": True})) == 2)

rows = [{**m(2.0, 90), "reason": "loud crack, ball hidden by netting"}, {**m(9.0, 90), "contact_seen": True}]
kept = final_hits(rows)
check("a clip counts as blocked when the AI's own reason mentions netting", [k["t"] for k in kept] == [9.0] and rows[0].get("possible_hit"))

# rule 4: unsure means ask
rows = [m(3.9, 42, conf=0.6)]
check("a 60%-sure hit the AI didn't see becomes a possible hit", final_hits(rows) == [] and rows[0].get("possible_hit"))
check("a 60%-sure hit the AI did see is kept", len(final_hits([{**m(3.9, 42, conf=0.6), "contact_seen": True}])) == 1)

# reply parsing tolerates the shapes models actually return
tool = lambda moments: {"content": [{"type": "tool_use", "name": "report_moments", "input": {"moments": moments}}]}
row = {"id": "M1", "label": "hit", "confidence": 0.9, "reason": "ball leaves bat"}
check("normal reply parses", parse_response(tool([row]))["M1"]["label"] == "hit")
check("list sent as a JSON string parses", parse_response(tool('[{"id":"M1","label":"miss","confidence":0.6,"reason":"x"}]'))["M1"]["label"] == "miss")
check("unknown labels are ignored", parse_response(tool([{**row, "label": "homerun"}])) == {})
check("empty / cut-off reply gives no labels", parse_response({"content": [{"type": "tool_use", "name": "report_moments", "input": {}}]}) == {})

check("sound hint names the loudest crack", "loudest sharp sound" in sound_hint(672, 1, 672))
check("sound hint gives relative loudness", "23%" in sound_hint(156, 3, 672))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
