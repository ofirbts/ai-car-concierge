#!/usr/bin/env python3
"""
Parse conv_metrics log lines and report conversation quality KPIs.

Usage:
    python scripts/parse_conv_metrics.py [logfile]
    uvicorn ... 2>&1 | python scripts/parse_conv_metrics.py

If no logfile is provided, reads from stdin.
Example log line (emitted by sales_dialogue.py):
    INFO     conv_metrics session=abc turn=3 intent=slot_answer lang=he phase=discovery slots_filled=['passengers', 'use_case']
    INFO     conv_metrics session=abc turn=3 phase=recommending lang=he vehicles=[55, 77] filters={'budget_max': 75000}
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from typing import TextIO


_INTENT_RE = re.compile(r"conv_metrics session=(\S+) turn=(\d+) intent=(\S+) lang=(\S+) phase=(\S+)")
_REC_RE = re.compile(r"conv_metrics session=(\S+) turn=(\d+) phase=(\S+) lang=(\S+) vehicles=\[([^\]]*)\]")


def parse(stream: TextIO) -> None:
    sessions: dict[str, dict] = defaultdict(lambda: {
        "turns": 0,
        "phases": [],
        "intents": [],
        "langs": [],
        "rec_vehicle_sets": [],
        "repeat_guard_hits": 0,
    })

    total_turns = 0
    hebrew_turns = 0
    repeat_guard_hits = 0
    phase_advances = 0
    intent_counts: dict[str, int] = defaultdict(int)

    for line in stream:
        m = _INTENT_RE.search(line)
        if m:
            sid, turn, intent, lang, phase = m.groups()
            s = sessions[sid]
            s["turns"] += 1
            s["intents"].append(intent)
            s["langs"].append(lang)
            prev_phase = s["phases"][-1] if s["phases"] else None
            s["phases"].append(phase)
            total_turns += 1
            if lang == "he":
                hebrew_turns += 1
            if "repeat_guard" in line:
                repeat_guard_hits += 1
                s["repeat_guard_hits"] += 1
            if prev_phase and prev_phase != phase:
                phase_advances += 1
            intent_counts[intent] += 1
            continue

        m = _REC_RE.search(line)
        if m:
            sid, turn, phase, lang, vehicles_str = m.groups()
            vehicle_ids = tuple(sorted(v.strip() for v in vehicles_str.split(",") if v.strip()))
            s = sessions[sid]
            prev_set = s["rec_vehicle_sets"][-1] if s["rec_vehicle_sets"] else None
            if prev_set and prev_set == vehicle_ids:
                repeat_guard_hits += 1
                s["repeat_guard_hits"] += 1
            s["rec_vehicle_sets"].append(vehicle_ids)

    if total_turns == 0:
        print("No conv_metrics lines found in input.")
        return

    print("=" * 50)
    print("CONVERSATION QUALITY REPORT")
    print("=" * 50)
    print(f"Sessions:           {len(sessions)}")
    print(f"Total turns:        {total_turns}")
    print(f"Hebrew turns:       {hebrew_turns} ({100*hebrew_turns//total_turns}%)")
    print(f"Phase advances:     {phase_advances} ({100*phase_advances//total_turns}%)")
    print(f"Repeat guard hits:  {repeat_guard_hits}")
    print()
    print("Intent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count // total_turns
        print(f"  {intent:<30} {count:4d}  ({pct}%)")
    print()

    loop_sessions = [sid for sid, s in sessions.items() if s["repeat_guard_hits"] > 2]
    if loop_sessions:
        print(f"⚠️  Sessions with 3+ repeat guard hits (potential loops): {len(loop_sessions)}")
        for sid in loop_sessions[:10]:
            print(f"   session={sid}  hits={sessions[sid]['repeat_guard_hits']}")
    else:
        print("✅ No looping sessions detected.")

    avg_turns = total_turns / len(sessions) if sessions else 0
    print(f"\nAvg turns/session:  {avg_turns:.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            parse(f)
    else:
        parse(sys.stdin)
