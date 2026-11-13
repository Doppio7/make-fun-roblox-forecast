"""Predict Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Method: an empirical rank-transition kernel learned from the chart history.
For every pair of reads up to 72 hours apart, record how a game at a given rank
band moved (which place it held at the later read, 11-30, or off the chart).
Group pairs into lag buckets (0-6h, 6-12h, 12-24h, 24-48h, 48-72h); each bucket
is one estimate of where a game sits that many hours after the history ends.
The forecast for a game is the equal-weight average of its band's distributions
over the five buckets, which approximates the average over the 72 hourly slots
of the scored window. Per place the named games share 0.80 of the probability
mass and 0.20 is left unnamed (spread evenly by the scorer over the games not
named at that place), which hedges new entrants and games outside the top 20.

Games whose details show they were created more than 26 days before the window
start are scaled down: the chart only lists games published within the last 28
days, so they leave the chart part-way through the window.

Standard library only, deterministic, no network, no randomness.
"""

import json
import math
import os
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LAG_BUCKETS = [(0, 6), (6, 12), (12, 24), (24, 48), (48, 73)]
TOPN = 20
UNNAMED = 0.20
MIN_P = 0.005
SMOOTH = 0.5  # pseudocount per outcome cell


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def band(rank):
    if rank == 1:
        return 1
    if rank == 2:
        return 2
    if rank == 3:
        return 3
    if rank <= 5:
        return 4
    if rank <= 10:
        return 5
    if rank <= 15:
        return 6
    if rank <= 20:
        return 7
    if rank <= 30:
        return 8
    return 9


def build_kernel(reads):
    """counts[(rank_band, lag_bucket)][outcome] -> distribution over outcomes.

    Outcome 1..10 = that place, 11 = rank 11-30, 12 = off the chart.
    """
    counts = {}
    n = len(reads)
    for i in range(n):
        for j in range(i + 1, n):
            lag = (reads[j]["at"] - reads[i]["at"]).total_seconds() / 3600.0
            if lag <= 0 or lag > 72:
                continue
            bucket = None
            for bi, (lo, hi) in enumerate(LAG_BUCKETS):
                if lo < lag <= hi:
                    bucket = bi
                    break
            if bucket is None:
                continue
            ci, cj = reads[i]["chart"], reads[j]["chart"]
            for uid, rank in ci.items():
                if rank > 30:
                    continue
                out = cj.get(uid)
                if out is None:
                    out = 12
                elif out > 10:
                    out = 11
                key = (band(rank), bucket)
                cell = counts.setdefault(key, {})
                cell[out] = cell.get(out, 0) + 1
    dist = {}
    for key, cell in counts.items():
        total = sum(cell.values())
        dist[key] = {
            k: (cell.get(k, 0) + SMOOTH) / (total + SMOOTH * 12)
            for k in range(1, 13)
        }
    return dist


def age_factor(details, uid, window_start):
    """Fraction of the 72h window a game stays eligible (28-day chart rule)."""
    if not details:
        return 1.0
    info = None
    if isinstance(details, dict):
        info = details.get(str(uid)) or details.get(uid)
    elif isinstance(details, list):
        for d in details:
            if isinstance(d, dict) and d.get("universe_id") == uid:
                info = d
                break
    if not info or not info.get("created"):
        return 1.0
    try:
        created = parse_ts(info["created"])
    except (ValueError, TypeError):
        return 1.0
    exit_at = created + timedelta(days=28)
    remaining_hours = (exit_at - window_start).total_seconds() / 3600.0
    if remaining_hours >= 72:
        return 1.0
    if remaining_hours <= 0:
        return 0.10
    return max(0.10, remaining_hours / 72.0)


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    window_start = parse_ts(task["window"]["from"])

    reads = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chart = {}
            order = []
            for g in sorted(row.get("up_and_coming") or [], key=lambda x: x["rank"]):
                chart[g["universe_id"]] = g["rank"]
                order.append(g["universe_id"])
            if chart:
                reads.append(
                    {"at": parse_ts(row["at"]), "chart": chart, "order": order,
                     "details": row.get("details")}
                )
    reads.sort(key=lambda r: r["at"])
    if not reads:
        raise SystemExit("no reads in history")

    dist = build_kernel(reads)
    # Pooled fallback: if a (rank band, lag bucket) cell has no pairs at all
    # (short history), fall back to all bands pooled for that bucket.
    pooled = {}
    for bi in range(len(LAG_BUCKETS)):
        cell = {}
        for (bd, b), d in dist.items():
            if b == bi:
                for k, v in d.items():
                    cell[k] = cell.get(k, 0.0) + v
        n_cells = sum(1 for (bd, b) in dist if b == bi)
        if n_cells:
            pooled[bi] = {k: v / n_cells for k, v in cell.items()}
    last = reads[-1]
    chart = last["order"]
    candidates = chart[:TOPN]

    game_p = {}
    for uid in candidates:
        rank = last["chart"][uid]
        bd = band(rank)
        p = {k: 0.0 for k in range(1, places + 1)}
        wtot = 0.0
        for bi in range(len(LAG_BUCKETS)):
            d = dist.get((bd, bi)) or pooled.get(bi)
            if d is None:
                continue
            for k in range(1, places + 1):
                p[k] += d.get(k, 0.0)
            wtot += 1.0
        if wtot:
            for k in range(1, places + 1):
                p[k] /= wtot
        # games never seen in a kernel cell keep a flat prior over the places
        if wtot == 0:
            for k in range(1, places + 1):
                p[k] = 0.02
        factor = age_factor(last["details"], uid, window_start)
        if factor != 1.0:
            for k in range(1, places + 1):
                p[k] *= factor
        game_p[uid] = p

    probabilities = []
    for k in range(1, places + 1):
        total = sum(game_p[u][k] for u in candidates)
        if total <= 0:
            continue
        for u in candidates:
            v = game_p[u][k] / total * (1.0 - UNNAMED)
            if v >= MIN_P:
                probabilities.append(
                    {"universe_id": u, "place": k, "p": round(v, 6)}
                )

    def expected_rank(uid):
        p = game_p[uid]
        mass = sum(p.values())
        return sum(k * p[k] for k in range(1, places + 1)) + (places + 1) * max(
            0.0, 1.0 - mass
        )

    top10 = sorted(candidates, key=lambda u: (expected_rank(u), u))[:places]

    forecast = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
