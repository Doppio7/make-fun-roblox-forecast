# Predict Roblox's Up-and-Coming top 10

[![make.fun](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun/bounty/roblox-up-and-coming-forecast-01)
[![owner](https://img.shields.io/badge/owner-Doppio7-green)](https://github.com/Doppio7/make-fun-roblox-forecast)
[![referral](https://img.shields.io/badge/referral-5wqvwk46-orange)](https://make.fun/r/5wqvwk46)

Entry for make.fun's **Predict Roblox's Up-and-Coming top 10** ($500) bounty:
a Python program that forecasts the Up-and-Coming chart's top 10 at every hour
of the 72 hours after entries close.

## Method

An empirical rank-transition kernel learned from the chart history in
`/input/history.ndjson`. For every pair of reads up to 72 hours apart, it
records how a game at a given rank band moved (which place it held at the
later read, rank 11-30, or off the chart). Pairs are grouped into lag buckets
(0-6h, 6-12h, 12-24h, 24-48h, 48-72h); each bucket estimates where a game sits
that many hours after the history ends. A game's forecast is the equal-weight
average of its band's distributions over the five buckets, approximating the
average over the 72 hourly slots of the scored window. Per place the named
games share 0.80 of the probability mass and 0.20 is left unnamed, hedging new
entrants and games outside the top 20. Games whose details show they were
created more than 26 days before the window start are scaled down, since the
chart only lists games published within the last 28 days.

Backtested on the practice window (history to 2026-11-07, scored on the 10
archive reads of 2026-11-08 to 2026-11-10): log score **-34.54** against the
house baseline's **-43.51**.

## Run

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Writes `out/forecast.json`: `probabilities` rows
(`universe_id`, `place`, `p`) and `top10`, the ten most likely games in order.
Standard library only, deterministic, no network.
