# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Eman
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Eman123-123/flyrank-ml-internship
- **Date:** 2026-09-14

## 0. Abstract

Which of a content team's existing pages should be prioritized for review this week, using only
observed search and content signals? I built a decline-risk classifier on the FlyRank ML
Internship starter dataset (30,000 pseudonymized pages, 32 clients) and compared it against a
transparent rule-based baseline on an identical client-held-out split. Logistic regression reached
a precision@50 of 0.74 versus the baseline's 0.48 (base rate 0.517) while staying interpretable,
and two deliberate stress tests — a naive random split and a reintroduced label-derived feature —
each showed how easily that number can be overstated if validation is sloppy. The output is a
ranked, reason-coded action queue meant for a human reviewer's weekly triage, not an automated
publishing decision.

## 1. Problem framing

**Unit of analysis:** one content page (row), pseudonymized, belonging to one pseudonymized
client. **Decision supported:** which pages a content/SEO team should review first for possible
refresh, given a limited weekly review budget. **Output:** a ranked queue with a probability
score, an action label (`refresh_priority_1` / `refresh_review` / `schedule_refresh` / `monitor` /
`no_action`), and a reason code. **Cost of a wrong call:** a false positive wastes reviewer time on
a page that wasn't really at risk; a false negative lets a genuinely declining page go unreviewed,
costing search visibility that's harder to win back the longer it's ignored. **Why ML helps:** a
single threshold rule (like the Week-4 baseline) can only combine 2-3 signals at once; the pattern
that separates a page trending down from one trending flat/up involves several signals
simultaneously (traffic volume, position, freshness, content depth), which is exactly where a
learned combination earns its keep over a hand rule — as long as it's honestly validated.

## 2. Data safety

**Source:** `data/raw/content_refresh_anonymized.csv` — 30,000 rows, 44 columns, 32 pseudonymized
clients, trailing-90-day metrics (per `docs/data-dictionary.md`).

**Deliberately excluded from every feature list in this project:**
- `trend_direction`, `trend_pct` — the label is computed FROM these.
- `impressions_last_30d`, `impressions_prev_30d`, and their `clicks_`/`sessions_` twins — their
  difference IS the label; w06's leakage audit shows re-adding just two of these columns pushes
  ROC AUC from 0.625 to 0.846 and precision@50 to a suspicious 1.0.
- `content_id`, `client_id` — pseudonyms, used only for grouping and splitting.
- `provider_used`, `model_used` — explicitly flagged "not a model feature" in the data dictionary.

No client names, domains, URLs, credentials, or raw queries appear anywhere in `work/`. Confirmed
by direct inspection of every notebook and this report.

## 3. Baseline

The Week-4 rule (`w04_baseline_score.ipynb`): `stale * visible * impressions` plus a bonus for
pages that rank well (avg position ≤ 10) but pull CTR below their position bucket's median — no
fitted weights, a rule a non-engineer can read in one sentence. For a fair model-vs-baseline
comparison in w05/capstone, its two thresholds (the impressions median, the per-position-bucket
median CTR) are **fit on the training clients only** and frozen before scoring the held-out test
clients — computing them on the same rows being scored would itself be a mild form of peeking, and
an earlier draft of this pipeline confirmed it: precision@50 on self-fit thresholds came out to an
inflated 0.88 versus the honest 0.48 once training-only thresholds were used.

| Model | ROC AUC | Avg precision | Precision@50 |
|---|---|---|---|
| **baseline_rules (Week 4)** | 0.544 | 0.556 | **0.48** |

Base rate on the same test split: 0.517 — the baseline barely beats chance at K=50.

## 4. Model / analysis

**Method:** per the training-honest-models menu (yes/no label, observed proxy → logistic
regression, then random forest), I trained logistic regression, a shallow decision tree, random
forest, and gradient boosting — testing whether added complexity earns its place rather than
assuming it does.

**Target:** `target_declining` = 1 when `trend_direction == "down"` — a 30-day-vs-previous-30-day
impressions swing, treated throughout as a proxy for "at risk," not a verified outcome.

**Features (leakage-safe):** trailing-90-day counts (log-transformed: impressions, clicks,
pageviews, sessions, users, engaged sessions, ai sessions, scroll events), `ctr`, `avg_position`
(cleaned so `0` = "no data" is `NaN`, with a `has_position` flag), `content_age_days`,
`days_since_last_update`, keyword-context fields (`search_volume`, `competition`, `cpc`, with
`has_keyword_data` flags for systematic missingness), `word_count`/`char_count` (with
`has_word_count` flags), and categoricals `competition_level`, `content_type`, `main_intent`.

## 5. Evaluation

**Split:** grouped by `client_id` (`GroupShuffleSplit`, 25% of clients held out, verified zero
client overlap) — the honest question is "does this generalize to a client the model has never
seen," not "can it memorize this client's baseline traffic level."

| Model | ROC AUC | Avg precision | Precision@50 |
|---|---|---|---|
| baseline_rules (Week 4) | 0.544 | 0.556 | 0.48 |
| **logistic_regression** | 0.625 | 0.622 | **0.74** |
| decision_tree | 0.605 | 0.588 | 0.42 |
| random_forest | 0.607 | 0.591 | 0.50 |
| gradient_boosting | 0.623 | 0.613 | 0.72 |

**Split-sensitivity check (w06):** the same logistic regression under a naive random row split
(instead of client-grouped) scores precision@50 = 0.94 — a 20-point overstatement versus the
honest 0.74, purely from letting a client's pages appear in both train and test.

**Leakage audit (w06):** adding `impressions_last_30d`/`impressions_prev_30d` back into the
"honest" feature set moves ROC AUC from 0.625 to 0.846 — the textbook confession that those
columns are label-derived, confirming they're correctly excluded from every real model here.

**Errors (w05):** false positives (flagged as declining, actually not) sit at a higher median
impression volume (~530) than false negatives (~82) or correct predictions (~290) — the model
over-flags some large, stable pages. False negatives concentrate in low-traffic `keyword article`
rows (17% FN rate vs <1% for `comparison article`) — exactly where the 30-day trend proxy is
noisiest, so some of this "error" is a proxy-label ceiling rather than a pure modeling gap.

## 6. Interpretation

Permutation importance (scored by average precision) on the winning model: overall traffic
volume — `log_impressions_90d`, `log_clicks_90d`, `log_users_90d` — dominates, with
`avg_position_clean` a clear second. This is a sane, non-circular story: lower-traffic pages have
noisier 30-day counts, so they cross the ±20%(ish) trend threshold more easily — the model has
partly learned "which pages are volatile enough to swing," a legitimate if limited signal. No
single feature pushed AUC anywhere near 1.0 alone, which is the leakage smell test coming back
clean. A negative/nuanced result worth stating plainly: the deeper models (decision tree, random
forest, gradient boosting) did **not** outperform logistic regression on the decision metric —
added complexity here bought nothing, which is itself a valid finding, not a shortfall.

## 7. Recommendation

The ranked action queue (`w07_action_playbook.ipynb`), built on the same held-out test rows the
model never trained on: **1,793** pages as `refresh_priority_1` (high decline risk, already
ranking on page 1 — highest payoff), **1,103** as `refresh_review` (high risk, not yet ranking
well), **3,538** as `monitor`, **681** as `no_action`. A content reviewer works the queue top-down
by score. **Confidence:** decision-support only — validated on 32 clients and one 90-day snapshot;
not causal; not evidence a refresh will fix any specific page. A human must sanity-check
`avg_position > 0` (real position data) and rule out seasonal/one-off dips before acting, and
nothing here should trigger automated publishing or de-indexing.

## 8. Reproducibility

Random seed `42` fixed everywhere (`GroupShuffleSplit`, `train_test_split`, every sklearn
estimator). Re-run from a fresh clone:

```bash
git clone https://github.com/Eman123-123/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w05_model.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w06_validation_audit.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w07_action_playbook.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```

All four notebooks were executed top-to-bottom with no errors on scikit-learn 1.8.0 / pandas
3.0.2 / numpy 2.4.4; small third-decimal shifts are expected on older library versions (per
`GUIDE.md`'s own note on this). The committed receipt for the action-queue numbers is
`work/outputs/w07_action_playbook_summary.json`.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

---

> **Claims checklist:** observed / measured / directional / decision-support language used
> throughout; base rate (0.517) reported next to every precision@50; no causal claims; no
> "predicted Google's algorithm" language; no client-identifying details anywhere in `work/`;
> numbers in this report match a fresh re-run of the four notebooks above.
