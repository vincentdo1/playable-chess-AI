# Project status

Last updated: 2026-08-11

This page separates four states that older project notes sometimes conflated:
implemented, trained, benchmarked, and production-deployed. See the
[model card](MODEL_CARD.md) for model-specific details and the
[Railway runbook](RAILWAY_DEPLOY.md) for release operations.

## Current state

| Area | Status | Evidence or caveat |
|---|---|---|
| v2 checkpoint compatibility | Implemented | Legacy `perspective_v2` loading remains in code. The checkpoint itself is not stored in Git. |
| v3 human-imitation model | Implemented and used as the code default | `MODEL_PATH` defaults to `model/grandmaster_resnet_v3.pt`. The checkpoint is not stored in Git. |
| v4 engine-distilled architecture | Implemented | `ChessModelV4` and `arch_version='v4'` loading are present. v4 shares the v3 tensor schema, so `arch_version` is required for dispatch. Because four-field training FENs did not supervise clock/repetition state, current v4 masks input channels 17-19 and uses 17 effective planes. |
| Full v4 training run | Reported complete | The roadmap records two epochs over 20 processed Lichess-evaluation shards. Raw logs, a source-dataset revision, and artifact hashes are not committed. |
| Versioned v4 ingest/resume pipeline | Implemented for future runs | New ingests require an immutable HF commit, validate/cache source shards with provenance, and write `ingest_manifest.json`; the trainer verifies the corpus and stores provenance, objective/config, optimizer/scaler/RNG, seed, and completion state. Mid-epoch resume replays and skips completed batches. This cannot recover the historical run's missing source identity. |
| v4 local backend smoke test | Reported complete for the predecessor path | The roadmap records a successful health response and a legal first move with MCTS-200 before the v4 input mask landed. The real checkpoint has not been smoke-tested under the corrected path by tracked evidence. |
| v4 historical strength benchmark | Point estimate recorded under the former protocol and model input behavior | 12W-9D-3L against Stockfish 18 `UCI_LimitStrength`/`UCI_Elo=2500`, 600 ms per Stockfish move, MCTS-200, 24 start-position games. The run predates the v4 channel mask and does not validate current masked inference. See “Benchmark interpretation.” |
| Release-quality evaluation harness | Implemented; v4 rerun pending | The current default uses 16 paired/color-reversed openings (32 games), a fixed seed, pair-level confidence units, artifact/environment metadata, JSON + PGN output, model latency percentiles, and optional fail-closed score/latency gates. |
| Railway download/serving path | Implemented | The backend requires an immutable HF revision by default, can verify `MAGNUS_MODEL_SHA256`, uses restricted checkpoint loading plus a smoke forward, exposes liveness/readiness, enforces server-owned MCTS limits, and bounds inference concurrency. |
| Railway production release of v4 | Not established by tracked evidence | The repository does not record an immutable model revision, checkpoint digest, deployment ID, or production verification result. |
| Dual human-style and engine-distilled product modes | Not implemented | One backend process loads one checkpoint. The browser exposes one neutrally labeled neural-network option plus an MCTS toggle. |

## Model generations

| Generation | Training signal | Architecture | Encoding | Intended role |
|---|---|---|---|---|
| v2 | Human games | ResNet + move-history LSTM | `perspective_v2` (17 planes) | Legacy compatibility |
| v3 | Magnus/GM games | 128-filter, 8-block ResNet | `perspective_v3` (20 planes) | Human-imitation baseline and local default |
| v4 | Lichess position evaluations distilled from Stockfish analysis | 256-filter, 12-block SE-ResNet | `perspective_v3` tensor schema plus `arch_version='v4'`; clock/repetition channels 17-19 masked | Engine-distilled model; current strength unconfirmed after input-mask change |

The browser now uses the neutral **Neural Network** label. The `magnus` API value
and `magnus_carlsen` desktop value remain as legacy compatibility identifiers.
They do not imply that v4 simulates Magnus Carlsen or that this project is
affiliated with or endorsed by him.

## Benchmark interpretation

The strongest recorded v4 result is:

| Model configuration | Opponent configuration | Games | Result | Score |
|---|---|---:|---:|---:|
| v4-full, predecessor unmasked input path, MCTS-200, policy temperature 1.5, blunder guard off | Stockfish 18, `UCI_LimitStrength=true`, `UCI_Elo=2500`, 600 ms/move | 24 | 12W-9D-3L | 68.8% |

This is a harness-specific historical point estimate. It is not a FIDE,
online-platform, or human tournament rating. The run used the predecessor
protocol, which alternated colors but started every game from the standard
initial position and relied on Stockfish's limited-strength variation for
diversity. It did not preserve a confidence interval, paired-opening record,
PGN bundle, machine-readable result manifest, or isolated model latency. Nine
draws in 24 games also exceeded the roadmap's own threshold for switching to a
paired-opening suite.

The benchmark also used the predecessor v4 inference path, which allowed
halfmove-clock and repetition channels to activate even though the four-field
training FENs never supervised them. Current `ChessModelV4` zeros channels
17-19 at its boundary. That is a defensible fail-closed correction for the
train/serve skew, but it changes model behavior; the historical result is not
evidence for the corrected path.

The evaluator has since been hardened. Its default protocol uses the built-in
16-opening suite with both colors for each opening (32 games), seed `20260705`,
and per-opening-pair averages as confidence-interval units. It records exact
arguments; code, model, Stockfish, and opening identities; environment details;
per-game outcomes; model-move p50/p95/max latency; and JSON plus PGN artifacts.
`--require_score_lower_bound` and `--require_p95_seconds` make those conditions
fail closed. No v4 result from this new protocol is recorded yet, so the old
12W-9D-3L point estimate must not be presented as if it came from the new
harness.

The original target additionally required the model to spend no more than
approximately two seconds per move on the local GPU. The historical run timed
whole games, including Stockfish time, so that latency condition has not been
demonstrated by tracked evidence. The new harness can measure and gate model
p95 latency on a rerun.

Accordingly, the narrow historical point-estimate threshold was met, but a
general claim that the model “is 2500 Elo” is not established. A release-quality
confirmation should run the new default paired protocol with a predeclared
lower-confidence-bound and p95-latency gate, then publish its JSON and PGN
artifacts together with immutable model and data provenance.

## Artifact and reproducibility gaps

Before treating v4 as a reproducible release, record all of the following in a
versioned result manifest and the model repository:

- Hugging Face repository and immutable commit revision;
- checkpoint filename, exact byte count, and SHA-256 digest;
- training code commit;
- source-dataset repository revision and per-shard digests;
- ingest configuration and generated data-manifest digest;
- complete training configuration, RNG seeds, and package/runtime versions;
- Stockfish binary version and digest;
- exact benchmark command, opening suite, PGNs, result JSON, hardware, and
  per-move latency statistics.

New ingests and checkpoints now record more of this provenance, but the
historical full model predates those safeguards. These values are intentionally
not filled with guesses in this repository.

## Release gate

A v4 production release should not be called verified until all of these are
true:

1. The model artifact is immutable, identifiable, and recoverable.
2. A staging cold start loads that exact artifact.
3. `/readyz` confirms the required model/MCTS capability is available;
   `/livez` process liveness alone is not sufficient.
4. The real checkpoint passes canonical-position smoke tests through the current
   masked input path and exact production MCTS cap.
5. Latency, error rate, and memory are observed under representative load.
6. The previous known-good deployment or complete prior artifact/config bundle
   has been tested as a rollback target.
7. The effective browser and server settings agree and are recorded.

## Prioritized follow-ups

1. Publish and pin the model artifact with a model card and digest.
2. Produce a versioned training/evaluation manifest and rerun the current
   masked model with paired openings and latency instrumentation.
3. Validate the Railway configuration in staging, then record the deployment
   and rollback target.
4. Decide whether the product should expose v3 human-imitation and v4
   engine-distilled models separately. Until then, describe the single loaded
   model accurately.
5. Keep the fast pytest CI green and expand clean-checkout coverage without
   disguising skipped checkpoint/data/engine audits as completed integration
   evidence.
6. Maintain the current `CODEOWNERS` assignments and document a separate
   operational escalation owner before production release.

## Documentation authority

- This page is the source of truth for current status.
- [`MODEL_CARD.md`](MODEL_CARD.md) is the source of truth for v4 limitations and
  model-release metadata.
- [`RAILWAY_DEPLOY.md`](RAILWAY_DEPLOY.md) is the operational runbook.
- [`ROADMAP_2500.md`](ROADMAP_2500.md) is a historical strategy and experiment
  log; later entries do not silently rewrite earlier observations.
