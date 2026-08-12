# Historical roadmap: v4 Stockfish-2500 harness target

> This is the dated strategy and experiment log for the 2026 v4 effort. For
> current implementation and release status, use
> [`PROJECT_STATUS.md`](PROJECT_STATUS.md). For model limitations and release
> provenance, use [`MODEL_CARD.md`](MODEL_CARD.md).

*2026-07-02 — written after targeted pipeline audits (see `tests/audit_*.py`)
found no defect in the exercised encoding, label, loading, and value-semantics
cases. Those tests bound the known failure modes; they do not prove the full
training stack correct or bug-free.*

**Interpretation rule:** unless stated otherwise, every Elo number below is a
historical estimate from this repository's model-vs-model or Stockfish
`UCI_LimitStrength` harness. It is not a FIDE, online-platform, or human
tournament rating. The underlying checkpoints, logs, PGNs, and dataset snapshot
are not all committed, so the dated results should be treated as experiment
records rather than independently reproducible release evidence.

## 1. Historical benchmark target

**Target used by the project:** score >= 50% against Stockfish 18 configured
with `UCI_LimitStrength=true` and `UCI_Elo=2500`, at 600 ms/move for Stockfish,
over >=24 alternating-color games, with the model spending <=~2 s/move on the
local GPU.

The 2026-08-11 run exceeded the score threshold as a 24-game point estimate,
but the harness did not isolate model latency or report a confidence interval.
Accordingly, this document records the narrow score threshold as met while the
full target and any general “2500 Elo” claim remain unverified.

The historical measurements below use
`evaluation/vs_stockfish.py --uci_elo <N>` (added 2026-07-02). Secondary metrics per
checkpoint: held-out top-1/top-5 (`evaluation/evaluate_model.py`), per-phase blunder
rate (`evaluation/diagnose_value_rerank.py`), and model-vs-model Elo
(`evaluation/eval_arena.py`).

The evaluator has since changed: its default is now 16 paired/color-reversed
openings (32 games) with a fixed seed, pair-level confidence units, JSON + PGN
artifacts, model latency percentiles, and optional fail-closed score/latency
gates. Unless a changelog entry explicitly says it used that protocol, assume it
used the historical start-position-only harness.

The current Phase 2 driver requires an explicit immutable dataset revision and
writes each gauntlet's artifacts under `evaluation/results/phase2_<elo>`. It
enforces p95 model latency <=2 seconds at every rung and a 95% score lower bound
>=50% at the 2500 rung. Those stronger gates apply to future runs; the historical
12W-9D-3L result was not rerun and does not inherit them. It also predates the
current v4 input mask, so it does not validate current masked inference.

## 2. Where we are (measured 2026-07-02)

| Metric | Value |
|---|---|
| v3 checkpoint (128x8 ResNet, 20-plane perspective encoding) | val_loss 1.6914, held-out top-1 44.2% |
| vs v2-resumed, greedy policy, 32 paired games | +137 Elo (95% CI [+40, +261]) |
| vs SF Skill 10 @600ms — policy+guard | 0W-1D-7L (6.2%) |
| vs SF Skill 10 @600ms — MCTS 200 sims | 0W-1D-7L (6.2%) |
| Historical harness-fit estimate | ~1400-1500 |

Two structural facts explain this, and both are well documented in the literature:

1. **Imitation of humans caps hundreds of Elo below the humans imitated.** The
   model predicts GM moves at 44% but chess punishes the *worst* move per game,
   not the average; errors compound into positions absent from GM training data.
   This interpretation is consistent with the cited Maia work, and the local
   phase-degradation diagnostic recorded a similar shape (blunder rate 0.7% in
   book openings -> ~14% in novel middlegames). The underlying diagnostic
   artifact is not committed.
2. **Search cannot rescue a noisy value head.** MCTS-200 added exactly nothing
   because the value head (trained on sparse game outcomes + shallow evals) has
   ~0.3 correlation after move 20. PUCT steered by noise just reshuffles moves.

## 3. The strategy in one paragraph

This section preserves the original plan. The implementation later kept a
scalar tanh value head rather than the proposed WDL softmax because the existing
consumers already used scalar values and the first distillation run made search
useful without that migration.

Switch the training signal from *"what did the human play"* to *"what does a
2600+-strength oracle say"* (engine distillation), at a scale of hundreds of
millions of positions that we get **for free** from the Lichess evaluations
database; train a moderately larger SE-ResNet with a WDL value head on it; then
let an upgraded MCTS amplify the now-trustworthy value head at inference. Self-play
RL is the *last* phase, used only to close a remaining gap — not the engine of the
plan, because distillation gets more Elo per GPU-hour on a single 8 GB card.

Evidence this works at our scale: DeepMind's *Grandmaster-Level Chess Without
Search* (2024) distilled Stockfish 16 action-values into transformers by pure
supervised learning — their **9M-parameter** model (smaller than our current net!)
reached ~2000 Elo *without any search*, and 270M reached grandmaster blitz level.
We don't have their 15B-datapoint budget, but we also don't need searchless play:
Leela-style engines show that search can materially improve a suitable
policy/value network. The project therefore treated the Stockfish-2500 harness
threshold with search as a plausible single-GPU target. That was a planning
hypothesis, not a guarantee of a human rating.

## 4. Elo budget (estimates, gated by measurement)

| Step | Expected strength | Basis |
|---|---|---|
| Today: v3 policy+guard | ~1450 | measured vs SF ladder |
| + engine-distilled retrain (WS1+WS2) | raw policy 2000-2200 | DeepMind 9M param ~2000 searchless; our net is bigger, data smaller |
| + search-worthy value head via distilled WDL + MCTS upgrades at 800-1600 nodes (WS3) | +300-500 | node-scaling behavior of Lc0-family nets (~+50-100 Elo per nodes doubling in this regime) |
| + self-play polish (WS4, only if needed) | +100-200 | AlphaZero-style policy improvement, infra already built |
| **Total** | **~2400-2700** | gates at each phase |

## 5. Workstreams

### WS0 — Measurement (done first; partially landed 2026-07-02)

- `vs_stockfish.py --uci_elo` ladder (1400/1700/2000/2300/2500), 6-12 games per
  rung, per checkpoint. Add `--openings` (reuse `eval_arena`'s paired suite) once
  draws become common at higher levels.
- Convention: every training phase ends with the same gauntlet; numbers go in this
  file's changelog.

### WS1 — Data: Lichess evaluations ingestion (the biggest lever)

**Historical source note:** at the time, project notes described the
`database.lichess.org` evaluations export and its HF mirror,
`Lichess/chess-position-evaluations`, as **944,957,425 rows / 388,458,657 unique
positions, ~41 GB Parquet, CC0**, updated monthly. Those mutable-source figures
and terms must be rechecked against a pinned snapshot before a new run. Schema
per row: `fen`, `line` (principal variation, UCI), `depth`, `knodes`, `cp`,
`mate`.

- **Policy target:** first move of `line` (Stockfish's best move at that depth).
- **Value target:** `cp`/`mate` mapped to WDL probabilities (logistic in cp;
  constants tuned in Phase 1; `tanh(cp/600)` scalar as fallback — already plumbed).
- **Filtering:** keep deepest record per FEN; drop `depth < 12`; cap |cp| at 1500;
  mate -> +/-1.0 WDL saturation.
- **Storage design decision:** store `(fen, best_move_uci, cp, mate, depth)` only
  (~20 GB Parquet), and **encode board tensors on the fly in DataLoader workers**.
  Materializing float32 tensors for 388M positions would be ~2 TB — a non-starter —
  while FEN->tensor encoding costs ~50 µs/position, so 4-8 persistent workers
  sustain ~10k positions/s, which was expected to saturate the GPU for the WS2
  net. This also reuses the targeted-audit-covered
  `board_to_tensor_v3`/`move_to_policy_index_v3` codecs. The original claim that
  four-field FEN training exactly matched serving was incorrect: live boards can
  supply a halfmove clock and repetition history that the corpus never
  supervised. Current v4 masks those three auxiliary input planes instead.
- **Correctness gate (non-negotiable):** an audit-style ingestion test in the
  spirit of `tests/audit_labels_data_test.py`:
  - verify the `cp` sign convention empirically (White-POV vs side-to-move POV)
    on known mate-in-1 rows before trusting any label;
  - assert every policy target is legal in its FEN via the existing codec;
  - spot-check WDL mapping monotonicity and symmetry under color swap.
- **Optional mix:** 10-20% existing GM chunks for opening breadth. Note the product
  tradeoff: pure distillation plays like a small Stockfish, not like Magnus. Keep
  the v3 checkpoint as the "human-style" serving mode (encoding dispatch already
  supports two generations side by side).

### WS2 — Model v4

- **Tower:** SE-ResNet (squeeze-and-excitation blocks, the Lc0-proven Elo/param
  upgrade), **256 filters x 12 blocks** (~35M params, ~140 MB fp32). Fits 8 GB
  VRAM with AMP at batch 512-1024 (gradient accumulation if needed).
- **Keep:** the 20-plane perspective tensor schema and the 76-plane
  convolutional policy head, with no encoding-version bump. The implemented v4
  correction masks clock/repetition channels 17-19 because the training FENs
  did not supervise them, leaving 17 effective model inputs. Targeted audits
  cover the exercised codec mappings; they are not a proof of every position.
- **Value head:** WDL 3-way softmax (win/draw/loss from side to move) replacing
  the scalar tanh; cross-entropy on soft WDL targets from cp. Better calibrated
  for search; `Q = P(win) - P(loss)` at inference. Optional later: moves-left head.
- **Recipe:** AdamW + cosine or step LR, label smoothing over legal moves (existing
  code), AMP + `channels_last` + `torch.compile` (~1.5-2x step speed).
- **Throughput (measured 2026-07-02, RTX 3070, AMP, no compile):** 256x12 =
  4,400 pos/s (14.9M params, 3.2 GiB peak); 192x10 = 8,100 pos/s; current 128x8 =
  18,200 pos/s. So: 50M-position epoch ~3.2 h (256x12); full-388M epoch ~24 h.
  `torch.compile` + `channels_last` expected to add ~1.5-2x on top.
- **Disk constraint:** C: has only ~28 GB free. Phase 1's 50M subset (~3 GB) fits;
  Phase 2's full filtered dataset (~15-20 GB compact parquet, processed by
  streaming shards and deleting raws) requires freeing ~30-40 GB first or using an
  external drive.

### WS3 — Search (inference) upgrades

In `inference/mcts_player.py`, in impact order:
1. **Tree reuse between moves** (advance the root; today the tree is rebuilt from
   scratch every move — this is a free 2-3x effective sims).
2. **fp16 inference + torch.compile** for the evaluator -> 2-4x more sims/s.
3. **FPU (first-play urgency):** initialize unvisited children to parent Q minus a
   margin instead of 0.0 — the 0-init currently over-explores refuted moves.
4. **Retune `policy_temperature`:** the 1.5 softening was a crutch for the peaky
   imitation policy; a distilled policy wants ~1.0-1.15.
5. Sims budget 800-1600 within ~1-2 s/move on the local GPU.
6. Retire the blunder guard for the v4 net (a depth-2 PST search will start
   vetoing moves *better* than its own understanding); keep it for v3 serving.
7. Stretch: Gumbel-MCTS root selection (better at low sim counts).

`inference/search_player.py` (NN alpha-beta + quiescence) stays as a measured
alternative — with a trustworthy value head it may rival MCTS at equal wall time.

### WS4 — Self-play RL (only if Phases 1-2 land short of gate)

The loop (`experiments/train_self_play.py`) has targeted tests, gating logic, and
anchoring, but generation is sequential (one game and one tree at a time).
Before any serious RL, the proposal was to run G=32-64 games concurrently and
pool leaf evaluations across games into batched GPU calls. The former 10-20x
throughput figure was a planning estimate, not a measured result. Bootstrap
from the v4 distilled net; anchor with distilled chunks (not human ones); reuse
the existing gate. Also fold in the two known small fixes: honor
`sample_weight` in the anchor mix, and preserve enough board history for
repetition-aware MCTS.

### WS5 — Serving / product

The following items were proposed, not delivered by the v4 training change:

- Separate human-imitation v3 and engine-distilled v4 product modes. The current
  backend loads one checkpoint and the UI exposes one neutrally labeled neural-
  network option, so dual-mode serving remains future work. The `magnus` API
  value remains only as a legacy compatibility identifier.
- Strength presets for playability (Elo-limited play via sims/temperature), so the
  strong engine is also fun to play against.
- Backend: expose sims/movetime per request (cap exists); health endpoint reports
  which generation is loaded (already does).

## 6. Phases and gates

| Phase | Work | Exit gate (vs_stockfish, 600ms SF) | Rough effort |
|---|---|---|---|
| 0 | UCI_Elo baseline ladder for v3 | numbers recorded below | done same-day |
| 1 | WS1 ingestion (50M subset) + WS2 net + first distillation train + ingestion correctness tests | raw policy >= 55% vs UCI_Elo 1900 | days (GPU-hours: ~10-20) |
| 2 | full-data retrain; WS3 upgrades originally proposed | MCTS >= 50% vs UCI_Elo 2300 | **Training complete 2026-08-11** — 75% point estimate vs 2300 at MCTS-200; most WS3 upgrades were not needed or implemented |
| 3 | close any remaining benchmark gap | >= 50% vs UCI_Elo 2500, >=24 games, plus the target latency budget | **Score point estimate exceeded 2026-08-11** — 68.8% vs 2500 at MCTS-200; confidence and isolated latency were not recorded |

**Fallback logic:** if Phase 1 misses its gate, scale data (full set) and/or width
(256x16) *before* touching RL — supervised distillation is the cheapest Elo here.
If Phase 3 stalls ~2300-2400, options are: bigger tower (needs patience on 8 GB),
longer thinking time, or revisiting the benchmark target explicitly.

## 7. Risks

- **cp sign convention wrong at ingestion** -> value head trained inverted; the
  mandatory WS1 mate-in-1 probes are intended to detect this exercised failure
  mode, not prove every source row correct.
- **Data distribution bias:** Lichess evals cover positions *humans requested
  analysis for* — heavy on real-game middlegames including mistakes. That is
  exactly the distribution our imitation net lacked; still, monitor opening play
  (option: small GM-data mix).
- **8 GB VRAM:** 256x12 + batch 1024 AMP fits; if not, accumulate. Never block on
  hardware — width beats depth for VRAM at equal params here.
- **Style regression:** distilled net plays engine-like. Separate v3/v4 product
  modes were proposed as mitigation but are not implemented.
- **Draw saturation at high rungs:** add paired openings to the gauntlet when
  draws exceed ~30%.
- **Evidence quality:** preserve PGNs and a machine-readable result manifest;
  report confidence intervals and model-only latency. The 2026-08-11 2500 rung
  had 9/24 draws (37.5%) but was still run from the standard initial position,
  so a paired-opening confirmation remains due. The evaluator now implements a
  seeded, paired/color-reversed default protocol with JSON, PGN, artifact
  identities, latency percentiles, and fail-closed lower-bound/latency gates;
  no v4 rerun under that protocol has been recorded yet.
- **Windows specifics:** DataLoader workers must stay picklable/persistent
  (existing `PolicyBatchCollator` pattern); long runs should write incremental
  checkpoints (existing behavior).

## 8. Sources

- Lichess open database (evaluations export): https://database.lichess.org/
- HF mirror + schema: https://huggingface.co/datasets/Lichess/chess-position-evaluations
- Ruoss et al., *Grandmaster-Level Chess Without Search*, 2024: https://arxiv.org/abs/2402.04494 (code: https://github.com/google-deepmind/searchless_chess)
- McIlroy-Young et al., *Maia: Aligning Superhuman AI with Human Behavior* (imitation-strength ceiling): https://maiachess.com/
- Leela Chess Zero project (SE-ResNet architecture, node-scaling behavior): https://lczero.org/

## 9. Changelog / measurements

- **2026-08-11 — full v4 training completed; Stockfish-2500 harness score
  threshold exceeded as a 24-game point estimate; local serving smoke-tested.**
  Full-dump distillation retrain finished: SE-ResNet
  256x12 (15.1M params), 2 epochs over the full 20-shard Lichess-evals dump —
  **393.2M unique positions** (392.97M train / 250k val, hash-disjoint),
  min_depth 12, WHITE-POV cp negated to side-to-move (mate-in-1 re-verified at
  ingest). Castling-best rows now included (`dropped_bad_castle: 0` — the 1.55%
  the Phase-1 strict reader skipped). Trained at ~4,190 pos/s (the fast path;
  `channels_last` stays off), LR staircase 1e-3 -> 3e-4. Final **val loss 1.3826,
  top-1 54.0%** (Phase-1 baseline 1.4760 / 52.0%). Checkpoint:
  `model/grandmaster_resnet_v4_full.pt`.

  Recorded gauntlet configuration: v4-full, MCTS-200, batch size 16, policy
  temperature 1.5, no blunder guard, Stockfish 18 limited-strength mode at
  600ms/move, 24 alternating-color games per rung. The harness started every
  game from the standard initial position.

  | Opponent setting | Result | Point-estimate score |
  |---|---|---|
  | `UCI_Elo=2000` | 22W-1D-1L | 93.8% |
  | `UCI_Elo=2300` | 15W-6D-3L | 75.0% |
  | `UCI_Elo=2500` | 12W-9D-3L | 68.8% |

  The 2500-rung point estimate exceeds the preselected 50% score threshold at
  MCTS-200 without the proposed tree reuse, FPU, 800+ simulations, or policy
  temperature retuning. It does **not** establish a precise implied Elo: no
  confidence interval was reported, nine draws exceed the roadmap's 30% trigger
  for paired openings, and the model's <=~2-second latency condition was not
  isolated by the harness. PGNs, logs, model digest, engine digest, and a
  machine-readable run manifest are not committed.

  This run also used the predecessor unmasked v4 input path. Current v4 masks
  the halfmove-clock and repetition channels that its four-field training FENs
  never supervised. Because that changes inference, this result must not be
  treated as a benchmark of the current masked model.

  A local backend smoke test via `deploy_v4.ps1` reportedly loaded the v4
  checkpoint with MCTS-200 and returned 1.e4 from the starting position. That
  establishes local integration only, not a Railway production deployment.
  The model file is gitignored and larger than GitHub's normal per-file limit;
  an immutable external artifact, digest, and tested rollback target still need
  to be recorded before declaring a production release. WS4 self-play was not
  needed to exceed the narrow score point-estimate threshold.

- **2026-07-05 — A machine-specific operational handoff was added.** It has
  since been superseded by `PROJECT_STATUS.md`, `MODEL_CARD.md`, and
  `RAILWAY_DEPLOY.md`; `HANDOFF_2500.md` now preserves only historical context.
  Support fixes
  landed with it: `run_phase2.ps1` pins `.venv\Scripts\python.exe` because the
  workstation's bare `python` was an unprovisioned 3.14; trainer `--resume`
  semantics were clarified in help + an explicit “nothing left to train” note.
  At that time a partially trained epoch counted as complete; the current
  trainer instead records completion state and replays/skips completed batches
  during mid-epoch resume.
  `tests/distill_ingest_test.py` dataset dir now overridable via
  `DISTILL_DATA_DIR` for validating the Phase-2 ingest. The local full-suite run
  was reported green; no CI run or result artifact is linked.

- **2026-07-03 — Phase-2 pre-flight review (fixes before the full-dump run):**
  - Castling-normalization paths (ingester + trainer reader) now covered by an
    end-to-end unit test on a synthetic raw shard (O-O, O-O-O normalized;
    rook-lift `e1h1` with king off e1 preserved; POV negation; dedupe-by-depth;
    depth filter) plus a 200k-row live sweep: the fixed reader consumes every
    row (was skipping 1.55%).
  - `--buckets` default 16 -> 64: at 388M uniques a 16-way dedupe dict
    (~24M rows) would exceed this machine's 15.7 GB RAM; 64-way stays ~2 GB.
  - Ingester deletes each temp bucket right after consuming it: peak disk for
    the full dump drops from ~25 GB to ~17-18 GB (30 GB currently free).
  - Trainer gained `--val_every_minutes` (default 120): mid-epoch validation +
    best-checkpoint saving, since a full-dump epoch is ~26 h (smoke-tested).
  - A local regression sweep was reported green: 31 unit tests + all audit
    suites, including strict arch-dispatched load of the v4 checkpoint. The
    checkpoint-dependent output is not committed.
  - The dump currently has exactly 20 shards (data_0000..0019).
  - **Serving-config measurements at UCI_Elo 2000 (8 games each):** v4 policy
    WITH blunder guard 31% (vs 50% without — the depth-2 PST guard now vetoes
    moves better than its own eval; retire it for v4). **v4 + MCTS-200: 87.5%
    (7W-0D-1L)** — with v3 the same search added nothing; the distilled value
    head unlocked it, consistent with the roadmap's core bet. A two-rung
    logistic fit produced a harness estimate near 2340, but eight games per
    rung are a smoke test rather than a precise rating estimate.
  - **Confirmation: v4 + MCTS-200 vs UCI_Elo 2300 = 4W-1D-3L (56.2%).** The
    Phase-2 exit gate (>=50% vs 2300) is met by the Phase-1 checkpoint at
    small sample. Combined harness fit over both MCTS rungs: **~2340**. The
    planning gap to the Stockfish-2500 target was therefore estimated near 160,
    against a Phase-2 budget of full-dump
    retrain (+ castling rows) plus tree reuse / 800+ sims / FPU / temp tuning.
    Re-verify the gate at >=24 paired games with the Phase-2 model.

- **2026-07-03 — Phase 1 COMPLETE.** Training: 3 epochs / 46.1M positions /
  ~10 h; val loss 1.4760, top-1 vs Stockfish-best 52.0%, value MAE 0.115, no
  train/val gap. Gauntlet (v4 raw policy, no guard, value rerank on, SF18
  @600ms):

  | Opponent | Result | Score | v3 baseline |
  |---|---|---|---|
  | UCI_Elo 1700 | 6W-0D-0L | 100% | 33% |
  | UCI_Elo 1900 | 1W-4D-3L | 37.5% | ~28% |
  | UCI_Elo 2000 | 3W-0D-3L | 50% | 25% |
  | UCI_Elo 2300 | 2W-2D-2L | 50% | — |

  Combined logistic harness fit over all 26 games: **~2040 (+370 relative to
  the earlier harness fit in one night)**.
  Gate (>=55% vs 1900): the single rung read 37.5%, but the whole-ladder fit
  estimated ~2040, above the ~1935 encoded by the gate — Phase 1 was treated as
  passed for planning;
  the 1900 rung is noise-dominated (8 games) and worth re-measuring during
  Phase 2 tuning.
  Known Phase-1 gaps carried into Phase 2: castling-best rows (1.55%) were
  skipped by the old strict reader (fixed in trainer + ingester + test —
  Lichess PVs use king-takes-rook UCI like e1h1); repetition blindness makes
  the raw policy shuffle into early threefolds in equal positions (4 draws at
  the 1900 rung) — MCTS in-tree repetition awareness + a draw-avoidance nudge
  address this.

- **2026-07-02 (Phase 1 build)** —
  - **cp/mate sign convention verified on raw shard 0: WHITE-POV.** Mate-in-1
    probe: black-to-move rows with `mate=+1` were black mates in 0/150 cases;
    with `mate=-1` in 150/150. Ingestion negates for black-to-move; the
    regression test (`tests/distill_ingest_test.py`) cross-checks the *output*
    against local Stockfish evals and should catch the exercised inversion
    cases if it is run with the required artifacts.
  - PV first move legal for the side to move in 5000/5000 sampled rows.
  - Landed: `training/ingest_lichess_evals.py` (stream, filter depth>=12,
    dedupe-by-deepest via hash buckets, hash-disjoint val split),
    `training/train_distill.py` (v4 trainer, mid-epoch snapshots, resume),
    `ChessModelV4` SE-ResNet + `arch_version` serving dispatch,
    `tests/v4_model_test.py` (all passing incl. tiny overfit).
  - **Windows gotcha:** pyarrow must be imported *before* torch
    (torch 2.6 + pyarrow 24: reverse order segfaults on first parquet read).
  - Ingestion of 3 raw shards: 155.8M rows -> **46.1M unique positions**
    (45.85M train / 250k val, hash-disjoint) in 26 min; 14.1M mate rows;
    dataset validation: all sampled moves legal, Stockfish cross-check
    corr 0.974, sign agreement 100%.
  - **Perf gotcha:** `channels_last` is ~4x SLOWER for this 8x8 workload on
    the RTX 3070 (1,033 vs ~3,600+ pos/s) — off by default in the trainer.
  - **Phase-1 training launched** (256x12 SE, 3 epochs over 46.1M, batch
    1024, ~10 h ETA): log `train_v4_distill.log`, checkpoint
    `model/grandmaster_resnet_v4_distill.pt` (+ `.midtrain` snapshots every
    30 min). Next: UCI_Elo gauntlet, gate >=55% vs 1900.

- **2026-07-02** — Baseline (v3, policy+guard, SF 18 @600ms/move, 6 games/rung):

  | Opponent | Result | Score |
  |---|---|---|
  | UCI_Elo 1400 | 5W-0D-1L | 83.3% |
  | UCI_Elo 1700 | 1W-2D-3L | 33.3% |
  | UCI_Elo 2000 | 0W-3D-3L | 25.0% |
  | Skill 10 (browser-equivalent) | 0W-1D-7L | 6.2% |

  Combined logistic harness fit: **v3 ~= 1670** (±~120 at these sample sizes;
  MCTS-200 measured identical to policy). The same fit placed
  Skill-10@600ms near 2100. The planning gap to the Stockfish-2500 target was
  therefore ~850, with the Phase-1 gate about +250 from this baseline.
