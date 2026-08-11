# HANDOFF: Road to 2500 Elo — Continuation Guide for Successor Agents

*Written 2026-07-05 by the agent that executed the audit and Phase 1. Audience: an AI
coding agent (Opus 4.8 or any successor) or a human picking this project up cold.
This document is deliberately self-contained: it assumes you have the repo and
nothing else — no conversation history, no memory files.*

**Read order:** this file top to bottom → `docs/ROADMAP_2500.md` (the strategic plan
and the measurement changelog — the single source of truth for all Elo numbers) →
the code files named in §3.

**Conflict rule:** if this document disagrees with the code, the code wins; if it
disagrees with `ROADMAP_2500.md`'s changelog on a *measurement*, the changelog wins.
Report the discrepancy to the user either way.

---

## Table of contents

- §0 State at handoff (start here)
- §1 Prime directives — how to work on this project
- §2 The machine and its environment (read before running anything)
- §3 Codebase map
- §4 Core invariants (never break these)
- §5 The distillation data pipeline, in detail
- §6 Phase 1 — COMPLETE (what was done, all numbers)
- §7 Phase 2 — READY (runbook, expected outputs, what to do after)
- §8 Phase 3 — CONTINGENT (decision tree)
- §9 Measurement protocol (how strength claims are made here)
- §10 Test suite reference
- §11 Debugging playbook (symptom → cause → fix)
- §12 Do's and don'ts
- §13 Decision log (why things are the way they are)
- §14 Open items at handoff

---

## §0 State at handoff (2026-07-05)

**User goal:** a chess neural network that plays at 2500 Elo.
**Definition of done:** ≥50% score vs Stockfish 18 at `UCI_Elo 2500`, 600 ms/move
for Stockfish, over ≥24 paired-color games, with the model spending ≤~2 s/move on
the local GPU (RTX 3070).

**Scoreboard (all measured, see ROADMAP changelog for details):**

| Config | Fitted Elo | Evidence |
|---|---|---|
| v3 (128×8 ResNet, human-imitation), policy+guard | ~1670 | 6-game rungs at 1400/1700/2000, logistic fit over 18 games |
| v4 Phase-1 (256×12 SE-ResNet, engine-distilled), raw policy, no guard | ~2040 | 26 games across 1700/1900/2000/2300 |
| v4 Phase-1 + MCTS 200 sims | **~2340** | 87.5% vs 2000 (8 games), 56.2% vs 2300 (8 games) |
| Goal | 2500 | gap ≈ 160 Elo |

**Where things stand:**
- **Phase 1 is complete** (2026-07-03). Checkpoint: `model/grandmaster_resnet_v4_distill.pt`.
- **Phase 2 is fully prepared and the USER runs it themselves** via `run_phase2.ps1`
  (≈3–4 h ingest + ~52 h training + gauntlet). Do not launch it for them.
- **Phase 3 runs only if Phase 2 lands short of 2500.**
- The Phase-2 exit gate (≥50% vs UCI_Elo 2300 with MCTS) is *already met at small
  sample by the Phase-1 checkpoint* — re-verify at ≥24 games with the Phase-2 model.
- **The entire multi-day working tree is UNCOMMITTED on branch `redo_selfplay`**
  (new: docs/, training/ingest_lichess_evals.py, training/train_distill.py,
  evaluation/vs_stockfish.py, inference/blunder_guard.py, tests/audit_*.py, v4 model
  code, run_phase2.ps1, and more). Offering to commit it is a standing, repeatedly
  deferred item — raise it early.
- Nothing is running in the background at handoff. The browser play servers
  (backend Flask + frontend) are down; start them only on request.
- All tests green as of 2026-07-05 (see §10).

**History in one paragraph:** the project began as a human-game imitation model
("play like Magnus"). A full seven-area pipeline audit (2026-07-02, `tests/audit_*.py`)
proved the training stack bug-free — the strength ceiling (~1450–1670) was the
*approach*: imitation caps hundreds of Elo below the imitated humans, and the value
head trained on sparse game outcomes was too noisy for search to exploit (MCTS added
exactly nothing on v3). The strategy pivoted to engine distillation from the Lichess
evaluations database (§5), which in one night of training produced +370 Elo raw and
made search finally pay (+300 more). That validated bet is the plan you are
continuing.

---

## §1 Prime directives — how to work on this project

These are behavioral rules the user has implicitly or explicitly established. Follow
them as if they were in the prompt.

1. **Measure, never assume.** Every strength claim comes from
   `evaluation/vs_stockfish.py --uci_elo N` (§9). "It should be stronger" is not a
   result. After any model or search change, run the gauntlet and record numbers.
2. **Loss is not strength.** This project's founding lesson: v3's loss curves looked
   great while it played at ~1450. Val top-1 and val loss are progress indicators;
   Elo is the metric.
3. **Record every measurement** in `docs/ROADMAP_2500.md` §9 (Changelog), dated.
   Future decisions get made from that table. If you measure it, write it down.
4. **Never start multi-hour training runs without the user's explicit go-ahead.**
   The user runs the long jobs themselves on their machine (they said so verbatim
   for Phase 2). You prepare scripts, verify correctness, and hand over commands.
5. **Never kill processes you did not start.** Check `Get-Process python*` before
   editing training code (a run may be in flight).
6. **Verify data conventions empirically before training on any new data.** The
   Lichess dump's cp/mate turned out to be White-POV (§4.5) — caught by probing
   before training. A sign error here trains an inverted value head that *still
   reduces loss*. The mandatory pattern: mate-in-1 sign probes + independent
   Stockfish cross-check of the ingested output (`tests/distill_ingest_test.py`).
7. **Write small standalone tests before/alongside pipeline changes** (the
   `tests/audit_*.py` pattern: runnable via `python tests/<file>.py`, assert hard,
   print one summary line). Keep the whole suite green.
8. **Prefer measured micro-decisions.** channels_last, blunder guard, MCTS-vs-policy:
   all were decided by measurement on this exact machine/model, not by lore (§13).
   When you're unsure, measure on a small scale first.
9. **Both model generations stay servable.** v3 = "human-style Magnus" product mode;
   v4 = "Magnus 2500". Don't delete or break v2/v3 loading paths (checkpoint
   dispatch handles them, §4.4).
10. **Ask before destructive or irreversible actions**; commit only when the user
    asks; never push without being asked.

---

## §2 The machine and its environment

Everything below was verified on this machine; assume nothing transfers to other
hardware without re-measurement.

### 2.1 Hardware / OS

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3070, **8 GB VRAM** |
| RAM | 15.7 GB (this is a real constraint — see dedupe buckets, §5.2) |
| CPU | 16 logical cores |
| OS | Windows 11 Home |
| Disk | C: only, ~28 GB free at handoff (476 GB total, 95% used) — check before any big job: `Get-PSDrive C` |

### 2.2 Python — THE most common trap on this machine

**The project interpreter is `.venv\Scripts\python.exe`** (Python 3.12.10) with:
torch 2.6.0+cu124 (CUDA available), pyarrow 24.0.0, python-chess, numpy, Flask.

- **Bare `python` on PATH is `C:\Python314\python.exe` (3.14) with NO packages
  installed.** Any bare `python` invocation fails with `ModuleNotFoundError`.
  Always call `.venv\Scripts\python.exe` explicitly (scripts here do), or activate
  with `.venv\Scripts\Activate.ps1` first.
- There is also a stale `./venv` (no dot) — Python 3.12 *without* pyarrow. Do not use.
- The venv has **no pytest and no scipy**. Tests are standalone scripts (§10); the
  Elo fit is done with the pure-python snippet in §9.3.

### 2.3 Windows/hardware gotchas (each cost real debugging time — details in §11)

1. **`import pyarrow` MUST come before `import torch`** in any process that reads
   parquet. Reverse order → hard segfault (exit 139 / 0xC0000005) on first parquet
   read. torch 2.6.0+cu124 + pyarrow 24.0 DLL clash. `training/train_distill.py`
   and the test files encode this ordering with a load-bearing comment — preserve
   it in any new script, including DataLoader worker re-imports (keeping the import
   at module top of the trainer covers spawned workers).
2. **`channels_last` is ~4× SLOWER on this GPU for 8×8 boards** (measured 1,033 vs
   4,141 pos/s, GPU at 95% both ways). It is an opt-in flag default-off in the
   trainer. Never enable it here; likewise be skeptical of `torch.compile` claims —
   measure before adopting.
3. **Redirected stdout is block-buffered; stderr is not.** In a redirected log the
   python traceback interleaves mid-file, not at the end — `tail` lies to you. Grep
   the whole file for `Traceback|Error`. All prints in the pipeline use
   `flush=True`; keep that discipline. `run_phase2.ps1` also sets
   `PYTHONUNBUFFERED=1`.
4. **Long runs must survive the terminal/session.** Jobs started as children of an
   agent session died when the session recycled. Launch multi-hour jobs detached:
   `Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList '-m','training.train_distill',... -RedirectStandardOutput log.out -RedirectStandardError log.err`
   — or better, have the *user* run them in their own terminal (the Phase-2 plan).
5. **Disable Windows sleep before multi-day runs:** `powercfg /change standby-timeout-ac 0`
   (admin PowerShell). Restore with a nonzero value after.
6. PowerShell 5.1 quirks if you script: no `&&`/`||`, no ternary; `Tee-Object` for
   logging preserves `$LASTEXITCODE` for native exes.

### 2.4 Stockfish

- Binary: `stockfish.exe` in the repo root (Stockfish 18). Override via
  `STOCKFISH_PATH` env var. `evaluation/vs_stockfish.py` absolutizes the path.
- `UCI_Elo` valid range **1320–3190** (the harness clamps). `UCI_LimitStrength`
  mode is the calibrated one — use it for absolute Elo claims. `Skill Level` (0–20)
  is NOT calibrated; it exists to mirror the browser opponent
  (browser = stockfish.js Skill 10 @600 ms ≈ 2100 by our fit).

---

## §3 Codebase map

| Path | Role |
|---|---|
| `neural_network.py` | The core: board encodings (v2/v3), move↔policy-index codecs, `ChessModel` (v2, ResNet+LSTM), `ChessModelV3` (128×8 conv-policy ResNet), **`ChessModelV4` (SE-ResNet, §4.3)**, loss functions, collator, `evaluate()`, legacy trainer `main()`. Env knobs at top (§4.6). |
| `load_model.py` | Serving-side loader (`load_trained_model` — arch dispatch + strict-load policy, §4.4) and move selection (`predict_next_move`, `_get_move_scores` with value reranking + optional blunder guard). |
| `inference/mcts_player.py` | PUCT MCTS with **negamax backup** (child Q negated at selection), batched leaf evaluation. Defaults: `c_puct 1.5`, `policy_temperature 1.5`, board copies `stack=16`. Phase-2 WS3 upgrades happen here. |
| `inference/search_player.py` | NN-guided alpha-beta + quiescence alternative; measured, kept as fallback comparison. |
| `inference/blunder_guard.py` | Depth-2 PST-based veto filter. Helps v2/v3; **HURTS v4** (§13.4) — always `--no_blunder_guard` / guard off for v4. |
| `training/ingest_lichess_evals.py` | Lichess-evals dump → compact distillation shards (§5.2). |
| `training/train_distill.py` | The v4 distillation trainer (§5.3). |
| `training/preprocess.py` | Legacy human-game PGN pipeline (v2/v3 era). Source of `VALUE_CP_SCALE = 600`. Not used by distillation, still used by legacy chunk tooling. |
| `training/dedup_positions.py`, `training/resume_training.py` | Legacy pipeline utilities. |
| `evaluation/vs_stockfish.py` | **The yardstick.** Model vs Stockfish gauntlet (§9). |
| `evaluation/evaluate_model.py` | Held-out top-1/top-5/val-loss on chunk datasets (encoding-aware via checkpoint). |
| `evaluation/eval_arena.py` | Model-vs-model paired matches with Elo CI. |
| `evaluation/diagnose_value_rerank.py`, `diagnose_blunder_guard.py`, `diagnose_phase_degradation.py` | Per-phase cp-loss diagnostics (how the midgame collapse and guard verdicts were measured). |
| `experiments/self_play.py`, `experiments/train_self_play.py` | Self-play RL loop (audited, gated, anchored; **generation is sequential** — needs parallelization before serious use, §8.2). Anchor-guard: raises if supervised anchor dirs contribute 0 positions; `--supervised_dirs ''` = explicit opt-out. |
| `backend/app.py` + `frontend/` | Flask API + browser UI. `MODEL_PATH` env picks the served checkpoint; request params `use_mcts`, `mcts_simulations` (capped), `blunder_guard`. Health endpoint reports the loaded generation. |
| `tests/` | Standalone test scripts (§10). |
| `docs/ROADMAP_2500.md` | Strategy + **measurement changelog (single source of truth)**. |
| `run_phase2.ps1` | The user's one-command Phase-2 driver (§7.1). |
| `model/` | Checkpoints (§6.3). `data/` | datasets (many legacy chunk dirs; the distillation ones are `distill_chunks_v4*`). |

Memory note: if you are running inside Claude Code on this machine, there is also a
persistent memory dir (`~/.claude/projects/C--Users-Vincent-github-playable-chess-AI/memory/`)
with condensed facts. This document supersedes it; trust repo docs on conflict.

---

## §4 Core invariants (never break these)

### 4.1 Board encoding v3 (`perspective_v3`, 20 planes)

- Everything is **side-to-move perspective**: when Black is to move, the board is
  rotated 180° and colors are swapped (`flip=True`), so "my pieces" always face the
  same direction. 20 planes = 12 piece planes (own 6 + opponent 6) + castling
  rights + en-passant + halfmove-clock plane + **2 repetition planes** (v3
  additions over v2's 17).
- The executable spec is `board_to_tensor_v3` in `neural_network.py`; its
  correctness (lossless round-trip, rot180+color-swap invariance, spatial alignment
  with the conv policy head) is pinned by `tests/audit_encoding_orientation_test.py`
  and `tests/v3_encoding_test.py`. If you think the encoding is wrong, run those
  first — they have already falsified that hypothesis once.
- **Repetition planes are all zeros at serving** (the backend reconstructs the board
  from bare FEN, no history) *and now also at training* (distillation encodes from
  FEN). Consistent — but it means the model is repetition-blind (§7.4, known gap).

### 4.2 Policy head and move indexing

- 76-plane convolutional policy head: logits shaped (batch, 64×76=**4864**)
  (`MOVE_VOCAB_SIZE_V3`). Index = `from_sq * 76 + plane`; planes: 56 queen-type
  moves (8 directions × 7 distances) + 8 knight moves + 12 underpromotions.
  `from_sq` is in the **rotated frame** when flipped.
- Codecs: `move_to_policy_index_v3`, `index_to_move_v3`, `legal_policy_indices_v3`.
  Illegal logits are masked (`mask_illegal_logits`) before softmax/argmax
  everywhere — training and serving.

### 4.3 Model zoo

| Arch | Class | Tower | Params | Encoding | Value head |
|---|---|---|---|---|---|
| v2 | `ChessModel` | ResNet+LSTM (move history) | — | `perspective_v2` (17 planes) | scalar tanh |
| v3 | `ChessModelV3` | 128×8 ResNet | ~5M | `perspective_v3` | scalar tanh |
| v4 | `ChessModelV4(filters, blocks)` | **SE-ResNet 256×12** (SqueezeExcitation reduction 8 in every block) | **~15.1M** | `perspective_v3` (same!) | **scalar tanh** (deliberate — §13.3) |

v4's width/depth come from **constructor args stored in the checkpoint**
(`residual_filters`, `residual_blocks`), not env vars — checkpoint metadata alone
rebuilds the exact architecture. Value output: scalar in [−1,1], side-to-move POV.

### 4.4 Checkpoint schema and loading dispatch

Saved payload (see `_save` in `train_distill.py`): `model_state_dict`,
`optimizer_state_dict`, `board_encoding='perspective_v3'`, `arch_version='v4'`,
`residual_filters`, `residual_blocks`, `value_loss_weight`,
`training_kind='lichess_eval_distillation'`, plus `epoch`, `val_loss`,
`val_policy_loss`, `val_value_loss`, `val_value_mae`, `val_move_acc` (and `step`
for mid-epoch saves). Saves are **atomic** (write `.tmp`, `os.replace`).

`load_model.load_trained_model` dispatch order:
1. `arch_version == 'v4'` → `ChessModelV4(filters=ckpt.residual_filters, blocks=ckpt.residual_blocks)`
   (encoding alone can't distinguish v4 from v3 — they share `perspective_v3`);
2. else by `board_encoding`: v3 → `ChessModelV3`, v2 → `ChessModel`.

**Strict-load policy:** any missing tensor aborts with *"refusing to serve a
partially initialized network"* — except the one legitimate legacy case (an old
policy-only checkpoint with no value head at all), which loads with
`value_head_trained=False` and value reranking disabled. Regression-tested in
`tests/audit_checkpoint_loading_test.py`. Don't weaken this.

### 4.5 Value convention (the highest-stakes invariant)

- **Repo-wide:** value = expected outcome for the **side to move**, in [−1,1].
  Training target from engine evals: `tanh(cp_stm / 600)` (`VALUE_CP_SCALE=600`),
  mate → ±1.0.
- **The Lichess evals dataset stores cp/mate from WHITE's point of view.** Verified
  empirically 2026-07-02 (mate-in-1 probe on raw shard 0: `mate=+1` with black to
  move was a black mate in 0/150 rows; `mate=−1` in 150/150). The ingester negates
  for black-to-move rows. `tests/distill_ingest_test.py` re-verifies the *output*
  against local Stockfish (corr 0.974, sign agreement 100%) so an inversion cannot
  silently recur. If you ever re-ingest from a new dump: run that test before
  training. Non-negotiable.
- MCTS backup is **negamax** (value flips sign at each ply; `_select_child` negates
  child Q). `tests/audit_value_semantics_test.py` pins terminals and a mate-in-1
  MCTS Q→+1.0.

### 4.6 Env knobs (defaults in parentheses)

`MODEL_PATH` (model/grandmaster_resnet_v3.pt — **v3 is still the serving default**,
promotion to v4 is a pending Phase-2 step), `VALUE_LOSS_WEIGHT` (0.25; the distill
trainer sets 1.0 before importing neural_network — dense engine labels deserve it),
`LABEL_SMOOTHING` (0.05, applied over legal moves), `HEAD_DROPOUT` (0.1),
`ARCH_VERSION` (v3; only affects `neural_network.main()` legacy training, NOT
serving), `MAGNUS_VALUE_WEIGHT` (2.0, serving-side value rerank),
`MAGNUS_VALUE_CANDIDATES` (0 = rerank all), `STOCKFISH_PATH`,
`DISTILL_DATA_DIR` (tests/distill_ingest_test.py target dir override).

### 4.7 The castling notation trap

Lichess PVs write castling as **king-takes-rook UCI** (`e1h1`, `e1a1`, `e8h8`,
`e8a8`) — ~1.55% of rows. Two rules:
- Parse moves with `board.parse_uci(uci)` (normalizes + validates), never
  `chess.Move.from_uci`.
- **Never validate legality with `Move.from_uci(x) in board.legal_moves`** — 
  python-chess's `__contains__` silently accepts/normalizes these, which is exactly
  how the original validation test missed the bug while the strict encoder skipped
  the rows. Strict checks must go through the actual trainer path
  (`parse_uci` → `move_to_policy_index_v3` → membership in
  `legal_policy_indices_v3`).
- A rook actually sliding e1→h1 also matches the string `e1h1` — that's why the
  ingester normalizes via `parse_uci` per-row instead of string-mapping
  (unit-tested with a rook-lift fixture).

---

## §5 The distillation data pipeline, in detail

### 5.1 The dataset

HF dataset **`Lichess/chess-position-evaluations`** (CC0, mirror of
database.lichess.org evaluations): ~945M rows / **388,458,657 unique positions**,
~41 GB parquet, in exactly **20 raw shards** `data/data_0000.parquet` …
`data_0019.parquet` (~2.1 GB / ~53.6M rows each), direct-download URL pattern in
`ingest_lichess_evals.py`. Row schema: `fen`, `line` (PV in UCI), `depth`,
`knodes`, `cp`, `mate`. Policy target = first PV move; value from cp/mate (§4.5).
Distribution note: these are positions humans requested analysis for — heavy on
real-game middlegames including mistakes, which is precisely the distribution the
imitation net lacked.

### 5.2 Ingestion: `training/ingest_lichess_evals.py`

Flow: download raw shard (curl) → stream-filter-convert → hash-partition to
per-bucket temp parts → (after all shards) per-bucket dedupe keeping the deepest
eval → shuffle → emit fixed-size output shards. Output schema:
`(fen: string, move: string (normalized UCI), value_target: float32 STM-POV,
depth: uint8)`, zstd parquet, `train_XXXX` + `val_XXXX`.

Key args: `--num_shards` (raw shards to pull), `--out_dir` (**must be empty/fresh
or it refuses**), `--min_depth 12`, `--val_positions 250000`, `--buckets 64`,
`--rows_per_shard 2000000`, `--keep_raw` (default: raws deleted after processing).

Engineering constraints encoded there (don't undo):
- **`--buckets 64`:** dedupe loads one bucket's dict in RAM. 388M uniques / 64 ≈
  6M entries ≈ ~2 GB — fits in 15.7 GB RAM. 16 buckets (~24M entries) would page
  the machine to death.
- **Per-bucket temp deletion** right after consumption keeps peak disk ~17–18 GB
  at full scale (vs ~25 GB if temp cleanup waited for the end; only ~28 GB free).
- **Val comes only from hash-bucket 0** → train/val are hash-disjoint by
  construction, no positional leakage.
- Castling normalization at ingest (only the 4 candidate strings pay a
  `chess.Board` construction, ~1.5% of rows; `dropped_bad_castle` stat).
- `ingest_stats.json` written next to the shards — always sanity-check it (§7.2).

### 5.3 Trainer: `training/train_distill.py`

- `DistillShardDataset(IterableDataset)`: workers split shards round-robin
  (`paths[worker.id::num_workers]`), shuffle shard order per epoch, shuffle rows
  within a shard, encode each row on the fly through the audited codecs
  (`parse_uci` → indices → `board_to_tensor_v3`), skip-and-count bad rows (a
  handful must not kill an overnight run; measured skip rate ≈ 0 after the
  castling fix). Yields the same 7-tuple the legacy collator
  (`make_collate_policy_batch`) expects.
- **FEN-only on-the-fly encoding is a design pillar:** materializing tensors for
  388M positions would be ~2 TB; FEN→tensor costs ~50 µs; 5 workers sustain
  ~4.1k pos/s which GPU-saturates this net; and training inputs are byte-identical
  to what serving builds from FEN (eliminated the old repetition-plane
  train/serve skew).
- Loss = masked-softmax CE with label smoothing 0.05 over legal moves
  + `VALUE_LOSS_WEIGHT (=1.0)` × MSE(value). AMP autocast + GradScaler. AdamW,
  weight_decay 1e-4. LR staircase: `lr × 0.3^(epoch−1)` → 1e-3, 3e-4, 9e-5.
- Defaults: `--batch_size 1024 --workers 5` (measured steady **~4,141 pos/s**,
  ~3.2 GiB VRAM). `--channels_last` exists but is 4× slower here — leave off.
- Robustness for ~26 h epochs: `.midtrain` snapshot every `--checkpoint_minutes 30`;
  mid-epoch validation + best-save every `--val_every_minutes 120`
  (`--val_batches 120` ≈ 123k positions); atomic saves; `--resume <ckpt>`.
- **Resume semantics (important):** a partially trained epoch **counts as
  complete** — `start_epoch = ckpt.epoch + 1`; an IterableDataset stream can't seek
  back. Consequence: if a `--epochs 2` run dies during epoch 2, resuming with
  `--epochs 2` trains nothing (the trainer now prints an explicit NOTE); resume
  with `--epochs 3`. This is deliberate: skipping the tail of one shuffled pass is
  far cheaper than redoing 20 h, and the next epoch covers all data anyway.
- `run_validation` calls `N.evaluate`, which leaves the model in eval mode — the
  trainer restores `model.train()` after every mid-epoch val (a silent-degradation
  bug class; keep the restore if you touch this code).

---

## §6 Phase 1 — COMPLETE (2026-07-02 → 07-03)

What was executed, so you know what exists and can reproduce the reasoning.

### 6.1 Ingest (3 raw shards)
155,837,479 rows in → **46,102,706 unique** (45,852,706 train / 250,000 val),
26 minutes. `dropped_depth 573,798` (~0.37%), `mate_rows 14,131,889`, zero
bad-FEN/no-eval. Output: `data/distill_chunks_v4/` (23 train + 1 val shards,
1.1 GB). Validation after ingest: all sampled moves legal, Stockfish cross-check
corr 0.974 / sign 100%. **Known Phase-1 defect (fixed since, but baked into this
dataset/model): castling-best rows (~1.55%) were skipped** by the then-strict
reader — the Phase-1 model never trained on castling-as-best-move positions. The
Phase-2 re-ingest includes them.

### 6.2 Training
`ChessModelV4` 256×12 (15.1M params), 3 epochs × 46.1M, batch 1024, AMP, ~10 h
total on the 3070. Final: val_loss **1.4760**, top-1 vs Stockfish-best **52.0%**,
value MAE **0.115**, no train/val gap (data ≫ params at this scale — regularization
is a non-issue).

### 6.3 Artifacts
- `model/grandmaster_resnet_v4_distill.pt` (+`.midtrain`) — Phase-1 checkpoint.
- `model/grandmaster_resnet_v3.pt` — serving default, keep.
- `model/grandmaster_resnet_v2_resumed.pt`, `model/grandmaster_model_perspective_resnet_negatives_v2.pt`, `data/grandmaster_model_v2.pt` — legacy, keep loadable.

### 6.4 Measurements (all in ROADMAP changelog)
Raw-policy ladder → fitted **~2040**. Serving-config probes: blunder guard ON
halves the score (31% vs 50% at 2000) → guard retired for v4; **MCTS-200: 87.5%
vs 2000, 56.2% vs 2300 → fitted ~2340**. Search pays on v4 exactly as the roadmap
bet (on v3 it added nothing — the value head was the bottleneck, now it isn't).

---

## §7 Phase 2 — READY (the user runs it; you support)

Objective: retrain v4 on the **full 20-shard dump** (≈388M uniques, now including
castling rows), then bank the cheap search Elo (WS3), promote to serving, and
measure against the gate and the goal.

### 7.1 Runbook (already delivered to the user)

One command: **`.\run_phase2.ps1`** from the repo root (pins
`.venv\Scripts\python.exe`, idempotent: skips ingest if output shards exist,
auto-resumes training if the output checkpoint exists, logs to `logs/`, then runs
the 24-game gauntlet at 2000/2300/2500). Manual equivalents:

```powershell
# 0. admin PowerShell once:
powercfg /change standby-timeout-ac 0

# 1. ingest  (~3–4 h, peak disk ~18 GB transient)
.venv\Scripts\python.exe -m training.ingest_lichess_evals --num_shards 20 --out_dir data/distill_chunks_v4_full

# 2. train   (~26 h/epoch × 2)
.venv\Scripts\python.exe -m training.train_distill --data_dir data/distill_chunks_v4_full --output model/grandmaster_resnet_v4_full.pt --epochs 2

# 3. gauntlet (per rung)
.venv\Scripts\python.exe -m evaluation.vs_stockfish --model model/grandmaster_resnet_v4_full.pt --mode mcts --sims 200 --uci_elo 2300 --games 24 --no_blunder_guard
```

Interruption recovery: re-run `run_phase2.ps1` (it resumes from the best
checkpoint). If it died **during epoch 2**, the resume needs `--epochs 3` run by
hand (§5.3 resume semantics — the trainer prints a NOTE when this applies).
Resuming from `model/...pt` (last *validated* best, ≤2 h stale) is the default;
`.midtrain` is fresher (≤30 min) but unvalidated — fine to use if the best is old.

### 7.2 What healthy output looks like

**Ingest** (~scale-up of Phase-1 actuals): `rows_in` ≈ 945M; `unique_positions` ≈
370–388M; `dropped_depth` ≈ 0.3–0.5% of rows_in; `dropped_bad_castle` near 0
(hundreds, not millions); `dropped_no_line/bad_fen/no_eval` ≈ 0; output ≈ 190+
train shards, ~9–10 GB. Then validate the output before training:

```powershell
$env:DISTILL_DATA_DIR = 'data/distill_chunks_v4_full'
.venv\Scripts\python.exe tests/distill_ingest_test.py
```

**Training log signature:** startup prints shard/position counts and
`ChessModelV4 256x12 | 15,0xx,xxx params`; step lines every 200 steps with
`~4,100 pos/s` (if you see ~1,000, channels_last got enabled somehow — §11.3);
`[dataset worker] skipped N bad rows` absent or tiny; `[midtrain snapshot ...]`
every 30 min; `[mid-epoch val ...]` every 2 h with val loss ideally dipping below
Phase-1's 1.4760 during epoch 1 and top-1 pushing past 52% toward ~54–56%.

### 7.3 Expected outcome and gate

- Phase-1 model already sits at ~2340 with MCTS-200. Phase 2 adds: full data
  (+ castling competence) → expect raw policy ≳2100 and MCTS-200 ≈ 2400±.
- **Exit gate: MCTS ≥50% vs UCI_Elo 2300 over ≥24 games** — re-verify even though
  the Phase-1 checkpoint met it at 8 games.
- If the 2500 rung reads ≥50% at 24 games with ≤2 s/move settings: **the project
  goal is met** — go to §7.5 promotion, then declare done in ROADMAP.

### 7.4 WS3 search upgrades (your implementation work, in measured-impact order)

Do these *after* the retrain lands, measuring each step at the 2300/2500 rungs
(8-game smoke → 24-game confirm). All in `inference/mcts_player.py` unless noted.

1. **Tree reuse between moves.** Today every move rebuilds the tree. Keep the root's
   chosen child subtree across our move + opponent reply (in `vs_stockfish`
   game-loop and — behind a per-game cache keyed by game id — the backend).
   Effectively 2–3× sims for free. Watch: the reused subtree's priors were computed
   pre-reply; that's standard and fine.
2. **fp16 leaf evaluation.** Wrap the evaluator forward in
   `torch.amp.autocast('cuda')` (do NOT `model.half()` — BatchNorm stats want
   fp32). Expect ~1.5–2× sims/s. Verify move-for-move equivalence on ~100
   positions vs fp32 first (tiny logit diffs are fine; changed argmax on >2–3% of
   positions is not).
3. **Raise `--mcts_batch_size`** (vs_stockfish uses 16; backend default 8) toward
   32–64 alongside sims 800–1600. Measure sims/s and Elo; the 8 GB card has room
   (serving batch of 64 boards is ~nothing next to training's 1024).
4. **FPU (first-play urgency).** In `_select_child`, unvisited children currently
   read Q=0 — an optimistic draw assumption that over-explores refuted moves.
   Initialize unvisited child Q to `−(parent_Q − fpu_reduction)` in the negamax
   frame (i.e. from the child's perspective), sweep `fpu_reduction ∈ {0.2, 0.35,
   0.5}`.
5. **Retune `policy_temperature`.** The 1.5 softening was a crutch for the peaky
   imitation policy. The distilled policy wants ~1.0–1.15. Sweep {1.0, 1.1, 1.25}
   at fixed sims vs 2300. Cheap, possibly worth 30+ Elo.
6. Stretch: in-tree repetition/50-move awareness (the `stack=16` board copies make
   `board.can_claim_threefold_repetition()` partially visible in-tree) — this is
   the structural fix for the known **repetition blindness** (raw policy shuffles
   into threefolds in drawn-ish positions; encoding rep planes are always 0 at
   serving, §4.1). MCTS already mitigates most of it.
7. Stretch: Gumbel root selection (better low-sims behavior) — only if cheap wins
   above stall short of 2500.

### 7.5 Serving promotion checklist (after gauntlet vets the new model)

1. `MODEL_PATH=model/grandmaster_resnet_v4_full.pt` for the backend process
   (loader dispatches by checkpoint metadata — no code change needed).
2. Blunder guard OFF for v4 (guard defaults live in `backend/app.py`; v3 keeps it).
3. Default MCTS on for the "Magnus 2500" mode with a sims/time slider
   (`use_mcts`/`mcts_simulations` request params exist; keep the server-side cap).
4. Keep v3 servable as the "human-style" mode (dual-generation dispatch already
   works). 5. Update README + ROADMAP changelog; commit.

---

## §8 Phase 3 — CONTINGENT (only if Phase 2 + WS3 land short of 2500)

### 8.1 Decision tree, cheapest first

1. **Sims/time budget:** the definition of done allows ~2 s/move. If 2500 is close,
   800→1600 sims (with fp16 + tree reuse) may close it alone (Lc0-family scaling
   ≈ +50–100 Elo per doubling in this regime — measure, don't assume).
2. **Third epoch** on the full dump (`--resume … --epochs 3`, LR 9e-5) if the
   epoch-2 val curve was still falling.
3. **Bigger tower:** 256×16 or 320×12 (VRAM check first; width beats depth for
   VRAM at equal params here; ~35–40% slower per epoch — days, not weeks).
   Distillation remains the cheapest Elo (§13.1) — exhaust it before RL.
4. **WS4 self-play RL** (below) as the final gap-closer.
5. If stalled at ~2400 after all that: raise Stockfish's per-move time asymmetry
   honestly in the report, or renegotiate the definition of done with the user —
   do not quietly move goalposts.

### 8.2 WS4 self-play, if it comes to that

The loop (`experiments/train_self_play.py`) is audited/gated/anchored but
**sequential** (~30–60 s/game) — useless throughput for RL. Before any serious run:
- **Evaluator-server parallelism:** G=32–64 concurrent games (async/thread), leaf
  evals pooled into batched GPU calls. Pure-Python is enough for 10–20×.
- Bootstrap from the Phase-2 v4; **anchor with distilled chunks, not human ones**
  (the anchor-guard raises if anchor dirs contribute 0 positions — that's the
  encoding-mismatch tripwire, don't bypass it except with explicit
  `--supervised_dirs ''`).
- Known small fixes to fold in: honor `sample_weight` in the anchor mix; verify the
  self-play MCTS board-copy history depth suffices for repetition awareness.
- Keep the existing promotion gate (new net must beat old in arena before replacing).

---

## §9 Measurement protocol (how strength claims are made here)

### 9.1 The gauntlet

`evaluation/vs_stockfish.py` — colors alternate every game (even game index =
model plays White); `--max_plies 300` then adjudicated draw; an illegal/None model
move **loses on the spot**; `claim_draw=True` (threefold counts).

- **Absolute Elo: always `--uci_elo N`** (calibrated `UCI_LimitStrength`;
  clamped 1320–3190). `--skill` exists only to mirror the browser opponent.
- `--movetime 0.6` = the standing convention (browser parity). All historical
  numbers are at 600 ms — never compare across different movetimes.
- `--mode policy` reproduces backend policy serving (value rerank weight 2.0,
  rerank-all; guard on unless `--no_blunder_guard` — **always pass
  `--no_blunder_guard` for v4**). `--mode mcts --sims N` reproduces the MCTS
  toggle (`--mcts_policy_temp 1.5` today; will change with WS3.5).

### 9.2 Sample-size discipline

6–8 games/rung = smoke test (±35% swings are normal); use for direction only.
Gate/goal claims: **≥24 paired games**. Single-rung percentages are noisy —
prefer the whole-ladder fit (the Phase-1 "37.5% vs 1900 but 50% vs 2300"
inversion was pure noise; the fit said 2040 and later rungs agreed).

### 9.3 Fitting Elo from ladder results (no scipy on this machine)

```python
import math
def fit_elo(rungs, lo=1000, hi=3200):
    """rungs = [(opponent_elo, points_scored, games_played), ...]; draws = 0.5 pt."""
    def nll(r):
        s = 0.0
        for opp, pts, n in rungs:
            e = min(max(1 / (1 + 10 ** ((opp - r) / 400)), 1e-9), 1 - 1e-9)
            s -= pts * math.log(e) + (n - pts) * math.log(1 - e)
        return s
    return min(range(lo, hi + 1), key=nll)

# Phase-1 raw policy: fit_elo([(1700,6,6), (1900,3,8), (2000,3,6), (2300,3,6)]) ≈ 2040
```

### 9.4 Reporting

Every measurement session → a dated bullet in `docs/ROADMAP_2500.md` §9 with:
config (checkpoint, mode, sims, guard, movetime), per-rung W-D-L, fit. That
changelog is how three days of work stayed coherent; keep it that way.

---

## §10 Test suite reference

**No pytest.** Every test file is a standalone script:
`.venv\Scripts\python.exe tests\<file>.py` — asserts hard, prints a final
"...passed" line. Run the relevant set after touching anything they cover; run all
of them before any training launch.

| File | Proves | Notes |
|---|---|---|
| `audit_encoding_orientation_test.py` | FEN↔tensor lossless round-trip; rot180+color-swap invariance; conv-head spatial alignment; end-to-end model orientation | run after ANY encoding/head change |
| `audit_labels_data_test.py` | legality of stored policy labels on real chunks; collator parity; full-chunk stream | legacy-chunk oriented |
| `audit_value_semantics_test.py` | STM sign probes; mate/stalemate terminal values; MCTS Q→+1.0 on mate-in-1 | pins §4.5 |
| `audit_overfit_tiny_test.py` | tiny-batch overfit; initial loss == uniform-over-legal entropy | catches dead gradients/masking bugs |
| `audit_checkpoint_loading_test.py` | strict loads; arch-aware rebuild incl. v4; partial-checkpoint rejection; default-path-is-v3; evaluate_model wiring | run after loader/schema changes |
| `v4_model_test.py` | v4 shapes; checkpoint round-trip through dispatch (bit-identical); v3 dispatch unaffected; tiny overfit | |
| `distill_ingest_test.py` | ingested labels legal via strict trainer path; values in range, mates at ±1; **independent Stockfish sign cross-check**; raw-shard mate probe (skips if no raw on disk); end-to-end ingester unit test (castling normalization, POV negation, dedupe-by-depth, temp cleanup); 200k-row reader sweep | **`DISTILL_DATA_DIR` env** selects the dataset (default `data/distill_chunks_v4`) — point it at the Phase-2 output after ingest |
| `blunder_guard_test.py`, `v3_encoding_test.py`, `fen_to_tensor_test.py`, `index_to_move_test.py`, `move_to_vector_test.py`, `move_sequence_to_vector_test.py`, `policy_move_encoding_test.py`, `preprocess_bad_move_test.py`, `value_head_test.py` | legacy/v2-v3 era coverage | keep green; rarely need changes |

All of the above were green on 2026-07-05.

---

## §11 Debugging playbook (symptom → cause → fix)

Every entry below actually happened on this machine.

1. **Segfault / exit 139 / 0xC0000005 on first parquet read** → torch imported
   before pyarrow → put `import pyarrow.parquet` above any torch import in that
   process (§2.3.1). DataLoader spawn workers re-import the module, so module-top
   ordering covers them.
2. **`ModuleNotFoundError: torch/pyarrow` from `python …`** → bare `python` is the
   empty C:\Python314 → use `.venv\Scripts\python.exe` (§2.2).
3. **Training at ~1,000 pos/s instead of ~4,100** → channels_last on → remove the
   flag; it's measured 4× slower on this GPU/workload.
4. **Loss falls nicely but play is bad** → probably NOT a code bug — the audits
   cleared the pipeline; think approach/serving-config. Confirm with the gauntlet
   and the diagnose_* tools (per-phase cp loss) before touching training code.
5. **Redirected log ends mid-line / no traceback at the end but the process died**
   → stdout block-buffering; traceback (stderr) interleaved earlier → grep the
   whole log for `Traceback|Error|error`; don't trust `tail`; also `$?` in bash
   reflects the last pipe stage (e.g. `tail`), not python — check
   `$LASTEXITCODE`/`echo ${PIPESTATUS[0]}` as appropriate.
6. **Long job vanished when the terminal/session recycled** → child process tied to
   session → detached `Start-Process` with log redirects, or the user runs it in
   their own terminal (preferred, §1.4).
7. **Machine slept mid-run** → `powercfg /change standby-timeout-ac 0` first.
8. **Ingest dedupe thrashes RAM / MemoryError** → too few buckets for the dump
   size → `--buckets 64` (or more; per-bucket dict must fit ~2 GB).
9. **Disk fills during ingest** → check per-bucket temp deletion is intact, raws
   being deleted (no `--keep_raw`), and that you started with ≥25 GB free;
   `Get-PSDrive C` to monitor.
10. **"refusing to serve a partially initialized network"** → checkpoint/arch
    mismatch or truncated file → inspect checkpoint keys (`arch_version`,
    `residual_filters/blocks`, `board_encoding`); saves are atomic so truncation
    is unlikely; most often MODEL_PATH points at the wrong file.
11. **Value head seems inverted (model courts losses)** → sign-convention
    regression → run `distill_ingest_test.py` (Stockfish cross-check) +
    `audit_value_semantics_test.py`; remember the RAW dataset is White-POV — only
    ingested output is STM.
12. **`[dataset worker] skipped N bad rows` is large** → a strict-parse path saw
    king-takes-rook castling UCI → that path must use `board.parse_uci`; see §4.7.
    (After the fix: N ≈ 0.)
13. **MCTS no better than raw policy** → on v3 that was *correct behavior* (noisy
    value head). On v4 search demonstrably pays; if it stops paying after a change,
    suspect your search change (verify with `audit_value_semantics_test.py`'s
    mate-probe and an 8-game smoke at 2000).
14. **Gauntlet numbers look contradictory across rungs** → small-sample noise →
    whole-ladder fit (§9.3), more games (§9.2). Do not chase single-rung swings.
15. **Threefold draws from clearly winning positions (raw policy)** → known
    repetition blindness (§4.1, §7.4.6) — not a new bug. MCTS mode mitigates.
16. **HF shard download died / corrupt parquet** → the "already present" check
    only requires size >1 MB, so a partial file can be trusted on re-run → delete
    the partial `data/lichess_evals_raw/data_XXXX.parquet` and re-run (curl
    `--fail` covers HTTP errors but not mid-transfer truncation).
17. **Backend serves the wrong generation** → `MODEL_PATH` env of the *backend
    process* decides; the startup banner and health endpoint print the loaded
    file/encoding — check those first.

---

## §12 Do's and don'ts

**Do**
- Use `.venv\Scripts\python.exe` for everything.
- Import pyarrow before torch in any new parquet-touching script.
- Run the gauntlet after every substantive change; log to ROADMAP changelog.
- Keep every print in long-running code `flush=True`.
- Check `Get-PSDrive C` (disk) and `Get-Process python*` (running jobs) before
  heavy operations.
- Keep train/val hash-disjoint (val = bucket 0 only).
- Keep checkpoint saves atomic and metadata-complete.
- Re-verify label conventions empirically for ANY new data source.
- Prefer 24+ paired games for claims; whole-ladder fits over single rungs.
- Keep v2/v3 checkpoints loadable (product feature + regression safety).

**Don't**
- Don't enable `channels_last` (4× slower here) or blindly trust generic GPU folklore
  — this is an 8×8 workload on one specific card; measure.
- Don't use the blunder guard with v4 (measured −19 points at Elo-2000).
- Don't validate UCI legality with `Move.from_uci(x) in board.legal_moves` (§4.7).
- Don't start >1 h jobs without user sign-off; don't kill user processes.
- Don't trust `tail` of a redirected log for crash diagnosis.
- Don't compare Elo numbers taken at different movetimes or vs `--skill` rungs.
- Don't "fix" the scalar value head into WDL as a drive-by (§13.3 — it's a
  considered decision; revisit only with measurements in hand).
- Don't let a resumed run silently no-op: watch for the trainer's "nothing left to
  train" NOTE (resume counts a partial epoch as complete).
- Don't write new data into a non-empty ingest out_dir (the tool refuses; don't
  bypass by deleting its guard).

---

## §13 Decision log (why things are the way they are)

1. **Distillation-first, self-play-last.** Single 8 GB GPU: sequential self-play
   generates ~1–2k positions/hour vs 388M engine-labeled positions free from
   Lichess. DeepMind (arXiv 2402.04494) showed 9M params → ~2000 Elo searchless on
   SF distillation. Outcome so far: +370 Elo in one overnight run — the bet paid.
   RL is a gap-closer, not the engine (§8).
2. **FEN-only storage, on-the-fly encoding.** ~2 TB avoided; ~50 µs/encode with
   5 workers saturates this GPU; byte-identical train/serve inputs; reuses the
   audit-proven codecs untouched (§5.3).
3. **v4 kept the scalar tanh value head** (ROADMAP WS2 originally sketched a WDL
   softmax). Rationale: every consumer (value rerank, MCTS negamax backup, arena,
   terminal handling) speaks scalar [−1,1]; MSE on tanh(cp/600) targets measured
   well (MAE 0.115) and search now pays — the WDL migration's calibration upside
   didn't justify touching audited plumbing mid-plan. Revisit only if Phase-3
   evidence points at value calibration as the limiter.
4. **Blunder guard retired for v4** — measured 31% (on) vs 50% (off) at Elo-2000:
   a depth-2 PST search now vetoes moves better than its own understanding. Keep
   for v3 serving.
5. **channels_last off** — measured 1,033 vs 4,141 pos/s (§2.3.2).
6. **64 dedupe buckets** — RAM math (§5.2); **val from bucket 0** — hash-disjoint
   split by construction.
7. **VALUE_LOSS_WEIGHT 1.0 for distillation** (vs 0.25 legacy): dense engine evals
   are trustworthy labels; noisy game outcomes were not.
8. **Staircase LR (×0.3/epoch) over cosine:** resume-friendly (epoch-granular
   resume, §5.3) and 2–3 total passes don't reward schedule finesse.
9. **UCI_Elo (not Skill Level) for absolute claims** — Skill is uncalibrated; the
   600 ms movetime convention mirrors the browser product experience.
10. **MCTS `policy_temperature 1.5`** was tuned for the peaky imitation policy; a
    scheduled WS3 retune (~1.0–1.15) is expected for the distilled policy — it is
    on the books deliberately, not an oversight (§7.4.5).
11. **Strict checkpoint loading with the one legacy exception** — a partially
    initialized net once played "plausibly bad", which is the worst failure mode;
    fail loudly instead (§4.4).

---

## §14 Open items at handoff

1. **User executes Phase 2** (`run_phase2.ps1`). Your role: standby, then validate
   `ingest_stats.json` + run `distill_ingest_test.py` with
   `DISTILL_DATA_DIR=data/distill_chunks_v4_full` before the training step if
   asked, and interpret the gauntlet.
2. **WS3 search upgrades** (§7.4) — the main implementation work left, ~160 Elo to
   find between retrain + search.
3. **Serving promotion of v4** (§7.5) after vetting; keep dual modes.
4. **Commit the working tree** on `redo_selfplay` — repeatedly offered, still
   pending; raise before more work piles up. (PR target branch: `main`.)
5. **Re-verify the Phase-2 gate at ≥24 games** with the Phase-2 model (met at 8
   games by Phase-1).
6. Minor: the noisy 1900-rung raw-policy reading (37.5%) is worth a clean 24-game
   re-measure sometime during Phase-2 tuning; `tests/distill_ingest_test.py`'s
   Stockfish cross-check thresholds (corr>0.75) have headroom if the full dump
   shifts the depth mix.
7. If Phase 2 ends ≥50% vs 2500 at 24+ games: update ROADMAP changelog, promote,
   declare done, and celebrate with the user — then §8 never runs.
