# Historical handoff: 2500 benchmark project

> Superseded on 2026-08-11. This filename is retained so older links do not
> break. Do not use the former Phase 2 instructions as the current operating
> runbook.

The original version of this document was a July 2026 point-in-time handoff
written before the full v4 distillation run completed. It mixed durable design
notes with machine-specific instructions, personal workflow guidance, and stale
open items. Those concerns made it unsafe as a cold-start guide after Phase 2.

Use the maintained documents instead:

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — what is trained, measured, deployed,
  and still unverified.
- [`MODEL_CARD.md`](MODEL_CARD.md) — v4 provenance, intended use, benchmark
  boundaries, and known limitations.
- [`ROADMAP_2500.md`](ROADMAP_2500.md) — historical plan and dated experiment
  log. Its Elo numbers are harness-specific estimates, not human tournament
  ratings.
- [`RAILWAY_DEPLOY.md`](RAILWAY_DEPLOY.md) — current serving rollout and rollback
  checklist.
- [`../README.md`](../README.md) — supported local entry points and repository
  map.

## Historical context

The July handoff recorded the transition from a v3 human-game imitation model
to a v4 engine-distilled model. It described a planned 20-shard Lichess
evaluation ingest, two training epochs, and a Stockfish gauntlet. That work was
subsequently reported complete in the roadmap on 2026-08-11.

The former handoff's local paths, free-disk estimates, Python installation
details, process-management advice, and “Phase 2 READY” status were properties
of one workstation at one point in time. They are intentionally not preserved
as project-wide requirements.

## Durable technical invariants

These remain useful review checkpoints, but tests provide evidence only for the
cases they exercise; they do not prove the pipeline bug-free.

- Model values use side-to-move perspective.
- v2 uses the 17-plane `perspective_v2` encoding and move history.
- v3 and v4 use the 20-plane `perspective_v3` tensor schema; v4 is
  distinguished by checkpoint `arch_version`, not by encoding alone. Current
  v4 masks unsupervised clock/repetition channels 17-19 and therefore uses 17
  effective input planes.
- Policy logits are masked to legal moves before training or move selection.
- Lichess evaluation values are converted from the source convention to
  side-to-move convention during ingestion.
- Checkpoint loading should reject architecture mismatches rather than serving
  randomly initialized layers.
- The v4 benchmark configuration disables the legacy blunder guard and uses
  MCTS. This does not imply that every production MCTS budget has the same
  measured strength.

Any future handoff should be generated from the current status page and linked
artifacts rather than reviving the old workstation-specific instructions.
