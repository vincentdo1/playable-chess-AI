# v4 model card

This card describes `grandmaster_resnet_v4_full.pt`. The checkpoint is not
included in this repository, and no public model revision or checksum is
currently recorded.

## Model

| Field | Value |
| --- | --- |
| Task | Chess policy and value prediction |
| Architecture | 256-filter, 12-block SE-ResNet (`ChessModelV4`) |
| Parameters | Approximately 15.1 million |
| Input | Side-to-move `perspective_v3` board tensor |
| Policy output | 4,864 move logits, masked to legal moves |
| Value output | Scalar in [-1, 1] from the side-to-move perspective |
| Training signal | Stockfish move and evaluation distillation |

v3 and v4 use the same board-encoding label. Checkpoints need
`arch_version='v4'` so the loader can select the correct architecture.

The v4 training records contain four-field FENs and do not include the halfmove
clock or repetition history. The model therefore zeros input channels 17–19 and
effectively uses 17 of the 20 input planes.

## Intended use

- Chess research and educational experiments.
- Playing through this repository's backend.
- Comparing human-game imitation with engine distillation.

The model is not a FIDE or online rating estimate, a cheating detector, or a
simulation of Magnus Carlsen. The `magnus` API value and `magnus_carlsen`
desktop value are old compatibility names. This project is not affiliated with
Magnus Carlsen.

PyTorch checkpoints can execute unsafe pickle payloads when loaded without
restricted settings. Only use checkpoints from a source you trust and verify
their checksum.

## Data and training

The v4 pipeline ingests the
[Lichess chess-position-evaluations dataset](https://huggingface.co/datasets/Lichess/chess-position-evaluations).
It uses the first principal-variation move as the policy target, converts the
engine evaluation to a side-to-move value target, deduplicates exact FENs, and
keeps a hash-disjoint validation set.

Project notes report a two-epoch run over 20 shards: about 393.2 million unique
positions, 250,000 validation positions, AdamW, mixed precision, and final
validation loss 1.3826 with 54.0% top-1 policy accuracy. The exact source
revision, input hashes, training code commit, environment, and raw logs were not
recorded, so that historical run is not reproducible from Git alone.

New ingests require a pinned Hugging Face commit and write an
`ingest_manifest.json`. New checkpoints record the manifest digest, training
configuration, optimizer/scaler/RNG state, seed, and completion state. Those
changes do not recover the missing provenance of the historical checkpoint.

## Recorded evaluation

The strongest recorded v4 run used the older, unmasked input path:

| Setting | Value |
| --- | --- |
| Model | `grandmaster_resnet_v4_full.pt` |
| Search | MCTS, 200 simulations, batch size 16 |
| Opponent | Stockfish 18, `UCI_LimitStrength=true`, `UCI_Elo=2500` |
| Opponent time | 600 ms per move |
| Games | 24 from the initial position, alternating colors |
| Result | 12 wins, 9 draws, 3 losses (68.8%) |

This is a result for one harness configuration, not a general 2500 Elo claim.
It has no paired openings, confidence interval, preserved PGNs, or isolated
model-latency measurement. It also predates the channel 17–19 mask, so it does
not validate the current inference path.

`evaluation.vs_stockfish` now supports paired openings, fixed seeds,
confidence intervals, latency measurements, and JSON/PGN output. No result for
the current masked model has been recorded with that protocol.

## Known limitations

- Checkpoint location, checksum, weight license, and exact training data
  revision are not recorded.
- Current v4 cannot use halfmove-clock or repetition state.
- Strength depends on the search budget, engine settings, openings, and
  hardware.
- The best recorded result uses MCTS; raw-policy and CPU-serving strength are
  not established.
- Production cold-start time, memory, latency, and concurrency have not been
  published.
