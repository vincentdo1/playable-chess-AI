# Model card: v4 engine-distilled chess model

Status: pre-release documentation

This card describes the checkpoint referred to in project notes as
`grandmaster_resnet_v4_full.pt`. The checkpoint is not stored in Git, and the
repository does not yet identify a public model repository, immutable revision,
or file digest. Fill the release record below with observed values before
publishing or promoting the model.

## Model summary

| Field | Value |
|---|---|
| Model family | Chess policy/value network |
| Architecture | 256-filter, 12-block squeeze-and-excitation ResNet (`ChessModelV4`) |
| Reported parameter count | Approximately 15.1 million |
| Input | `perspective_v3` 20-plane tensor, side-to-move perspective; v4 masks unsupervised halfmove/repetition channels 17-19 (17 effective planes) |
| Policy output | 4,864 move logits (64 from-squares × 76 move-type planes), masked to legal moves |
| Value output | Scalar in [-1, 1] from the side-to-move perspective |
| Training objective | Stockfish best-move policy distillation plus scalar evaluation regression |
| Search used in strongest recorded result | MCTS, 200 simulations, policy temperature 1.5 |

v4 uses the same board-encoding label as v3. Loaders must use checkpoint
`arch_version='v4'` to distinguish the architectures.

## Intended use

- Research and educational experimentation with chess policy/value networks.
- Playing chess through this repository's backend with legal-move masking.
- Comparing supervised human-game imitation with engine distillation.
- Bootstrapping search or future chess-training experiments after independently
  validating the checkpoint and configuration.

## Uses that are not established

- A claim of a FIDE, online-platform, or human tournament rating.
- A simulation of Magnus Carlsen's style or decision-making.
- An official Magnus Carlsen product or an endorsed use of his identity.
- A drop-in chess authority for high-stakes cheating detection, player
  assessment, or competition adjudication.
- Safe deserialization of an untrusted checkpoint. Only load artifacts from a
  controlled source whose identity and digest have been verified.

The browser uses the neutral **Neural Network** label. The `magnus` API value
and `magnus_carlsen` desktop value remain as legacy compatibility identifiers.
v4 is trained from engine-evaluation data and should be described as
engine-distilled. This project is not affiliated with or endorsed by Magnus
Carlsen.

## Training data

The training pipeline reads the Hugging Face mirror of the Lichess chess
position evaluations dataset. Each source row contains a FEN, principal
variation, search depth, and centipawn or mate evaluation. Ingestion:

1. filters records below the selected depth threshold;
2. converts the first principal-variation move into the policy target;
3. converts source evaluations to side-to-move value targets;
4. deduplicates exact FEN strings, retaining the deepest record; and
5. creates a hash-disjoint validation bucket.

The source FENs used for v4 contain only four fields. They do not carry the
halfmove clock or game history needed to supervise the three auxiliary
`perspective_v3` clock/repetition planes. Current `ChessModelV4` therefore zeros
channels 17-19 at its boundary. This preserves the shared 20-plane tensor
contract while limiting the model to the 17 planes that received training
signal.

Project notes report a full run over 20 source shards, producing approximately
393.2 million unique FEN records, with about 392.97 million used for training
and 250,000 for validation. Those counts are historical observations, not an
immutable dataset identity. The source URL previously followed a mutable
`main` revision, and the exact upstream revision and input digests used for the
reported checkpoint are not recorded in Git.

The upstream dataset is described in existing project notes as CC0. Verify the
terms and provenance of the exact snapshot before reuse. This repository does
not currently declare a license for the model weights; this card does not
select one.

## Training procedure

The recorded full run used the v4 distillation trainer for two epochs with
AdamW, automatic mixed precision, legal-move masking, label smoothing, and a
scalar value loss. The roadmap reports a learning-rate staircase from 1e-3 to
3e-4 and throughput around 4,190 positions/second on an RTX 3070.

The historical checkpoint schema recorded the model and optimizer state,
encoding, architecture version, width/depth, value-loss weight, epoch, and
selected validation metrics. It did not capture every release-provenance field,
such as the source-data revision, all input hashes, package lock, full training
arguments, RNG state, and training code commit. Treat those as release gaps for
the recorded v4-full model unless a separately published manifest supplies
them.

The current pipeline improves future runs: ingestion requires an immutable
Hugging Face commit, validates downloaded shards, and writes a completed
`ingest_manifest.json` with source and output identities. The trainer verifies
that corpus and stores its provenance digest, objective/configuration,
optimizer/scaler/RNG state, seed, and epoch-completion state in checkpoints.
Mid-epoch resume replays and skips completed batches rather than silently
counting a partial epoch as complete. These changes do not retroactively
identify the data or environment used for the historical checkpoint described
here.

## Evaluation

### Recorded offline validation

Project notes report final validation loss 1.3826 and top-1 policy accuracy
54.0% on the generated validation split. The split is disjoint by exact FEN
hash, but positions from related games or analyses may still be correlated.
The dataset artifact and raw evaluation log are not committed, so these values
are not independently reproducible from Git alone.

### Recorded Stockfish harness result

| Field | Recorded configuration |
|---|---|
| Checkpoint | `grandmaster_resnet_v4_full.pt` |
| Model input path | Predecessor behavior; clock/repetition channels 17-19 were not masked |
| Model move selection | MCTS-200, batch size 16, policy temperature 1.5, blunder guard off |
| Opponent | Stockfish 18 with `UCI_LimitStrength=true`, `UCI_Elo=2500` |
| Opponent time | 600 ms per move |
| Games | 24, alternating model color |
| Result | 12 wins, 9 draws, 3 losses |
| Point estimate | 68.8% score |

This result establishes only a point estimate under the predecessor harness
used for that run. It started each game from the standard initial position and
relied on Stockfish limited-strength behavior for variation. It did not preserve
paired-opening PGNs, a confidence interval, immutable run metadata, or
per-model-move latency. A rough interval around 24 games is wide enough that
the result should not be converted into a precise general Elo claim.

The run also predates the current v4 input-plane mask. Its inference path could
activate channels 17-19 from a live position even though those weights received
no corresponding training signal. Current masked inference is a behavior
change and has not been rerun against Stockfish, so the 12W-9D-3L result cannot
be used as validation of the corrected path.

The current evaluator is stronger than the historical protocol: by default it
uses 16 openings with both colors (32 games), a fixed seed, per-opening-pair
confidence units, artifact and environment metadata, model latency percentiles,
and JSON plus PGN output. It can fail closed on a required score lower bound and
p95 latency. The v4 model has not yet been rerun with that protocol, so none of
those newer safeguards can be retroactively attributed to the 12W-9D-3L result.

Stockfish's `UCI_Elo` option configures a limited-strength engine opponent; it
does not certify a human rating for the model. The model's recorded 68.8% score
should therefore be reported verbatim with its harness settings, not as
“2640 Elo” or an unqualified “2500 Elo model.”

The original project target also included an approximately two-second model
move budget on the local GPU. Whole-game timings in the historical run do not
isolate model latency, so the tracked result does not demonstrate that part of
the target. A new run can enforce it with `--require_p95_seconds 2.0`.

## Known limitations

- **Statistical uncertainty:** 24 games are insufficient for a precise strength
  estimate, and no confidence-bound acceptance criterion was applied.
- **Opening coverage of the recorded result:** all historical Stockfish games
  start from the initial position; nine draws exceed the roadmap's threshold
  for adopting paired openings. The current harness fixes the protocol, but a
  v4 rerun is pending.
- **Artifact provenance:** the model location, immutable revision, byte count,
  and digest are not recorded here yet.
- **Data reproducibility:** the historical training source was mutable and the
  exact source snapshot is not identified in the repository.
- **Clock/history blindness:** current v4 deliberately ignores halfmove-clock
  and repetition planes because its training corpus did not supervise them. It
  cannot condition moves on those draw-rule signals, even when a live board
  supplies them.
- **Style mismatch:** engine distillation does not preserve Magnus- or
  grandmaster-imitation style.
- **Search dependence:** the strongest recorded result uses MCTS-200. Raw
  policy, low-simulation CPU serving, and other time caps have different and
  currently unreported strength. The strongest result also predates the v4
  input mask, so current masked MCTS strength is unreported.
- **Deployment constraints:** production CPU latency, concurrency behavior,
  cold-start time, and memory have not been captured in a versioned benchmark.
- **Evaluation transfer:** results against one Stockfish version/configuration
  do not automatically transfer to other engines, hardware, time controls, or
  human opponents.

## Release record

Complete this table with actual release values. Do not substitute branch names
such as `main` where an immutable revision is required.

| Field | Release value |
|---|---|
| Model repository | Not recorded |
| Immutable model revision | Not recorded |
| Checkpoint filename | `grandmaster_resnet_v4_full.pt` |
| Exact byte count | Not recorded |
| SHA-256 | Not recorded |
| Training code commit | Not recorded |
| Source-dataset revision | Not recorded |
| Ingest manifest/digest | Not recorded |
| Runtime/package lock | Not recorded |
| Stockfish binary digest used for release benchmark | Not recorded |
| Benchmark result/PGN artifact | Not recorded |
| Model-weight license | Not specified |

## Release and maintenance expectations

- Publish weights under a versioned, immutable revision and verify the digest
  before loading them in production.
- Keep model, data, code, and evaluation identities together in one release
  manifest.
- Preserve the previous known-good artifact and complete serving configuration
  for rollback.
- Update this card when the artifact, training data, serving configuration, or
  benchmark changes. Do not silently replace a file under an existing release
  identity.
- Report current operational status in [`PROJECT_STATUS.md`](PROJECT_STATUS.md),
  not in this model card.
