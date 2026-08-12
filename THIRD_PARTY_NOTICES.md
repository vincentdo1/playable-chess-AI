# Third-party notices and unresolved provenance

This file is an inventory, not a license for this repository's original code
or model weights. The project owner must choose those licenses separately.

## Stockfish JavaScript/WebAssembly

`stockfish.js` identifies itself as a build derived from Stockfish and
`niklasf/stockfish.js`, released under GNU GPL v3. Preserve its embedded
copyright header. Before distributing a release, identify the exact upstream
source revision/build, include the applicable GPL license text, and make the
corresponding source available as required by that license.

- Upstream named in the file: https://github.com/niklasf/stockfish.js
- Stockfish: https://github.com/official-stockfish/Stockfish

The repository does not currently record the exact revision or build recipe
for the checked-in generated file, so this inventory does not establish full
license compliance.

## Lichess position evaluations

The v4 ingestion pipeline reads
`Lichess/chess-position-evaluations` from Hugging Face. The dataset page labels
the dataset CC0. Every reproducible run must retain the exact dataset revision
and source-shard hashes; historical v4 training did not do so.

- Dataset: https://huggingface.co/datasets/Lichess/chess-position-evaluations
- Lichess database exports: https://database.lichess.org/

## Visual assets

The repository contains chess-piece images and generated network media without
a committed source/license manifest. Record their authors, source revisions,
and redistribution terms before treating a public release as provenance
complete.
