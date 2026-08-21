# Contributing

Keep pull requests focused and explain what changed. Include the commands you
used to test the change.

Run the default test suite with:

```bash
python -m pip install --requirement requirements-test.txt
python -m pytest -q
```

Model, dataset, and Stockfish-dependent tests may need local files that are not
stored in Git. When those files are required, note which checkpoint, dataset,
or engine version you used.

This repository does not currently include a license. Do not assume permission
to redistribute its code or model weights.
