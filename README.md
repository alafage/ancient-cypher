# Ancient Cypher

Benchmark Copilot vision models on historical French scans using the Copilot SDK.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and ensure the Copilot CLI is authenticated and your account has access to the models you want to test. Sync the locked dependencies:

```sh
uv sync --locked
```

The project requires Python 3.14 or newer; uv can install a suitable Python version if needed.

## Samples

Place one image (`.png`, `.jpg`, `.jpeg`, `.webp`, or `.gif`) and its UTF-8 reference transcription named `transcript.txt` in each sample directory under `data/`. Nested directories are supported.

## Run

List available vision-capable models:

```sh
uv run --locked main.py --list-models
```

Benchmark all available vision-capable models:

```sh
uv run --locked main.py
```

Select specific models by repeating `--model`:

```sh
uv run --locked main.py --model MODEL_ID --model OTHER_MODEL_ID
```

Use `--data-dir` to select another sample directory, `--timeout` to set the per-request timeout in seconds, and `--output` to choose a new results file. See `uv run --locked main.py --help` for all options.

Results are saved as JSON under `results/` by default, including each transcription, errors, elapsed time, and aggregate character and word error rates for successful samples. Scoring normalizes Unicode to NFC and collapses whitespace while preserving case and punctuation. Failed samples are recorded separately and result in a nonzero exit code.
