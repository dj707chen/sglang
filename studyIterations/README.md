# studyIterations

## Ask

Add a README.md under /Users/WChen/AI/sglangTry/sglang/studyIterations to document that /Users/WChen/AI/sglangTry/sglang/python/setup_env.sh needs to be run first.

Later (2026-08-22): document getting `localRun0814_A` running again — the stale venv and model paths, why HuggingFace downloads fail on this network, and the ModelScope + corporate-CA workaround.

## Overview

Scratch space for local SGLang study runs on this machine. Each `run*/`
subdirectory is one self-contained experiment (scripts, notes, plan, generated
artifacts). Nothing here is part of the SGLang package — it is untracked
working material and is not imported by anything under `python/sglang/`.

## Run `setup_env.sh` first

**Before anything in this directory will work, set up the Python environment:**

```bash
# from the repo root
cd python
./setup_env.sh
source ../.venv/bin/activate
```

`python/setup_env.sh` creates `<repo>/.venv` and installs SGLang into it as an
**editable** install, using the pyproject that matches this machine. On Apple
Silicon it auto-detects the `mps` variant, builds the venv on Python 3.12, and
installs the `all_mps` extra (MLX + `torch` MPS build).

You cannot skip it and just `pip install -e python`: `python/pyproject.toml` is
the CUDA/Linux build and does not resolve on macOS, and Homebrew's default
`python3` is too new for the platform pins. The script swaps in
`pyproject_other.toml` for the duration of the install and restores it
afterwards, so `git status` stays clean. Full rationale:
[SETUP_ENV.md](../python/SETUP_ENV.md).

Useful variations:

```bash
./python/setup_env.sh --extras dev_mps   # also install test dependencies
./python/setup_env.sh --variant empty    # pure-Python subset, no torch
./python/setup_env.sh --recreate         # rebuild the venv from scratch
./python/setup_env.sh --help
```

Editable matters: the run scripts exercise the code in this checkout, so edits
under `python/sglang/` take effect on the next server start with no reinstall.

## Pointing a run at the venv

The run scripts each resolve their own interpreter:

| Run | Defaults to | Overridable |
| --- | --- | --- |
| [localRun0813/config.sh](localRun0813/config.sh) | `localRun0813/.venv` | yes — `export VENV=<repo>/.venv` before calling `start.sh` |
| [localRun0814_A/bin/env.sh](localRun0814_A/bin/env.sh) | `<repo>/.venv` | yes — `export SGLANG_STUDY_VENV=<path>` |

`localRun0814_A` used to expect `studyIterations/venvs/mps-py312` and needed a symlink;
it now points at `<repo>/.venv` directly, so `setup_env.sh` is all the venv
setup it needs (weights are separate — see below).

`localRun0813` still wants its own venv, which does not exist in a fresh
checkout. Use the override instead — `<repo>/.venv` satisfies it, and its
default `STUDY_MODEL_PATH` is the same `models/Qwen3-0.6B` that
`fetch_model.sh` populates, so nothing else is needed:

```bash
VENV="$(git rev-parse --show-toplevel)/.venv" ./localRun0813/start.sh
```

If a run ever dies with `ModuleNotFoundError` for something you know is
installed, check which interpreter it actually resolved — `python/.venv` is a
stale Python 3.11 leftover that is easy to land on by accident.

## Model weights

Runs read weights from `studyIterations/models/<model>` as a local directory, not a
Hugging Face hub id. `models/` is gitignored and not created by `setup_env.sh`,
so you have to fetch them.

```bash
./localRun0814_A/bin/fetch_model.sh          # Qwen/Qwen3-0.6B bf16 from ModelScope, ~1.4 GB
```

It is idempotent — it exits immediately if the weights are already there — and
takes `--repo`, `--dest`, `--source hf|modelscope`, and `--force`. Current
default lands `Qwen3-0.6B` (bf16, 1.4 GB) in `models/Qwen3-0.6B`, which is what
[localRun0814_A/bin/env.sh](localRun0814_A/bin/env.sh) points at.

### Use ModelScope, not Hugging Face

**The HF weight CDN is blocked on this network and an `HF_TOKEN` will not fix
it.** `huggingface.co` itself is allowed, so metadata (`config.json`, the
tokenizer, the safetensors *index*) downloads fine and you are left with a
plausible-looking ~4 MB model directory containing no weights. Every LFS/Xet
redirect lands on `us.aws.cdn.hf.co` / `cdn-lfs*.hf.co` / `*.xethub.hf.co`,
which the proxy answers with an HTML block page carrying a `403` — category
`"Generative AI"`, id `10046`. `hf-mirror.com` is blocked the same way.

That 403 is the misleading part: `hf` reports it as
`Make sure your token has the correct permissions`, which sends you looking for
an auth problem that does not exist. `fetch_model.sh` detects the block page and
says so instead.

ModelScope is not blocked. Note the underscore — the two CLIs disagree:

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B   # modelscope: --local_dir
hf         download        Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B    # hf:         --local-dir
```

The tradeoff is that HF-only repos are simply unavailable here — including the
`mlx-community` pre-quantized 4-bit builds. Hence bf16, which is fine at this
model size. Quantize at load time instead with `--quantization mlx_q4` if you
want 4-bit. Getting the HF-only repos needs an IT allowlist exception for those
CDN hosts; quote the category above.

### The other failure: Python and the corporate CA

A second, separate problem, easy to confuse with the block. TLS interception
here is **selective**, and which side of it a host falls on decides the error
you get:

| Host | Intercepted | Python without a corporate CA bundle |
| --- | --- | --- |
| `huggingface.co` | no | fine |
| `cdn-lfs-us-1.hf.co`, `*.xethub.hf.co` | **yes** | `CERTIFICATE_VERIFY_FAILED` |
| `www.modelscope.cn`, `*.aliyuncs.com` | no | fine |

On intercepted hosts the proxy presents a corporate root that lives in the macOS
keychain. `curl` trusts it; Python's bundled `certifi` does not, so Python dies
with `self-signed certificate in certificate chain` on hosts where `curl`
succeeds. That is why the `curl` recipe in
[localRun0813/NOTES.md](localRun0813/NOTES.md) worked where `hf` did not.

**This is not why ModelScope works** — ModelScope is not intercepted, and
`modelscope download` succeeds with or without the bundle. Fixing the certs gets
you past the TLS error on the HF hosts and then straight into the 403 block
page. Two problems, and solving the cert one does not solve the other.

`fetch_model.sh` builds a merged bundle (certifi + the keychain roots) at
`~/.config/certs/corp-ca-bundle.pem` on first run and exports it regardless of
source, which costs nothing and covers the HF path if the block is ever lifted.
For any *other* Python tool reaching an intercepted host:

```bash
export SSL_CERT_FILE=~/.config/certs/corp-ca-bundle.pem
export REQUESTS_CA_BUNDLE=~/.config/certs/corp-ca-bundle.pem
```

Delete the file to force a rebuild after a cert rotation. Xet transfers stay
disabled by default because that client is Rust and ignores `SSL_CERT_FILE`
entirely, so it cannot use the bundle at all.

## Serving locally

SGLang has a first-class Apple Metal backend selected by `SGLANG_USE_MLX=1`;
without it the runtime falls back to `torch.mps`, which is far less supported.
See [apple_metal.mdx](../docs/docs/hardware-platforms/apple_metal.mdx).

With the venv built and weights fetched, `localRun0814_A` is one button:

```bash
cd localRun0814_A
./bin/restart_all.sh    # stop everything, start Prometheus/Grafana, start sglang, smoke test
./bin/status.sh         # one-shot snapshot: processes, endpoints, disk, latest logs
./bin/tui.sh            # live read-only dashboard
./bin/stop_all.sh
```

Verified working 2026-08-22: `restart_all.sh` completes all four stages and the
smoke test passes (`/generate`, `/v1/chat/completions`, streaming SSE) with
`device = mps`, `max_total_num_tokens = 32768`.

Two things that look like failures but are not:

- **`tui.sh` showing `UNREACHABLE`** just means no server is running. The TUI is
  read-only and starts fine on its own.
- **An empty `content` with the text in `reasoning`** is
  `--reasoning-parser qwen3` correctly splitting Qwen3's `<think>` block. With a
  small `--max-tokens` the model never gets past reasoning, so `content` stays
  empty.

`start_sglang.sh` refuses to launch unless `$MODEL_PATH` actually contains
`*.safetensors`. A bare directory check is not enough — a blocked or interrupted
download leaves the config and tokenizer behind, and the server would otherwise
die minutes later with an opaque load error.

## What is and isn't kept

[.gitignore](.gitignore) in this directory excludes the heavy and
regenerable parts: `models/`, `venvs/`, and each run's `logs/`, `pcap/`, `run/`,
`data/`, and `obs/data/`. The scripts, configs, notes, and dashboard JSON are
kept.
