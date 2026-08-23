#!/usr/bin/env bash
# Download the study-run model weights into $MODEL_PATH. Idempotent.
#
# Usage: fetch_model.sh [--repo <id>] [--dest <dir>] [--source hf|modelscope]
#                       [--force] [--xet]
#   e.g. fetch_model.sh                                       # whatever env.sh names
#        fetch_model.sh --source hf --repo mlx-community/Qwen3-0.6B-4bit
#
# Default source is ModelScope. On this machine HuggingFace weights cannot be
# downloaded at all -- see trap 2 below -- while ModelScope serves the same Qwen
# weights unblocked. The HF path is kept because it is the right default on any
# network without the block.
#
# Two traps this exists to handle, both hit for real on 2026-08-22:
#
#   1. Python can't verify TLS to the HF CDN even though curl can. Netskope
#      MITMs those hosts with a corporate root that lives in the macOS keychain,
#      which certifi knows nothing about. Symptom is CERTIFICATE_VERIFY_FAILED
#      "self-signed certificate in certificate chain". Handled by building
#      $CORP_CA_BUNDLE below.
#
#   2. A "403 Forbidden" from the CDN that is not HuggingFace saying no -- it is
#      a Netskope block page (category "Generative AI") served with a 403. The
#      raw hf CLI error points at HF permissions and sends you hunting for an
#      HF_TOKEN you don't need. diagnose_failure() detects the block page and
#      says so.
#
# Metadata (config.json, tokenizer, the safetensors *index*) comes from
# huggingface.co, which is NOT blocked, so a blocked run still leaves a
# plausible-looking ~4 MB directory with no weights in it. That is why the
# success check is "are there .safetensors bytes", never "does the dir exist".

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

FORCE=0
USE_XET=0   # off by default: the xet transfer path 403s through the proxy, and
            # its Rust client ignores SSL_CERT_FILE, so it can't use our bundle.

while (( $# )); do
  case "$1" in
    --repo)   MODEL_REPO="$2"; MODEL_PATH="$STUDY_ROOT/models/${2##*/}"; shift 2 ;;
    --dest)   MODEL_PATH="$2"; shift 2 ;;
    --source) MODEL_SOURCE="$2"; shift 2 ;;
    --force)  FORCE=1; shift ;;
    --xet)    USE_XET=1; shift ;;
    -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

case "$MODEL_SOURCE" in
  hf|modelscope) ;;
  *) die "--source must be 'hf' or 'modelscope', got: $MODEL_SOURCE" ;;
esac

DOWNLOADER="$VENV/bin/$([[ $MODEL_SOURCE == hf ]] && echo hf || echo modelscope)"
[[ -x "$DOWNLOADER" ]] || die "$MODEL_SOURCE CLI not found at $DOWNLOADER -- is the venv built? See studyIterations/SETUP_ENV.md"

# Already done? `hf download` is itself incremental, but skipping entirely keeps
# this cheap to call from other scripts.
if (( ! FORCE )) && compgen -G "$MODEL_PATH"/*.safetensors >/dev/null; then
  log "already present: $MODEL_PATH ($(du -sh "$MODEL_PATH" | cut -f1))"
  exit 0
fi

# --- CA bundle ---------------------------------------------------------------
# certifi's roots plus whatever corporate roots the keychain holds. Rebuilt only
# when missing; delete the file to force a refresh after a cert rotation.
build_ca_bundle() {
  local tmp
  tmp="$(mktemp)"
  "$PY" -c 'import certifi,sys; sys.stdout.write(open(certifi.where()).read())' > "$tmp" || {
    rm -f "$tmp"; return 1
  }
  # -c matches on the cert's common name. Both are needed: the leaf-signing CA
  # and the Netskope root above it. Absent on a non-corporate machine, which is
  # fine -- certifi alone is then correct and the greps simply add nothing.
  security find-certificate -a -c "ca.jackhenry.goskope.com" -p 2>/dev/null >> "$tmp" || true
  security find-certificate -a -c "certadmin" -p 2>/dev/null >> "$tmp" || true

  mkdir -p "$(dirname "$CORP_CA_BUNDLE")"
  mv "$tmp" "$CORP_CA_BUNDLE"
  log "built CA bundle: $CORP_CA_BUNDLE ($(grep -c 'BEGIN CERT' "$CORP_CA_BUNDLE") certs)"
}

if [[ ! -f "$CORP_CA_BUNDLE" ]]; then
  log "no CA bundle at $CORP_CA_BUNDLE, building one"
  build_ca_bundle || warn "could not build CA bundle, falling back to certifi defaults"
fi
if [[ -f "$CORP_CA_BUNDLE" ]]; then
  export SSL_CERT_FILE="$CORP_CA_BUNDLE"
  export REQUESTS_CA_BUNDLE="$CORP_CA_BUNDLE"
fi

(( USE_XET )) || export HF_HUB_DISABLE_XET=1

# --- Diagnosis ---------------------------------------------------------------
# Called only after a failure. Fetches one real weight URL and looks at what
# actually came back, because the CLI's own error message is misleading when a
# proxy is the one refusing.
diagnose_failure() {
  local probe body code

  # The block-page probe below only makes sense against HuggingFace. Guessing a
  # ModelScope resolve URL would produce a confident, wrong answer.
  if [[ "$MODEL_SOURCE" != hf ]]; then
    warn "download from $MODEL_SOURCE did not produce weights."
    warn "Re-run without the tail filter to see the full CLI output:"
    warn "    $DOWNLOADER download --model $MODEL_REPO --local_dir $MODEL_PATH"
    return
  fi

  # Learn a real weight filename from the index rather than guessing
  # "model.safetensors" -- sharded repos don't have that file. The index is
  # metadata, so it downloads even when the CDN is blocked.
  probe="$("$PY" - "$MODEL_PATH" <<'EOF' 2>/dev/null || true
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
idx = d / "model.safetensors.index.json"
if idx.exists():
    print(sorted(set(json.load(idx.open())["weight_map"].values()))[0])
else:
    print("model.safetensors")
EOF
)"
  [[ -n "$probe" ]] || probe="model.safetensors"

  body="$(mktemp)"
  code="$(curl -sL -o "$body" -w '%{http_code}' -m 60 \
    "https://huggingface.co/$MODEL_REPO/resolve/main/$probe" 2>/dev/null || echo 000)"

  if grep -qi 'NS_APP_CATEGORY\|netskope' "$body" 2>/dev/null; then
    local cat
    cat="$(grep -o 'appCategory = "[^"]*"' "$body" 2>/dev/null | head -1 | cut -d'"' -f2)"
    warn "the CDN returned a Netskope block page (HTTP $code), category: ${cat:-unknown}"
    warn "huggingface.co is allowed here, but the weight CDN hosts are not:"
    warn "    us.aws.cdn.hf.co  cdn-lfs*.hf.co  *.xethub.hf.co"
    warn "This is a corporate network policy, not an HF permissions problem --"
    warn "an HF_TOKEN will NOT help. It needs an IT allowlist exception for"
    warn "those hosts (quote the category above), or a network without Netskope."
  elif (( code == 000 )); then
    warn "could not reach huggingface.co at all (HTTP $code) -- offline, or DNS/proxy is down"
  else
    warn "weight fetch returned HTTP $code and the body is not a known block page."
    warn "First 200 bytes of the response:"
    head -c 200 "$body" | sed 's/^/    /' >&2; echo >&2
  fi
  rm -f "$body"
}

# --- Download ----------------------------------------------------------------
log "source: $MODEL_SOURCE"
log "repo  : $MODEL_REPO"
log "dest  : $MODEL_PATH"
mkdir -p "$(dirname "$MODEL_PATH")"

# `|| true`: a non-zero exit here is expected in the blocked case and the
# diagnosis below is far more useful than the CLI's own message. Real success is
# decided by the weights check, not by this exit code.
#
# Note the flag spelling differs between the two CLIs: hf takes --local-dir,
# modelscope takes --local_dir. Easy to get wrong when copying one to the other.
if [[ "$MODEL_SOURCE" == hf ]]; then
  "$DOWNLOADER" download "$MODEL_REPO" --local-dir "$MODEL_PATH" 2>&1 | tail -20 || true
else
  "$DOWNLOADER" download --model "$MODEL_REPO" --local_dir "$MODEL_PATH" 2>&1 | tail -5 || true
fi

if compgen -G "$MODEL_PATH"/*.safetensors >/dev/null; then
  log "OK -- $(du -sh "$MODEL_PATH" | cut -f1) in $MODEL_PATH"
  ls -lh "$MODEL_PATH"/*.safetensors | sed 's/^/    /'
  log "start the server with: $BIN_DIR/start_sglang.sh"
  exit 0
fi

warn "no *.safetensors in $MODEL_PATH -- metadata only, so the weights were refused"
diagnose_failure
die "model download incomplete"
