# RadixAttention: where it's computed, and running the radix_cache demo locally

Date: 2026-08-13

A Q&A session on `RadixAttention` plus the environment work needed to actually
execute the radix cache demo on this MacBook Pro M3.

---

## Q1 — Where / when / how is RadixAttention computed?

`RadixAttention` is two halves that meet in the forward pass: an `nn.Module` that
every model layer owns, and the radix-tree prefix cache the scheduler maintains
around it. Neither one is "the kernel" — the actual math happens in an attention
backend.

### The layer

Defined at [radix_attention.py:91](../../python/sglang/srt/layers/radix_attention.py#L91). It
holds no weights and does no math itself — it's config (head counts, `layer_id`,
scaling, sliding-window size, KV quant scales) plus a dispatch `forward`.

**Constructed** once per transformer layer at model build time, e.g.
[llama.py:209](../../python/sglang/srt/models/llama.py#L209) inside `LlamaAttention.__init__`.
The `layer_id` it carries is what later indexes into the paged KV pool.

**Called** every forward pass from the model's attention block after QKV proj +
RoPE — [llama.py:262](../../python/sglang/srt/models/llama.py#L262):

```python
attn_output = self.attn(q, k, v, forward_batch)
```

### How the call resolves

[`RadixAttention.forward`](../../python/sglang/srt/layers/radix_attention.py#L150) picks one
of two routes:

1. **Direct** — `get_attn_backend().forward(q, k, v, self, forward_batch, save_kv_cache)`
   ([line 279](../../python/sglang/srt/layers/radix_attention.py#L279)). This is the decode
   path and the plain-eager extend path.
2. **Custom-op** — for extend under torch.compile / piecewise CUDA graphs, it
   allocates the output buffer itself and calls `unified_attention_with_output`
   ([line 405](../../python/sglang/srt/layers/radix_attention.py#L405)), a registered
   custom op / graph split point. It re-narrows padded tensors to real token
   counts and then calls the same backend at
   [line 366](../../python/sglang/srt/layers/radix_attention.py#L366). Variants exist for
   LSE return (chunked-prefix MHA), sparse/indexer attention, and kwargs that
   can't cross the op schema.

Either way it lands in
[`AttentionBackend.forward`](../../python/sglang/srt/layers/attention/base_attn_backend.py#L217),
which branches on `forward_batch.forward_mode` into `forward_decode` /
`forward_extend` / `forward_mixed`. The concrete implementation is whichever
backend is selected (`flashattention_backend.py`, `triton_backend.py`,
`flashinfer_backend.py`, the MLA/NSA ones, etc. — see
[layers/attention/](../../python/sglang/srt/layers/attention/)).

Inside the backend, e.g.
[triton_backend.py:1278](../../python/sglang/srt/layers/attention/triton_backend.py#L1278),
the two steps are always:

- **write** the new K/V into the paged pool at `forward_batch.out_cache_loc` via
  `set_kv_buffer` (gated by `save_kv_cache`, done *before* the kernel);
- **read** the full history back through `kv_indptr` / `kv_indices` and run the
  kernel ([line 1447](../../python/sglang/srt/layers/attention/triton_backend.py#L1447)).
  Q covers only the new tokens; K/V cover prefix + new.

Per-forward metadata (`kv_indptr`, `kv_indices`, `qo_indptr`, masks, max extend
len) is built by `init_forward_metadata` once per batch before the model runs —
[eager_runner.py:232](../../python/sglang/srt/model_executor/runner/eager_runner.py#L232), or
pre-planned by the CUDA-graph runners.

### The radix half (why only new tokens are computed)

This runs on the scheduler, outside the model:

- **Match** — before a batch is formed, `tree_cache.match_prefix(...)` looks up
  the longest cached prefix of the request's token ids:
  [schedule_policy.py:156](../../python/sglang/srt/managers/schedule_policy.py#L156)
  (scheduling decisions) and
  [schedule_batch.py:1356](../../python/sglang/srt/managers/schedule_batch.py#L1356) (the
  authoritative match that sets `req.prefix_indices`). Implementation:
  [`RadixCache.match_prefix`](../../python/sglang/srt/mem_cache/radix_cache.py#L352).
- **Carry** — `prefix_lens` becomes `forward_batch.extend_prefix_lens`
  ([forward_batch_info.py:771](../../python/sglang/srt/model_executor/forward_batch_info.py#L771)),
  and the matched page indices go straight into the backend's `kv_indices`.
  That's the whole trick: matched tokens are never re-run through Q, but their
  KV is still attended.
- **Insert** — after prefill chunks and at request completion, KV is handed back
  to the tree via `cache_unfinished_req` / `cache_finished_req`
  ([radix_cache.py:434](../../python/sglang/srt/mem_cache/radix_cache.py#L434), called
  through [`maybe_cache_unfinished_req`](../../python/sglang/srt/managers/scheduler.py#L2917)).
  Eviction is LRU over the tree.

Turning this off with `--disable-radix-cache`
([server_args.py:929](../../python/sglang/srt/server_args.py#L929)) swaps `RadixCache` for
`ChunkCache` — the `RadixAttention` layer and the kernels are unchanged, prefix
hits just always come back empty.

There are many tree variants for different memory layouts
([`hiradix_cache.py`](../../python/sglang/srt/mem_cache/hiradix_cache.py),
[`swa_radix_cache.py`](../../python/sglang/srt/mem_cache/swa_radix_cache.py),
[`mamba_radix_cache.py`](../../python/sglang/srt/mem_cache/mamba_radix_cache.py), a C++
one), all behind the same `BasePrefixCache` interface.

---

## Q2 — Why `create_simulated` at radix_cache.py:816?

Line 816 is inside the `if __name__ == "__main__":` block at
[radix_cache.py:815](../../python/sglang/srt/mem_cache/radix_cache.py#L815) — a standalone
demo you can run with `python -m sglang.srt.mem_cache.radix_cache` to eyeball the
tree's insert/match/split behavior.

The plain constructor can't be used there. `RadixCache.__init__` takes a
[`CacheInitParams`](../../python/sglang/srt/mem_cache/cache_init_params.py) carrying a
`req_to_token_pool` and a `token_to_kv_pool_allocator` — real GPU-backed pools
that only exist once a scheduler has booted a model. `create_simulated`
([line 310](../../python/sglang/srt/mem_cache/radix_cache.py#L310)) fills those with
`None`/a mock so you get a tree with no memory pools behind it.

That's enough for what the demo does, because insert / `match_prefix` /
`pretty_print` are pure tree operations over token-id keys. The allocator is only
consulted for the parts the demo never reaches:

- `self.device` falls back to `torch.device("cpu")` when there's no allocator
  ([lines 295-302](../../python/sglang/srt/mem_cache/radix_cache.py#L295-L302)), so the
  empty match result and index tensors land on CPU;
- eviction and `cache_finished_req` are what actually call into the allocator to
  free KV pages.

Same reason the unit tests use it —
[test_radix_cache_unit.py](../../test/registered/unit/mem_cache/test_radix_cache_unit.py)
and [test_schedule_policy.py](../../test/manual/test_schedule_policy.py) exercise
tree logic without a GPU, passing `mock_allocator=` when a test needs to observe
frees.

Nit worth noting if you touch it: it's decorated `@classmethod` but its first
parameter is named `self` rather than `cls`
([line 311](../../python/sglang/srt/mem_cache/radix_cache.py#L311)). It works — the name is
arbitrary — but it reads as an instance method.

---

## Q3 — Running `python -m sglang.srt.mem_cache.radix_cache`

### Result

```
 0 array('q') r=1
   5 array('q', [8, 9, 10, 11, 12]) r=0
   2 array('q', [1, 2]) r=0
     2 array('q', [4, 5]) r=0
       2 array('q', [6, 7]) r=0
     1 array('q', [3]) r=0
#tokens: 12

MatchResult(device_indices=tensor([1, 2, 3]), last_device_node=<TreeNode at 0x13d66f4d0>,
            last_host_node=<same>, best_match_node=<same>, host_hit_length=0,
            swa_host_hit_length=0, mamba_host_hit_length=0, mamba_branching_seqlen=None,
            cache_protected_len=None, full_kv_hit_length=0, cache_actions=())
```

Reading it: indent is depth, the leading number is `len(node.key)`, then the
first 10 token ids, `r=` is `lock_ref`
([`_print_helper`](../../python/sglang/srt/mem_cache/radix_cache.py#L760)). The five inserts
produce exactly the split you'd want — the duplicate `[1,2,3]` is a no-op, then
`[1,2,4,5]` splits the existing node into `[1,2]` + `[3]` and hangs `[4,5]` off
the shared prefix, `[1,2,4,5,6,7]` extends that with `[6,7]`, and
`[8,9,10,11,12]` is an unrelated root child. `r=1` on root is its permanent lock;
12 total tokens = 2+1+2+2+5.

The match on `[1, 2, 3, 13, 14]` returns `device_indices=tensor([1, 2, 3])` — a
3-token hit that stops where the tree diverges. Those values *are* the token ids,
because with no allocator `insert` falls back to using the token ids as the KV
indices ([lines 423-426](../../python/sglang/srt/mem_cache/radix_cache.py#L423-L426)); in a
real server they'd be page indices into the KV pool.

### What it took to get there

The venv at `sglang/.venv` (Python 3.14.6) started out as an analysis-only
environment — torch plus a handful of base packages, with `sglang` itself not
installed (see [SETUP_ENV.md](../../studyIterations/SETUP_ENV.md)). Two
separate problems had to be solved.

#### 1. `sglang` isn't installed, and its `__init__` pulls a long dependency chain

Run with the source on the path:

```bash
PYTHONPATH=/Users/WChen/AI/sglangTry/sglang/python python -m sglang.srt.mem_cache.radix_cache
```

`sglang/__init__.py` imports `hf_transformers_patches`, `lang.api`, and the
OpenAI-protocol entrypoints, and `radix_cache`'s own chain
(`base_prefix_cache` → `memory_pool` → `layers.quantization` → `layers.linear` →
`layers.moe`) drags in most of the runtime. Installed, in order of discovery:

```bash
pip install orjson psutil pybase64 requests packaging pillow starlette typing_extensions
pip install torchvision tqdm ipython aiohttp dill openai partial_json_parser
pip install xgrammar sentencepiece einops compressed-tensors gguf
pip install "transformers==5.12.1"
```

Notes:

- **`transformers` must be pinned to 5.12.1**, the version in
  [pyproject.toml:87](../../python/pyproject.toml#L87). pip's default (5.15.0) fails at
  import because [configs/qwen3_asr.py:167](../../python/sglang/srt/configs/qwen3_asr.py#L167)
  does `AutoConfig.register("qwen3_asr", ...)` and upstream transformers now owns
  that model type:
  `ValueError: 'qwen3_asr' is already used by a Transformers config`.
- `torchvision` 0.28.0 resolves cleanly against the installed torch 2.13.0 —
  verified with `pip install --dry-run` first so it wouldn't silently swap torch.
- No CUDA-only packages (`sgl-kernel`, `flashinfer`) were needed for this path.

#### 2. A macOS-specific bug in sglang's triton stub

Even with every dependency present, the import dies at
[moe_runner/deep_gemm.py:81](../../python/sglang/srt/layers/moe/moe_runner/deep_gemm.py#L81),
a module-level `@torch.compile` decorator that drags in `torch._inductor`:

```
torch/_inductor/runtime/triton_heuristics.py:132: TypeError:
    unsupported operand type(s) for |: 'module' and 'type'
    CompiledKernel | StaticallyLaunchedCudaKernel | StaticallyLaunchedXpuKernel
```

Cause: [`_triton_stub.py`](../../python/sglang/_triton_stub.py) fakes the `triton` package
on Apple silicon, and its `_TritonFinder` resolves *any* unknown `triton.*`
submodule to a `_MockModule`. Torch's `triton_compat` does
`from triton.compiler import CompiledKernel` and then unions it as a type — so it
gets a module where a class is required. The stub provides
`triton.backends.compiler` ([lines 224-228](../../python/sglang/_triton_stub.py#L224-L228))
but nothing for `triton.compiler`.

`TORCH_COMPILE_DISABLE=1` does **not** help — inductor is imported regardless of
whether compilation is enabled.

Workaround used (throwaway file at `/tmp/run_radix_demo.py`, no repo files
changed):

```python
import runpy

import sglang  # noqa: F401  -- installs the triton/mps stubs

import triton.compiler as _tc

if not isinstance(getattr(_tc, "CompiledKernel", None), type):

    class CompiledKernel:  # minimal stand-in
        pass

    _tc.CompiledKernel = CompiledKernel

runpy.run_module("sglang.srt.mem_cache.radix_cache", run_name="__main__")
```

```bash
PYTHONPATH=/Users/WChen/AI/sglangTry/sglang/python python /tmp/run_radix_demo.py
```

The durable fix would be adding a `triton.compiler` module to the stub with
`CompiledKernel` as a real class, next to the existing `triton.backends.compiler`
entry. Not applied — the demo run didn't require modifying the repo.

Two harmless warnings are printed on the way through:

```
awq.py:52: UserWarning: Only CUDA, HIP and XPU support AWQ currently.
gguf.py:69: UserWarning: Only CUDA, MUSA and NPU support GGUF quantization currently.
```
