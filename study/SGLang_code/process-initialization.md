# Process Creation in the SGLang Initialization Sequence

## Ask

How are the processes are created in the initialization sequence? create a MD file under sglang/study/SGLang_code to document this, with linked to the line in the source files.

Later: The links in line 58 do not work, missing the ../../ prefix

## Overview
How `sglang serve` (or `sgl.Engine(...)`) turns one Python process into the multi-process
runtime: who spawns whom, in what order, with what arguments, and how the parent knows the
children are alive.

Source anchors point at `main` as of commit `5ffe8d8f57`.

---

## 1. The short version

```
sglang serve                          (process 0 — "main"/tokenizer process)
│
├── weight cache daemons              subprocess.Popen, only --weight-cache-mode daemon
│     `python -m sglang.srt.weight_cache.daemon`      one per local PP x TP rank
│
├── scheduler processes               mp.Process, one per (pp_rank, tp_rank) on this node
│     sglang::scheduler_TP0 ...       <-- OR -->
│   └── data parallel controller      mp.Process, when dp_size > 1 / ep_join_mode=="scale"
│         sglang::data_parallel_controller
│         └── scheduler processes     mp.Process, one per (dp_rank, pp_rank, tp_rank)
│
├── detokenizer process(es)           mp.Process
│     sglang::detokenizer             (1 worker)
│     sglang::detokenizer_0..N-1  +  sglang::detokenizer_router   (N workers)
│
├── expert backup manager             mp.Process, only --enable-elastic-expert-backup
│
├── sidecar process                   mp.Process(spawn ctx), only --sidecar
│
└── uvicorn HTTP workers              only --tokenizer-worker-num > 1
      sglang::tokenizer_worker:<pid>  one TokenizerWorker per uvicorn worker process
```

Everything not in that tree stays in process 0: the FastAPI/uvicorn app, the
`TokenizerManager` (or `MultiTokenizerRouter`), the `TemplateManager`, and the watchdog
threads. That is what the `launch_server` docstring means by "The HTTP server, Engine, and
TokenizerManager all run in the main process"
([http_server.py:2785-2787](../../python/sglang/srt/entrypoints/http_server.py#L2785-L2787)).

---

## 2. Getting to the launcher

Three entry paths converge on the same function.

| Entry | Path                                                                                                                                                                                                    |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sglang serve ...` | [cli/main.py:37-40](../../python/sglang/cli/main.py#L37-L40) → [cli/serve.py:166](../../python/sglang/cli/serve.py#L166) → backend dispatch → [cli/serve.py:95-99](../../python/sglang/cli/serve.py#L95-L99) `run_server` |
| `python -m sglang.launch_server` | [launch_server.py:88](../../python/sglang/launch_server.py#L88) `run_server`                                                                                                                                  |
| `sgl.Engine(...)` | [engine.py:232](../../python/sglang/srt/entrypoints/engine.py#L232) `Engine.__init__`                                                                                                                         |

`run_server` picks a server flavor — encoder-only, legacy SMG gRPC, Ray, or the default
HTTP path ([launch_server.py:22-74](../../python/sglang/launch_server.py#L22-L74)). The default
path lands in [http_server.py:2766](../../python/sglang/srt/entrypoints/http_server.py#L2766)
`launch_server`, which immediately calls
[`Engine._launch_subprocesses`](../../python/sglang/srt/entrypoints/engine.py#L1068) at
[http_server.py:2798](../../python/sglang/srt/entrypoints/http_server.py#L2798).

So the HTTP server and the offline Engine share **one** process-creation routine:
`Engine._launch_subprocesses`.

---

## 3. `Engine._launch_subprocesses` — the spine

[engine.py:1068-1264](../../python/sglang/srt/entrypoints/engine.py#L1068-L1264). In order:

1. **Configure the global environment** —
   [`_set_envs_and_config`](../../python/sglang/srt/entrypoints/engine.py#L1659) at
   [engine.py:1092](../../python/sglang/srt/entrypoints/engine.py#L1092). This is where the
   start method is fixed:

   ```python
   mp.set_start_method("spawn", force=True)
   ```
   [engine.py:1754](../../python/sglang/srt/entrypoints/engine.py#L1754)

   **Spawn, not fork.** Every child is a fresh interpreter that re-imports sglang and
   re-receives its arguments by pickle. That is why `ServerArgs` / `PortArgs` must be
   picklable, why each child re-runs `load_plugins()` and `publish(...)`, and why CUDA
   state is never inherited.

   The same function installs the launch-phase `SIGQUIT` handler
   ([engine.py:1733-1739](../../python/sglang/srt/entrypoints/engine.py#L1733-L1739)): any child
   that dies signals `SIGQUIT` to the parent, and the parent tears down the whole process
   tree with `kill_process_tree(os.getpid())`.

2. **Allocate the IPC endpoints** — `PortArgs.init_new(server_args)` at
   [engine.py:1102](../../python/sglang/srt/entrypoints/engine.py#L1102), defined at
   [server_args.py:9984](../../python/sglang/srt/server_args.py#L9984). Single-node uses
   `ipc://` unix sockets backed by temp files
   ([server_args.py:10024-10030](../../python/sglang/srt/server_args.py#L10024-L10030));
   multi-node switches to `tcp://` addresses derived from `--dist-init-addr`
   ([server_args.py:10108-10120](../../python/sglang/srt/server_args.py#L10108-L10120)). The
   `PortArgs` object is then passed by value into every child, which is how the processes
   find each other — there is no discovery step.

3. **Publish the resolved config** — `publish(server_args, role="tokenizer")` at
   [engine.py:1131](../../python/sglang/srt/entrypoints/engine.py#L1131)
   ([runtime_context.py:1285](../../python/sglang/srt/runtime_context.py#L1285)). Each spawned
   child re-publishes under its own role before reading any config.

4. **Weight cache daemons** (optional) — [engine.py:1136-1137](../../python/sglang/srt/entrypoints/engine.py#L1136-L1137).
5. **Scheduler processes** — [engine.py:1145](../../python/sglang/srt/entrypoints/engine.py#L1145).
6. **Expert backup manager** (optional) — [engine.py:1152-1156](../../python/sglang/srt/entrypoints/engine.py#L1152-L1156).
7. **Early return for non-zero node ranks** — [engine.py:1158-1185](../../python/sglang/srt/entrypoints/engine.py#L1158-L1185).
8. **Detokenizer process(es)** — [engine.py:1212](../../python/sglang/srt/entrypoints/engine.py#L1212).
9. **TokenizerManager in-process** — [engine.py:1221-1228](../../python/sglang/srt/entrypoints/engine.py#L1221-L1228).
10. **Barrier: wait for schedulers to report ready** — [engine.py:1233](../../python/sglang/srt/entrypoints/engine.py#L1233).
11. **Start the liveness watchdog** — [engine.py:1248-1251](../../python/sglang/srt/entrypoints/engine.py#L1248-L1251).

Note the ordering in 8-10: the detokenizer and the tokenizer manager are created *before*
the code blocks on scheduler readiness, so model loading (the slow part) overlaps with
tokenizer/template initialization.

---

## 4. Scheduler processes

[`_launch_scheduler_processes`](../../python/sglang/srt/entrypoints/engine.py#L858), lines
858-976. It branches on whether a data-parallel controller is needed
([engine.py:874-876](../../python/sglang/srt/entrypoints/engine.py#L874-L876)):

```python
use_dp_controller = get_parallel().dp_size > 1 or get_exec().moe.ep_join_mode == "scale"
```

### 4a. No DP controller — flat TP x PP fan-out

[engine.py:878-930](../../python/sglang/srt/entrypoints/engine.py#L878-L930).

- [`_calculate_rank_ranges`](../../python/sglang/srt/entrypoints/engine.py#L1832) decides which
  slice of the global `(pp_rank, tp_rank)` grid **this node** owns, from `nnodes` and
  `node_rank`. Each node launches only its own slice; there is no cross-node process
  spawning.
- For every `(pp_rank, tp_rank)` in that slice:
  - a one-way pipe is created for the readiness handshake —
    [engine.py:896](../../python/sglang/srt/entrypoints/engine.py#L896);
  - `gpu_id` is computed from `base_gpu_id`, the per-node PP/TP sizes, and `gpu_id_step`
    ([engine.py:897-901](../../python/sglang/srt/entrypoints/engine.py#L897-L901));
  - [`_compute_parallelism_ranks`](../../python/sglang/srt/entrypoints/engine.py#L1867) derives
    the attention-CP, MoE-DP, and MoE-EP ranks from the TP rank;
  - the process is constructed at
    [engine.py:908-922](../../python/sglang/srt/entrypoints/engine.py#L908-L922) and started at
    [engine.py:927](../../python/sglang/srt/entrypoints/engine.py#L927).

  ```python
  proc = mp.Process(
      target=run_scheduler_process_func,
      args=(server_args, port_args, gpu_id, tp_rank, attn_cp_rank,
            moe_dp_rank, moe_ep_rank, pp_rank, None, writer),
  )
  with (memory_saver_adapter.configure_subprocess(),
        numa_utils.configure_subprocess(server_args, gpu_id)):
      proc.start()
  ```

  Two context managers wrap `.start()` because they mutate process-inherited state:
  `maybe_reindex_device_id` ([engine.py:906](../../python/sglang/srt/entrypoints/engine.py#L906))
  sets `CUDA_VISIBLE_DEVICES` for the child, and `numa_utils.configure_subprocess` pins the
  child's NUMA node. With `spawn`, the child inherits the environment as it stands at
  `start()`.

### 4b. With DP controller — one hop of indirection

[engine.py:931-947](../../python/sglang/srt/entrypoints/engine.py#L931-L947). The engine spawns a
*single* process, and that process spawns the schedulers:

```python
proc = mp.Process(
    target=run_data_parallel_controller_process,
    kwargs=dict(server_args=..., port_args=..., pipe_writer=writer,
                run_scheduler_process_func=run_scheduler_process_func),
)
```

`run_scheduler_process_func` is passed *through* the controller, which is why it must be a
module-level function (picklable) rather than a closure. `Engine` exposes it as an override
point at [engine.py:223-226](../../python/sglang/srt/entrypoints/engine.py#L223-L226).

---

## 5. Inside the DataParallelController

[`run_data_parallel_controller_process`](../../python/sglang/srt/managers/data_parallel_controller.py#L811)
sets its proc title, installs `kill_itself_when_parent_died`, publishes under role
`dp_controller`, then constructs
[`DataParallelController`](../../python/sglang/srt/managers/data_parallel_controller.py#L135) —
whose `__init__` is what actually launches the schedulers
([data_parallel_controller.py:198-209](../../python/sglang/srt/managers/data_parallel_controller.py#L198-L209)):

- **DP attention on** →
  [`launch_dp_attention_schedulers`](../../python/sglang/srt/managers/data_parallel_controller.py#L545).
  DP ranks share one TP group, so there is a single call to `launch_tensor_parallel_group`
  after the worker ZMQ ports are bound on rank 0 and broadcast to the other nodes.
- **DP attention off** →
  [`launch_dp_schedulers`](../../python/sglang/srt/managers/data_parallel_controller.py#L363).
  One **thread** per DP rank
  ([data_parallel_controller.py:383-387](../../python/sglang/srt/managers/data_parallel_controller.py#L383-L387)),
  each thread calling `launch_tensor_parallel_group` for its rank so the TP groups come up
  concurrently. Each rank gets its own `PortArgs.init_new`, and the NCCL port is held open
  with `bind_port` until all threads start so ranks cannot collide
  ([data_parallel_controller.py:377](../../python/sglang/srt/managers/data_parallel_controller.py#L377)).

  These launcher threads then sleep forever
  ([data_parallel_controller.py:420-424](../../python/sglang/srt/managers/data_parallel_controller.py#L420-L424)) —
  the comment explains why: exiting the thread would look like parent death to
  `kill_itself_when_parent_died` in the scheduler.

[`launch_tensor_parallel_group`](../../python/sglang/srt/managers/data_parallel_controller.py#L593)
is the DP-aware twin of §4a: same rank-range math, same `gpu_id` arithmetic, same pipe
handshake, and the actual spawn at
[data_parallel_controller.py:701-722](../../python/sglang/srt/managers/data_parallel_controller.py#L701-L722).
It passes three extra `display_*` ranks used only for log prefixes when elastic EP shifts
the global rank numbering.

The controller waits for all its children
([data_parallel_controller.py:728-730](../../python/sglang/srt/managers/data_parallel_controller.py#L728-L730)),
then reports readiness upward, including the child PIDs so the engine can track the full
tree ([data_parallel_controller.py:845-852](../../python/sglang/srt/managers/data_parallel_controller.py#L845-L852),
collected back at [engine.py:955-958](../../python/sglang/srt/entrypoints/engine.py#L955-L958)).

---

## 6. What a scheduler child actually does

[`run_scheduler_process`](../../python/sglang/srt/managers/scheduler.py#L5140):

1. `load_plugins()` — the spawned interpreter has none of the parent's imports.
2. `publish(server_args, role="scheduler")` — before anything reads config.
3. [`configure_scheduler_process`](../../python/sglang/srt/managers/scheduler.py#L5075) —
   `kill_itself_when_parent_died()`, the `sglang::scheduler_TP0_EP1...` proc title, the
   log prefix, `faulthandler.enable()`, CPU affinity, and NUMA binding.
4. Construct `Scheduler(...)` — this is the expensive step: distributed init, weight load,
   KV cache allocation, CUDA graph capture.
5. `pipe_writer.send(scheduler.get_init_info())` — the readiness message the parent is
   blocking on ([scheduler.py:5204](../../python/sglang/srt/managers/scheduler.py#L5204)).
6. `scheduler.run_event_loop()` — never returns until shutdown.

On exception it sends `SIGQUIT` to the parent
([scheduler.py:5212](../../python/sglang/srt/managers/scheduler.py#L5212)), which is the other
half of the handler installed in `_set_envs_and_config`.

---

## 7. Detokenizer processes

[`_launch_detokenizer_subprocesses`](../../python/sglang/srt/entrypoints/engine.py#L979).

- `detokenizer_worker_num <= 1` → one `mp.Process` running
  [`run_detokenizer_process`](../../python/sglang/srt/managers/detokenizer_manager.py#L516),
  bound to `port_args.detokenizer_ipc_name`
  ([engine.py:1000-1003](../../python/sglang/srt/entrypoints/engine.py#L1000-L1003)).
- `detokenizer_worker_num > 1` → N workers, each given a *private* IPC socket by
  temporarily rewriting `port_args.detokenizer_ipc_name` before each spawn and restoring it
  in a `finally`
  ([engine.py:1011-1025](../../python/sglang/srt/entrypoints/engine.py#L1011-L1025)) — a
  deliberate mutate-spawn-restore, since `spawn` pickles `port_args` at `start()` time.
  Then one extra process runs
  [`run_multi_detokenizer_router_process`](../../python/sglang/srt/managers/multi_tokenizer_mixin.py#L625),
  which owns the original IPC name and fans out to the workers by `crc32` of
  `http_worker_ipc` so a request's outputs always hit the same detokenizer
  ([multi_tokenizer_mixin.py:555-573](../../python/sglang/srt/managers/multi_tokenizer_mixin.py#L555-L573)).

Detokenizers have no readiness handshake — they are only watched by the watchdog.

---

## 8. Tokenizer processes (`--tokenizer-worker-num > 1`)

These are **not** created by `_launch_subprocesses`. They are uvicorn worker processes, so
the fork/spawn happens inside uvicorn.

1. In the main process, `_launch_subprocesses` builds a
   [`MultiTokenizerRouter`](../../python/sglang/srt/managers/multi_tokenizer_mixin.py#L429)
   instead of a `TokenizerManager`
   ([engine.py:1225-1228](../../python/sglang/srt/entrypoints/engine.py#L1225-L1228)).
2. `_setup_and_run_http_server` writes `port_args`, `server_args`, and `scheduler_info`
   into POSIX shared memory
   ([http_server.py:2578-2585](../../python/sglang/srt/entrypoints/http_server.py#L2578-L2585)),
   because uvicorn workers are started by uvicorn from the module path and cannot be handed
   Python objects.
3. `uvicorn.run("sglang.srt.entrypoints.http_server:app", workers=N, ...)`
   ([http_server.py:2705-2719](../../python/sglang/srt/entrypoints/http_server.py#L2705-L2719)).
4. Each worker runs the FastAPI
   [`lifespan`](../../python/sglang/srt/entrypoints/http_server.py#L269), which takes the
   non-single-tokenizer branch and calls
   [`init_multi_tokenizer`](../../python/sglang/srt/entrypoints/http_server.py#L216): read the
   shared memory, mint a fresh per-process `tokenizer_ipc_name`, and construct a
   [`TokenizerWorker`](../../python/sglang/srt/managers/multi_tokenizer_mixin.py#L647) which
   registers itself with the router.
5. The shared memory segment is unlinked in the main process's `finally`
   ([http_server.py:2721-2723](../../python/sglang/srt/entrypoints/http_server.py#L2721-L2723)).

With `--tokenizer-worker-num 1` (the default) none of this happens: the app object is
passed to `uvicorn.run` directly, and `app.is_single_tokenizer_mode = True` tells `lifespan`
to reuse the already-constructed `TokenizerManager`
([http_server.py:2545-2553](../../python/sglang/srt/entrypoints/http_server.py#L2545-L2553)).

---

## 9. Optional processes

| Process | Gate | Created at |
|---|---|---|
| Weight cache daemons | `--weight-cache-mode daemon` | `subprocess.Popen` at [engine.py:786](../../python/sglang/srt/entrypoints/engine.py#L786), one per local PP x TP rank, launched as `python -m sglang.srt.weight_cache.daemon` ([engine.py:742-768](../../python/sglang/srt/entrypoints/engine.py#L742-L768)) |
| Expert backup manager | `--enable-elastic-expert-backup` + an EP backend | [expert_backup_manager.py:181-186](../../python/sglang/srt/elastic_ep/expert_backup_manager.py#L181-L186) |
| Sidecar | `--sidecar <module>` | [sidecar.py:121-125](../../python/sglang/srt/entrypoints/sidecar.py#L121-L125), started from `lifespan` at [http_server.py:407-409](../../python/sglang/srt/entrypoints/http_server.py#L407-L409) |

The weight cache daemons are the odd one out: `Popen` rather than `mp.Process`, so they are
real external commands, and readiness is a filesystem poll for a `.ready` file rather than a
pipe ([engine.py:796-822](../../python/sglang/srt/entrypoints/engine.py#L796-L822)). Partial
failure terminates the siblings already spawned so no GPU-resident daemon leaks.

---

## 10. Synchronization and failure handling

**Readiness.** Every scheduler holds the write end of an `mp.Pipe`; the parent polls the
read ends in [`_wait_for_scheduler_ready`](../../python/sglang/srt/entrypoints/engine.py#L1800).
It uses `poll(timeout=5.0)` in a loop rather than a blocking `recv()` specifically so that a
child SIGKILLed by the OS OOM killer is noticed instead of hanging the launch forever
([engine.py:1824-1827](../../python/sglang/srt/entrypoints/engine.py#L1824-L1827)); the error
message even tells you to check `dmesg` for OOM
([engine.py:1789-1797](../../python/sglang/srt/entrypoints/engine.py#L1789-L1797)).

**Three independent death mechanisms:**

1. *Child dies → parent notices.*
   [`SubprocessWatchdog`](../../python/sglang/srt/utils/watchdog.py#L166) polls
   `proc.is_alive()` in a daemon thread every second and raises `SIGQUIT` on the main
   process when a child exits non-zero — this covers C++-level aborts (NCCL timeouts) where
   no Python handler ever runs.
2. *Child hits a Python exception → parent notices.* The child explicitly
   `parent_process.send_signal(signal.SIGQUIT)`
   ([scheduler.py:5212](../../python/sglang/srt/managers/scheduler.py#L5212),
   [detokenizer_manager.py:539](../../python/sglang/srt/managers/detokenizer_manager.py#L539),
   [data_parallel_controller.py:865](../../python/sglang/srt/managers/data_parallel_controller.py#L865)).
3. *Parent dies → children notice.*
   [`kill_itself_when_parent_died`](../../python/sglang/srt/utils/common.py#L3121) uses
   `prctl(PR_SET_PDEATHSIG, SIGKILL)` on Linux and a kqueue watchdog on macOS. Called first
   thing in every child entry point.

All three converge on the `SIGQUIT` handler installed at
[engine.py:1733-1739](../../python/sglang/srt/entrypoints/engine.py#L1733-L1739), which calls
`kill_process_tree(os.getpid())`. PIDs for that teardown are accumulated in
`SchedulerInitResult.all_child_pids`
([engine.py:949](../../python/sglang/srt/entrypoints/engine.py#L949),
[engine.py:1217-1218](../../python/sglang/srt/entrypoints/engine.py#L1217-L1218)).

---

## 11. Variants that skip parts of this

| Mode | Difference |
|---|---|
| `node_rank >= 1` | Launches its slice of schedulers, then either returns immediately (`SGLANG_BLOCK_NONZERO_RANK_CHILDREN=0`) or serves a dummy health-check server and blocks — no detokenizer, no tokenizer manager ([engine.py:1158-1185](../../python/sglang/srt/entrypoints/engine.py#L1158-L1185)) |
| `SGLANG_RUST_SERVER=1` | The Rust server inside the rank-0 scheduler owns HTTP, tokenization, and detokenization; no Python detokenizer or tokenizer-manager processes ([engine.py:1191-1208](../../python/sglang/srt/entrypoints/engine.py#L1191-L1208)). Rejected for `sgl.Engine` at [engine.py:256-261](../../python/sglang/srt/entrypoints/engine.py#L256-L261) |
| `--use-ray` | `_launch_scheduler_processes` is overridden to create Ray actors; `scheduler_procs` comes back `None` and the watchdog has nothing to poll ([ray/engine.py:263](../../python/sglang/srt/ray/engine.py#L263), [ray/data_parallel_controller.py:49](../../python/sglang/srt/ray/data_parallel_controller.py#L49)) |
| `sgl.Engine(...)` | Identical process tree minus the uvicorn/HTTP layer; `atexit.register(self.shutdown)` at [engine.py:268](../../python/sglang/srt/entrypoints/engine.py#L268) replaces the server's teardown |

---

## 12. Reading the tree at runtime

Every child sets `setproctitle`, so `ps` shows the topology directly:

```
sglang::data_parallel_controller     data_parallel_controller.py:817
sglang::scheduler_DP0_TP1            scheduler.py:5120
sglang::detokenizer                  detokenizer_manager.py:522
sglang::detokenizer_router           multi_tokenizer_mixin.py:631
sglang::tokenizer_worker:<pid>       multi_tokenizer_mixin.py:655
```
